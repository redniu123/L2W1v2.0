#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v5.1 Phase 3: Efficiency Frontier Grand Loop

评测数据集: Test 集
目标预算: [0.05, 0.10, 0.20, 0.30]
策略: AgentA_Only / Random / ConfOnly / SH-DA++
输出: results/stage2_v51/efficiency_frontier.csv
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import Levenshtein


def compute_cer(T_final: str, T_GT: str) -> float:
    if not T_GT:
        return 0.0
    return Levenshtein.distance(T_final, T_GT) / len(T_GT)


def build_agent_b_callable(config: dict, max_retries: int = 3) -> Callable:
    """构建 Agent B 调用函数（含重试）"""
    import re
    agent_b_cfg = config.get("agent_b", {})
    skip = agent_b_cfg.get("skip", True)
    backend = agent_b_cfg.get("backend", "qwen")

    if skip:
        print("[Agent B] skip=true，Mock 模式")
        def mock_fn(prompt: dict) -> str:
            user_prompt = prompt.get("user_prompt", "")
            m = re.search(r'\u3010\s*(.+?)\s*\u3011', user_prompt)
            return m.group(1).strip() if m else prompt.get("T_A", "")
        return mock_fn

    # Gemini 后端
    if backend == "gemini":
        try:
            from modules.vlm_expert.gemini_expert import GeminiAgentB, GeminiConfig
            agent = GeminiAgentB(config=GeminiConfig())
            print(f"[Agent B] Gemini backend: {agent.config.model_name}")

            def gemini_fn(prompt: dict) -> str:
                T_A = prompt.get("T_A", "")
                image_path = prompt.get("image_path", "")
                min_conf_idx = prompt.get("min_conf_idx", -1)
                if min_conf_idx is None:
                    min_conf_idx = -1
                suspicious_char = T_A[min_conf_idx] if (0 <= min_conf_idx < len(T_A)) else ""
                manifest = {
                    "ocr_text": T_A,
                    "suspicious_index": min_conf_idx,
                    "suspicious_char": suspicious_char,
                    "risk_level": "medium",
                }
                try:
                    result = agent.process_hard_sample(image_path, manifest)
                    return result["corrected_text"]
                except Exception as e:
                    print(f"[Gemini] error: {e}")
                    return T_A

            return gemini_fn
        except Exception as e:
            print(f"[Agent B] Gemini load failed: {e}, fallback to mock")
            def mock_fn(prompt: dict) -> str:
                return prompt.get("T_A", "")
            return mock_fn

    # Qwen 后端
    model_path = agent_b_cfg.get("model_path") or agent_b_cfg.get("model_name", "Qwen/Qwen2.5-VL-3B-Instruct")
    try:
        from modules.vlm_expert.agent_b_expert import AgentBExpert, AgentBConfig
        agent = AgentBExpert(config=AgentBConfig(model_path=model_path), lazy_init=False)
        print(f"[Agent B] model loaded: {model_path}")

        def real_fn(prompt: dict) -> str:
            T_A = prompt.get("T_A", "")
            image_path = prompt.get("image_path", "")
            min_conf_idx = prompt.get("min_conf_idx", -1)
            if min_conf_idx is None:
                min_conf_idx = -1
            suspicious_char = T_A[min_conf_idx] if (0 <= min_conf_idx < len(T_A)) else ""
            manifest = {
                "ocr_text": T_A,
                "suspicious_index": min_conf_idx,
                "suspicious_char": suspicious_char,
                "risk_level": "medium",
            }
            for attempt in range(max_retries):
                try:
                    result = agent.process_hard_sample(image_path, manifest)
                    return result["corrected_text"]
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(1.0 * (attempt + 1))
                    else:
                        print(f"[Agent B] error: {e}")
            return T_A

        return real_fn
    except Exception as e:
        print(f"[Agent B] load failed: {e}, fallback to mock")

        def mock_fn(prompt: dict) -> str:
            return prompt.get("T_A", "")

        return mock_fn


def infer_all_samples(samples, recognizer, domain_engine, data_root, image_root):
    """全量 Agent A 推理，提取 5 维特征"""
    import cv2
    results = []
    for sample in tqdm(samples, desc="Agent A 全量推理"):
        image_path = sample.get("image") or sample.get("image_path", "")
        T_GT = sample.get("gt_text") or sample.get("text") or sample.get("label", "")
        if not image_path or not T_GT:
            continue
        img_path = Path(image_path)
        if not img_path.is_absolute():
            img_path = Path(image_root).resolve() / img_path
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        try:
            output = recognizer([img])
            if not output or not output.get("results"):
                continue
            T_A, conf = output["results"][0]
        except Exception:
            continue

        top2_info_list = output.get("top2_info", [])
        top2_info = top2_info_list[0] if top2_info_list else {}
        boundary_stats_list = output.get("boundary_stats", [])
        boundary_stats = boundary_stats_list[0] if boundary_stats_list else {}

        top1_probs = (top2_info or {}).get("top1_probs") or []
        mean_conf = float(np.mean(top1_probs)) if top1_probs else float(conf)
        min_conf = float(np.min(top1_probs)) if top1_probs else float(conf)
        min_conf_idx = int(np.argmin(top1_probs)) if top1_probs else None

        bl = float((boundary_stats or {}).get("blank_mean_L", 0.0))
        br = float((boundary_stats or {}).get("blank_mean_R", 0.0))
        pl = float((boundary_stats or {}).get("blank_peak_L", 0.0))
        pr = float((boundary_stats or {}).get("blank_peak_R", 0.0))
        b_edge = max(0.6*bl + 0.4*pl, 0.6*br + 0.4*pr)
        drop = float(np.clip(abs(bl - br), 0.0, 1.0))
        r_d = domain_engine.compute_r_d(T_A) if domain_engine else 0.0

        results.append({
            "image_path": str(image_path),
            "img_path": str(img_path),
            "T_A": T_A,
            "T_GT": T_GT,
            "conf": float(conf),
            "mean_conf": mean_conf,
            "min_conf": min_conf,
            "min_conf_idx": min_conf_idx,
            "b_edge": b_edge,
            "drop": drop,
            "r_d": r_d,
            "top2_info": top2_info,
            "boundary_stats": boundary_stats,
        })
    return results












def run_pipeline(
    strategy, target_budget, all_results,
    router, backfill_controller, prompter, agent_b_callable,
):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    N = len(all_results)
    if N == 0:
        return None
    n_call_target = int(round(N * target_budget))

    if strategy == 'AgentA_Only':
        cer_num = sum(Levenshtein.distance(r['T_A'], r['T_GT']) for r in all_results)
        gt_len = sum(len(r['T_GT']) for r in all_results)
        return {
            'Strategy': 'AgentA_Only', 'Target_Budget': 0.0,
            'Actual_Call_Rate': 0.0,
            'Overall_CER': round(cer_num / gt_len, 6) if gt_len else 0,
            'AER': 0.0, 'CVR': 0.0, 'N_valid': N,
        }

    # 计算路由分数
    if strategy == 'Random':
        scores = [random.random() for _ in range(N)]
    elif strategy == 'ConfOnly':
        scores = [(1.0 - r['mean_conf']) + (1.0 - r['min_conf']) + r['drop']
                  for r in all_results]
    else:
        scores = []
        for r in all_results:
            dec = router.route(
                boundary_stats=r['boundary_stats'],
                top2_info=r['top2_info'],
                r_d=r['r_d'],
                agent_a_text=r['T_A'],
            )
            scores.append(dec.q)

    upgrade_set = set(
        sorted(range(N), key=lambda i: scores[i], reverse=True)[:n_call_target]
    )

    # 批量并发调用 Agent B
    from modules.router.backfill import RouteType
    upgrade_results = {}  # {index: T_cand}
    
    if upgrade_set:
        # 自动根据 Key 数量设置并发数
        try:
            n_keys = agent_b_callable.__self__.config.key_manager.get_key_count()
        except Exception:
            n_keys = 10
        print(f"  [{strategy}] Calling Agent B for {len(upgrade_set)} samples ({n_keys} concurrent)...")
        
        def call_agent_b(idx, r):
            """单个样本的 Agent B 调用，无等待"""
            prompt = prompter.generate_targeted_correction_prompt(
                T_A=r['T_A'], min_conf_idx=r['min_conf_idx'],
                domain='地质勘探', image_path=r['img_path'],
            )
            prompt['T_A'] = r['T_A']
            return idx, agent_b_callable(prompt)
        
        # 并发调用（Key 数量个线程，每个线程独占一个 Key）
        with ThreadPoolExecutor(max_workers=n_keys) as executor:
            futures = {
                executor.submit(call_agent_b, i, all_results[i]): i
                for i in upgrade_set
            }
            
            # 收集结果（带进度条）
            for future in tqdm(as_completed(futures), total=len(upgrade_set), 
                              desc=f'{strategy} B={target_budget:.2f} [API]', leave=False):
                try:
                    idx, T_cand = future.result()
                    upgrade_results[idx] = T_cand
                except Exception as e:
                    idx = futures[future]
                    upgrade_results[idx] = all_results[idx]['T_A']  # 失败降级
    
    # 回填与统计
    cer_num = 0
    gt_len = 0
    n_upgraded = 0
    n_accepted_edit = 0
    n_rejected = 0
    
    for i, r in enumerate(all_results):
        T_A = r['T_A']
        T_GT = r['T_GT']
        
        if i in upgrade_set:
            n_upgraded += 1
            T_cand = upgrade_results.get(i, T_A)
            bf = backfill_controller.apply_backfill(
                T_A=T_A, T_cand=T_cand, route_type=RouteType.BOUNDARY,
            )
            T_final = bf.T_final
            if bf.is_rejected:
                n_rejected += 1
            elif T_final != T_A:
                n_accepted_edit += 1
        else:
            T_final = T_A
        
        cer_num += Levenshtein.distance(T_final, T_GT)
        gt_len += len(T_GT)

    actual_rate = n_upgraded / N if N > 0 else 0.0
    overall_cer = cer_num / gt_len if gt_len > 0 else 0.0
    aer = n_accepted_edit / n_upgraded if n_upgraded > 0 else 0.0
    cvr = n_rejected / n_upgraded if n_upgraded > 0 else 0.0
    
    return {
        'Strategy': strategy, 'Target_Budget': target_budget,
        'Actual_Call_Rate': round(actual_rate, 4),
        'Overall_CER': round(overall_cer, 6),
        'AER': round(aer, 4), 'CVR': round(cvr, 4), 'N_valid': N,
    }


def main():
    parser = argparse.ArgumentParser(description='SH-DA++ v5.1 Phase 3')
    parser.add_argument('--config', default='configs/router_config.yaml')
    parser.add_argument('--test_jsonl', default='data/raw/hctr_riskbench/test.jsonl')
    parser.add_argument('--image_root', default='data/geo')
    parser.add_argument('--rec_model_dir', default='./models/agent_a_ppocr/PP-OCRv5_server_rec_infer')
    parser.add_argument('--rec_char_dict_path', default='ppocr/utils/ppocrv5_dict.txt')
    parser.add_argument('--geo_dict', default='data/dicts/Geology.txt')
    parser.add_argument('--output_dir', default='results/stage2_v51')
    parser.add_argument('--budgets', default='0.05,0.10,0.20,0.30')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_samples', type=int, default=None, help='Limit to first N samples (for testing)')
    parser.add_argument('--use_gpu', action='store_true', default=False, help='Use GPU for Agent A (default: CPU)')
    parser.add_argument('--use_cache', action='store_true', default=False, help='Load Agent A results from cache instead of re-inferring')
    parser.add_argument('--rebuild_cache', action='store_true', default=False, help='Force rebuild Agent A cache even if it exists')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    print('[1/4] Init Agent A...')
    print(f'  Agent A device: {"GPU" if args.use_gpu else "CPU"}')
    import argparse as _ap
    rec_args = _ap.Namespace(
        rec_model_dir=args.rec_model_dir,
        rec_char_dict_path=args.rec_char_dict_path,
        rec_image_shape='3, 48, 320', rec_batch_num=6,
        rec_algorithm='SVTR_LCNet', use_space_char=True,
        use_gpu=args.use_gpu, use_xpu=False, use_npu=False, use_mlu=False,
        use_metax_gpu=False, use_gcu=False, ir_optim=True,
        use_tensorrt=False, min_subgraph_size=15, precision='fp32',
        gpu_mem=500, gpu_id=0, enable_mkldnn=None, cpu_threads=10,
        warmup=False, benchmark=False, save_log_path='./log_output/',
        show_log=False, use_onnx=False, max_batch_size=10,
        return_word_box=False, drop_score=0.5, max_text_length=25,
        rec_image_inverse=True, use_det=False, det_model_dir='',
    )
    from modules.paddle_engine.predict_rec_modified import TextRecognizerWithLogits
    recognizer = TextRecognizerWithLogits(rec_args)
    print('[2/4] Init Router and components...')
    from modules.router.sh_da_router import SHDARouter
    from modules.router.backfill import BackfillConfig, StrictBackfillController
    from modules.vlm_expert.constrained_prompter import ConstrainedPrompter
    from modules.router.domain_knowledge import DomainKnowledgeEngine
    router = SHDARouter.from_yaml(args.config)
    backfill_controller = StrictBackfillController(BackfillConfig())
    prompter = ConstrainedPrompter()
    domain_engine = DomainKnowledgeEngine(args.geo_dict)
    agent_b_callable = build_agent_b_callable(config)
    print('[3/4] Load test set...')
    samples = []
    with open(args.test_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f'  Test: {len(samples)} samples')

    # Agent A 推理缓存：避免每次重复推理
    cache_path = Path(args.output_dir) / 'agent_a_cache.json'
    if args.use_cache and not args.rebuild_cache and cache_path.exists():
        print(f'[4/4] Agent A cache HIT: loading from {cache_path}')
        with open(cache_path, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
        print(f'  Loaded {len(all_results)} cached results')
    else:
        print('[4/4] Agent A full inference...')
        all_results = infer_all_samples(samples, recognizer, domain_engine, None, args.image_root)
        # 保存缓存（完整集，不限制 n_samples）
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        # top2_info/boundary_stats 中可能含 numpy，序列化前转 list
        cache_data = []
        for r in all_results:
            rc = dict(r)
            for k in ('top2_info', 'boundary_stats'):
                if isinstance(rc.get(k), dict):
                    rc[k] = {kk: (vv.tolist() if hasattr(vv, 'tolist') else vv)
                             for kk, vv in rc[k].items()}
            cache_data.append(rc)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False)
        print(f'  Agent A cache saved: {cache_path} ({len(all_results)} samples)')

    # 限制样本数（用于快速测试）
    if args.n_samples and args.n_samples < len(all_results):
        all_results = all_results[:args.n_samples]
        print(f'  Limited to first {args.n_samples} samples for testing')

    print(f'  Valid: {len(all_results)}')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / 'efficiency_frontier.csv'
    fieldnames = ['Strategy', 'Target_Budget', 'Actual_Call_Rate',
                  'Overall_CER', 'AER', 'CVR', 'N_valid']
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        budgets = [float(b) for b in args.budgets.split(',')]
        
        print('\n--- AgentA_Only (Baseline 0) ---')
        row = run_pipeline('AgentA_Only', 0.0, all_results, router,
                           backfill_controller, prompter, agent_b_callable)
        writer.writerow({k: row.get(k, '') for k in fieldnames})
        csvfile.flush()
        print(f"  CER={row['Overall_CER']:.4%}")
        
        # 并行处理 3 个策略
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        for B in budgets:
            print(f'\n=== Budget B={B:.2f} ===')
            tasks = []
            with ThreadPoolExecutor(max_workers=3) as executor:
                for strategy in ['Random', 'ConfOnly', 'SH-DA++']:
                    future = executor.submit(
                        run_pipeline, strategy, B, all_results, router,
                        backfill_controller, prompter, agent_b_callable
                    )
                    tasks.append((strategy, future))
                
                for strategy, future in tasks:
                    try:
                        row = future.result()
                        writer.writerow({k: row.get(k, '') for k in fieldnames})
                        csvfile.flush()
                        print(
                            f"  [{strategy:10s}] CER={row['Overall_CER']:.4%}"
                            f"  AER={row['AER']:.2%}"
                            f"  CVR={row['CVR']:.2%}"
                            f"  ActualRate={row['Actual_Call_Rate']:.2%}"
                        )
                    except Exception as e:
                        print(f"  [{strategy:10s}] ERROR: {e}")
    
    print(f'\nDone: {csv_path}')


if __name__ == '__main__':
    main()
