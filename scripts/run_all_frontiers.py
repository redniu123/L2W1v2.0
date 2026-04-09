#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v5.1 Task 1: Multi-Model Pareto Grand Loop

依次跑完所有模型和预算点，输出:
  results/stage2_v51/master_efficiency_frontier.csv

用法:
  # 云服务器挂机运行
  nohup python scripts/run_all_frontiers.py > logs/run_all_frontiers.log 2>&1 &

  # 快速测试 (前 50 个样本)
  python scripts/run_all_frontiers.py --n_samples 50
"""

import argparse
import csv
import gc
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import yaml
from tqdm import tqdm

# 默认使用第三张显卡
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import Levenshtein

# ============================================================
# 模型配置表（6 款模型）
# ============================================================
MODEL_CONFIGS = [
    {
        "key": "smolvlm",
        "label": "SmolVLM-500M",
        "size_b": 0.5,
        "backend": "local_vlm",
        "model_type": "smolvlm",
        "model_path": "./models/agent_b_vlm/SmolVLM-500M-Instruct",
        "torch_dtype": "float16",
    },
    {
        "key": "minicpm_v",
        "label": "MiniCPM-V-2.6-int4",
        "size_b": 8.0,
        "backend": "local_vlm",
        "model_type": "minicpm_v",
        "model_path": "./models/agent_b_vlm/MiniCPM-V-2_6-int4",
        "torch_dtype": "float16",
    },
    {
        "key": "llava",
        "label": "LLaVA-1.5-7B",
        "size_b": 7.0,
        "backend": "local_vlm",
        "model_type": "llava",
        "model_path": "./models/agent_b_vlm/llava-1.5-7b-hf",
        "torch_dtype": "float16",
    },
    {
        "key": "qwen2.5_vl",
        "label": "Qwen2.5-VL-7B",
        "size_b": 7.0,
        "backend": "local_vlm",
        "model_type": "qwen2.5_vl",
        "model_path": "./models/agent_b_vlm/Qwen2.5-VL-7B-Instruct",
        "torch_dtype": "float16",
    },
    {
        "key": "qwen3.5",
        "label": "Qwen3.5-9B",
        "size_b": 9.0,
        "backend": "local_vlm",
        "model_type": "qwen3.5",
        "model_path": "./models/agent_b_vlm/Qwen3.5-9B",
        "torch_dtype": "float16",
    },
    {
        "key": "gemini",
        "label": "Gemini-3-Flash",
        "size_b": 999,  # Cloud API
        "backend": "gemini",
        "model_type": "gemini",
        "model_path": "",
        "torch_dtype": "float16",
    },
]

BUDGETS = [0.05, 0.10, 0.20, 0.30]


# ============================================================
# CER 计算
# ============================================================
def compute_cer(T_final: str, T_GT: str) -> float:
    if not T_GT:
        return 0.0
    return Levenshtein.distance(T_final, T_GT) / len(T_GT)


# ============================================================
# Agent B 构建
# ============================================================
def build_agent_b(model_cfg: dict, base_config: dict) -> Callable:
    """根据模型配置构建 Agent B callable"""
    backend = model_cfg["backend"]

    if backend == "gemini":
        try:
            from modules.vlm_expert.gemini_expert import GeminiAgentB, GeminiConfig
            agent = GeminiAgentB(config=GeminiConfig())
            print(f"  [AgentB] Gemini: {agent.config.model_name}")

            def gemini_fn(prompt: dict) -> str:
                T_A = prompt.get("T_A", "")
                image_path = prompt.get("image_path", "")
                min_conf_idx = prompt.get("min_conf_idx", -1) or -1
                suspicious_char = T_A[min_conf_idx] if 0 <= min_conf_idx < len(T_A) else ""
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
                    print(f"  [Gemini] error: {e}")
                    return T_A
            return gemini_fn
        except Exception as e:
            print(f"  [AgentB] Gemini load failed: {e}, using mock")
            return lambda p: p.get("T_A", "")

    # 本地 VLM
    cfg_override = dict(base_config)
    cfg_override["agent_b"] = {
        "skip": False,
        "backend": "local_vlm",
        "model_type": model_cfg["model_type"],
        "model_path": model_cfg["model_path"],
        "torch_dtype": model_cfg["torch_dtype"],
        "max_new_tokens": 128,
    }

    from modules.vlm_expert import AgentBFactory
    # AgentBFactory.create() 接收顶层 config（含 agent_b 节点）
    expert = AgentBFactory.create(cfg_override)
    print(f"  [AgentB] {model_cfg['label']} loaded")

    def local_fn(prompt: dict) -> str:
        T_A = prompt.get("T_A", "")
        image_path = prompt.get("img_path", prompt.get("image_path", ""))
        min_conf_idx = prompt.get("min_conf_idx", -1) or -1
        suspicious_char = T_A[min_conf_idx] if 0 <= min_conf_idx < len(T_A) else ""
        manifest = {
            "ocr_text": T_A,
            "suspicious_index": min_conf_idx,
            "suspicious_char": suspicious_char,
            "risk_level": "medium",
        }
        try:
            result = expert.process_hard_sample(image_path, manifest)
            # process_hard_sample 返回 dict，提取 corrected_text
            if isinstance(result, dict):
                return result.get('corrected_text', T_A)
            return result if isinstance(result, str) else T_A
        except Exception as e:
            print(f"  [{model_cfg['key']}] error: {e}")
            return T_A

    return local_fn


def release_model(model_cfg: dict):
    """释放模型显存"""
    if model_cfg["backend"] == "local_vlm":
        import torch
        torch.cuda.empty_cache()
        gc.collect()
        print(f"  [VRAM] Released after {model_cfg['label']}")


# ============================================================
# Agent A 全量推理
# ============================================================
def infer_all_samples(samples, recognizer, domain_engine, image_root):
    """全量 Agent A 推理，提取 5 维特征"""
    import cv2
    results = []
    for sample in tqdm(samples, desc="Agent A 全量推理"):
        image_path = sample.get("image") or sample.get("image_path", "")
        T_GT = sample.get("gt_text") or sample.get("gt") or sample.get("text") or sample.get("label", "")
        if not image_path or not T_GT:
            continue
        img_path = Path(image_path)
        if not img_path.is_absolute():
            rel_path = Path(image_path)
            if rel_path.parts[:2] == ('dataset', 'images'):
                rel_path = Path(*rel_path.parts[2:])
            img_path = Path(image_root).resolve() / rel_path
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
        b_edge = max(0.6 * bl + 0.4 * pl, 0.6 * br + 0.4 * pr)
        drop = float(np.clip(abs(bl - br), 0.0, 1.0))
        r_d = domain_engine.compute_r_d(T_A, domain=sample.get("domain", "geology")) if domain_engine else 0.0

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


# ============================================================
# 单策略评测
# ============================================================
def run_pipeline(
    strategy, target_budget, all_results,
    router, backfill_controller, prompter, agent_b_callable,
    model_label="",
):
    N = len(all_results)
    if N == 0:
        return None
    n_call_target = int(round(N * target_budget))

    if strategy == "AgentA_Only":
        cer_num = sum(Levenshtein.distance(r["T_A"], r["T_GT"]) for r in all_results)
        gt_len = sum(len(r["T_GT"]) for r in all_results)
        return {
            "Strategy": "AgentA_Only", "Target_Budget": 0.0,
            "Actual_Call_Rate": 0.0,
            "Overall_CER": round(cer_num / gt_len, 6) if gt_len else 0,
            "AER": 0.0, "CVR": 0.0, "N_valid": N,
        }

    # 计算路由分数
    if strategy == "GCR":
        scores = [1.0 - r["conf"] for r in all_results]
    elif strategy == "BAUR":
        scores = []
        for r in all_results:
            dec = router.route(
                boundary_stats=r["boundary_stats"],
                top2_info=r["top2_info"],
                r_d=0.0,
                agent_a_text=r["T_A"],
            )
            scores.append(max(dec.s_b, dec.s_a))
    elif strategy == "DAR":
        scores = []
        for r in all_results:
            dec = router.route(
                boundary_stats=r["boundary_stats"],
                top2_info=r["top2_info"],
                r_d=r["r_d"],
                agent_a_text=r["T_A"],
            )
            scores.append(dec.q)
    else:  # SH-DA++
        scores = []
        for r in all_results:
            dec = router.route(
                boundary_stats=r["boundary_stats"],
                top2_info=r["top2_info"],
                r_d=r["r_d"],
                agent_a_text=r["T_A"],
            )
            scores.append(dec.q)

    upgrade_set = set(
        sorted(range(N), key=lambda i: scores[i], reverse=True)[:n_call_target]
    )

    # 调用 Agent B
    upgrade_results = {}
    if upgrade_set:
        print(f"    [{strategy}] Calling Agent B for {len(upgrade_set)} samples...")
        for idx in tqdm(sorted(upgrade_set),
                        desc=f"{model_label} {strategy} B={target_budget:.2f}",
                        leave=False):
            r = all_results[idx]
            domain_label = {
                'geology': '地质勘探',
                'finance': '金融财会',
                'medicine': '医学',
            }.get(r.get('domain', 'geology'), '地质勘探')
            prompt = prompter.generate_targeted_correction_prompt(
                T_A=r["T_A"],
                min_conf_idx=r["min_conf_idx"],
                domain=domain_label,
                image_path=r["img_path"],
            )
            prompt["T_A"] = r["T_A"]
            try:
                upgrade_results[idx] = agent_b_callable(prompt)
            except Exception as e:
                print(f"    [WARN] Agent B failed for idx={idx}: {e}")
                upgrade_results[idx] = r["T_A"]
            # API 限流保护
            if strategy == "SH-DA++":
                time.sleep(0.05)

    # 回填与统计
    from modules.router.backfill import RouteType
    cer_num = 0
    gt_len = 0
    n_upgraded = 0
    n_accepted_edit = 0
    n_rejected = 0

    for i, r in enumerate(all_results):
        T_A = r["T_A"]
        T_GT = r["T_GT"]
        if i in upgrade_set:
            n_upgraded += 1
            T_cand = upgrade_results.get(i, T_A)
            if not isinstance(T_cand, str) or not T_cand:
                T_cand = T_A
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
        "Strategy": strategy, "Target_Budget": target_budget,
        "Actual_Call_Rate": round(actual_rate, 4),
        "Overall_CER": round(overall_cer, 6),
        "AER": round(aer, 4), "CVR": round(cvr, 4), "N_valid": N,
    }


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="SH-DA++ v5.1 Multi-Model Grand Loop")
    parser.add_argument("--config", default="configs/router_config.yaml")
    parser.add_argument("--test_jsonl", default="data/l2w1data/test.jsonl")
    parser.add_argument("--image_root", default="data/l2w1data/images")
    parser.add_argument("--rec_model_dir", default="./models/agent_a_ppocr/PP-OCRv5_server_rec_infer")
    parser.add_argument("--rec_char_dict_path", default="ppocr/utils/ppocrv5_dict.txt")
    parser.add_argument("--geo_dict", default="data/dicts/Geology.txt")
    parser.add_argument("--finance_dict", default="data/dicts/Finance.txt")
    parser.add_argument("--medicine_dict", default="data/dicts/Medicine.txt")
    parser.add_argument("--output_dir", default="results/stage2_v51")
    parser.add_argument("--budgets", default="0.05,0.10,0.20,0.30")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--models", default="smolvlm,minicpm_v,llava,qwen2.5_vl,qwen3.5,gemini")
    parser.add_argument("--skip_baselines", action="store_true")
    parser.add_argument("--use_cache", action="store_true", default=False, help="Load Agent A results from cache")
    parser.add_argument("--rebuild_cache", action="store_true", default=False, help="Force rebuild Agent A cache")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config, "r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    budgets = [float(b) for b in args.budgets.split(",")]
    model_keys = [k.strip() for k in args.models.split(",")]
    model_cfgs = [m for m in MODEL_CONFIGS if m["key"] in model_keys]

    print("[1/4] Init Agent A...")
    import argparse as _ap
    rec_args = _ap.Namespace(
        rec_model_dir=args.rec_model_dir, rec_char_dict_path=args.rec_char_dict_path,
        rec_image_shape="3, 48, 320", rec_batch_num=6, rec_algorithm="SVTR_LCNet",
        use_space_char=True, use_gpu=True, use_xpu=False, use_npu=False, use_mlu=False,
        use_metax_gpu=False, use_gcu=False, ir_optim=True, use_tensorrt=False,
        min_subgraph_size=15, precision="fp32", gpu_mem=500, gpu_id=0,
        enable_mkldnn=None, cpu_threads=10, warmup=False, benchmark=False,
        save_log_path="./log_output/", show_log=False, use_onnx=False,
        max_batch_size=10, return_word_box=False, drop_score=0.5,
        max_text_length=25, rec_image_inverse=True, use_det=False, det_model_dir="",
    )
    from modules.paddle_engine.predict_rec_modified import TextRecognizerWithLogits
    recognizer = TextRecognizerWithLogits(rec_args)

    print("[2/4] Init Router and components...")
    from modules.router.sh_da_router import SHDARouter
    from modules.router.backfill import BackfillConfig, StrictBackfillController
    from modules.vlm_expert.constrained_prompter import ConstrainedPrompter
    from modules.router.domain_knowledge import DomainKnowledgeEngine
    router = SHDARouter.from_yaml(args.config)
    backfill_controller = StrictBackfillController(BackfillConfig())
    prompter = ConstrainedPrompter()
    domain_engine = DomainKnowledgeEngine({
        'geology': args.geo_dict,
        'finance': args.finance_dict,
        'medicine': args.medicine_dict,
    })

    print("[3/4] Load test set...")
    samples = []
    with open(args.test_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"  Test: {len(samples)} samples")

    # Agent A 推理缓存
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "agent_a_cache.json"
    if args.use_cache and not args.rebuild_cache and cache_path.exists():
        print(f"[4/4] Agent A cache HIT: {cache_path}")
        import json as _json
        with open(cache_path, "r", encoding="utf-8") as f:
            all_results = _json.load(f)
        print(f"  Loaded {len(all_results)} cached results")
    else:
        print("[4/4] Agent A full inference...")
        all_results = infer_all_samples(samples, recognizer, domain_engine, args.image_root)
        cache_data = []
        for r in all_results:
            rc = dict(r)
            for k in ("top2_info", "boundary_stats"):
                if isinstance(rc.get(k), dict):
                    rc[k] = {kk: (vv.tolist() if hasattr(vv, "tolist") else vv) for kk, vv in rc[k].items()}
            cache_data.append(rc)
        import json as _json
        with open(cache_path, "w", encoding="utf-8") as f:
            _json.dump(cache_data, f, ensure_ascii=False)
        print(f"  Agent A cache saved: {cache_path}")
    if args.n_samples and args.n_samples < len(all_results):
        all_results = all_results[:args.n_samples]
        print(f"  Limited to {args.n_samples} samples")
    print(f"  Valid: {len(all_results)}")
    csv_path = output_dir / "master_efficiency_frontier.csv"
    fieldnames = ["Model", "Strategy", "Target_Budget", "Actual_Call_Rate",
                  "Overall_CER", "AER", "CVR", "N_valid"]
    write_header = not csv_path.exists()
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    def write_row(model_key, row):
        row["Model"] = model_key
        writer.writerow({k: row.get(k, "") for k in fieldnames})
        csv_file.flush()

    if not args.skip_baselines:
        print("\n" + "="*60 + "\n  BASELINES\n" + "="*60)
        row = run_pipeline("AgentA_Only", 0.0, all_results, router,
                           backfill_controller, prompter, lambda p: p.get("T_A", ""))
        write_row("baseline", row)
        print(f"  AgentA_Only  CER={row['Overall_CER']:.4%}")
        for B in budgets:
            for strategy in ["GCR", "BAUR", "DAR", "SH-DA++"]:
                row = run_pipeline(strategy, B, all_results, router,
                                   backfill_controller, prompter,
                                   lambda p: p.get("T_A", ""), "baseline")
                write_row("baseline", row)
                print(f"  [{strategy:10s}] B={B:.2f}  CER={row['Overall_CER']:.4%}")

    for model_cfg in model_cfgs:
        print("\n" + "="*60 + f"\n  MODEL: {model_cfg['label']}\n" + "="*60)
        try:
            agent_b_callable = build_agent_b(model_cfg, base_config)
        except Exception as e:
            print(f"  [ERROR] Failed to load {model_cfg['label']}: {e}")
            continue
        for B in budgets:
            print(f"  -- Budget B={B:.2f} --")
            row = run_pipeline("SH-DA++", B, all_results, router,
                               backfill_controller, prompter, agent_b_callable,
                               model_cfg["label"])
            if row:
                write_row(model_cfg["key"], row)
                print(f"  [SH-DA++ {model_cfg['label']:20s}] B={B:.2f} "
                      f"CER={row['Overall_CER']:.4%} AER={row['AER']:.2%} "
                      f"CVR={row['CVR']:.2%}")
        release_model(model_cfg)
        time.sleep(2.0)

    csv_file.close()
    print(f"\nDone! Results: {csv_path}")
    print("Next: python scripts/visualize_master_frontier.py")


if __name__ == "__main__":
    main()
