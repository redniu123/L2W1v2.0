#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v5.1 Phase 3: Efficiency Frontier Grand Loop

评测数据集: Test 集
目标预算: [0.05, 0.10, 0.20, 0.30]
策略: AgentA_Only / GCR / BAUR / DAR / BAUR-only / SH-DA++
输出: results/stage2_v51/efficiency_frontier.csv
"""

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
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
        def mock_fn(prompt: dict) -> dict:
            t0 = time.perf_counter()
            user_prompt = prompt.get("user_prompt", "")
            m = re.search(r'\u3010\s*(.+?)\s*\u3011', user_prompt)
            corrected_text = m.group(1).strip() if m else prompt.get("T_A", "")
            return {
                "corrected_text": corrected_text,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
                "token_usage": None,
                "error_type": "none",
            }
        return mock_fn

    # Gemini 后端
    if backend == "gemini":
        try:
            from modules.vlm_expert.gemini_expert import GeminiAgentB, GeminiConfig
            agent = GeminiAgentB(config=GeminiConfig())
            print(f"[Agent B] Gemini backend: {agent.config.model_name}")

            def gemini_fn(prompt: dict) -> dict:
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
                t0 = time.perf_counter()
                try:
                    result = agent.process_hard_sample(image_path, manifest)
                    return {
                        "corrected_text": result.get("corrected_text", T_A),
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
                        "token_usage": result.get("token_usage"),
                        "error_type": result.get("error_type", "none"),
                    }
                except Exception as e:
                    print(f"[Gemini] error: {e}")
                    return {
                        "corrected_text": T_A,
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
                        "token_usage": None,
                        "error_type": type(e).__name__,
                    }

            return gemini_fn
        except Exception as e:
            print(f"[Agent B] Gemini load failed: {e}, fallback to mock")
            def mock_fn(prompt: dict) -> dict:
                t0 = time.perf_counter()
                return {
                    "corrected_text": prompt.get("T_A", ""),
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
                    "token_usage": None,
                    "error_type": "gemini_load_failed",
                }
            return mock_fn

    # Qwen 后端
    model_path = agent_b_cfg.get("model_path") or agent_b_cfg.get("model_name", "Qwen/Qwen2.5-VL-3B-Instruct")
    try:
        from modules.vlm_expert.agent_b_expert import AgentBExpert, AgentBConfig
        agent = AgentBExpert(config=AgentBConfig(model_path=model_path), lazy_init=False)
        print(f"[Agent B] model loaded: {model_path}")

        def real_fn(prompt: dict) -> dict:
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
            t0 = time.perf_counter()
            last_error_type = "none"
            for attempt in range(max_retries):
                try:
                    result = agent.process_hard_sample(image_path, manifest)
                    return {
                        "corrected_text": result.get("corrected_text", T_A),
                        "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
                        "token_usage": result.get("token_usage"),
                        "error_type": result.get("error_type", "none"),
                    }
                except Exception as e:
                    last_error_type = type(e).__name__
                    if attempt < max_retries - 1:
                        time.sleep(1.0 * (attempt + 1))
                    else:
                        print(f"[Agent B] error: {e}")
            return {
                "corrected_text": T_A,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
                "token_usage": None,
                "error_type": last_error_type,
            }

        return real_fn
    except Exception as e:
        print(f"[Agent B] load failed: {e}, fallback to mock")

        def mock_fn(prompt: dict) -> dict:
            t0 = time.perf_counter()
            return {
                "corrected_text": prompt.get("T_A", ""),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
                "token_usage": None,
                "error_type": "agent_b_load_failed",
            }

        return mock_fn


def infer_all_samples(samples, recognizer, domain_engine, data_root, image_root):
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
        b_edge = max(0.6*bl + 0.4*pl, 0.6*br + 0.4*pr)
        drop = float(np.clip(abs(bl - br), 0.0, 1.0))
        r_d = domain_engine.compute_r_d(T_A, domain=sample.get("domain", "geology")) if domain_engine else 0.0

        results.append({
            "sample_id": sample.get("sample_id", f"sample_{len(results):06d}"),
            "source_image_id": sample.get("source_image_id", ""),
            "domain": sample.get("domain", "geology"),
            "split": sample.get("split", "test"),
            "professional_terms": sample.get("professional_terms", []),
            "has_professional_terms": sample.get("has_professional_terms", False),
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
    run_id: str = '', prompt_version: str = 'prompt_v1.0', agent_b_label: str = 'configured_agent_b',
):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    N = len(all_results)
    if N == 0:
        return None
    n_call_target = int(round(N * target_budget))

    if strategy == 'AgentA_Only':
        cer_num = sum(Levenshtein.distance(r['T_A'], r['T_GT']) for r in all_results)
        gt_len = sum(len(r['T_GT']) for r in all_results)
        per_sample = []
        for r in all_results:
            per_sample.append({
                'sample_id': r.get('sample_id', ''),
                'image_path': r.get('image_path', ''),
                'source_image_id': r.get('source_image_id', ''),
                'domain': r.get('domain', 'geology'),
                'split': r.get('split', 'test'),
                'gt': r['T_GT'],
                'ocr_text': r['T_A'],
                'router_name': 'AgentA_Only',
                'router_score': 0.0,
                'budget': 0.0,
                'budget_mode': 'online',
                'selected_for_upgrade': False,
                'vlm_model': 'none',
                'prompt_version': prompt_version,
                'vlm_raw_output': '',
                'latency_ms': None,
                'token_usage': None,
                'error_type': 'not_applicable',
                'final_text_if_upgraded': '',
                'final_text': r['T_A'],
                'backfill_status': 'not_applicable',
                'backfill_reason': 'not_upgraded',
                'run_id': run_id,
            })
        return {
            'summary': {
                'Strategy': 'AgentA_Only', 'Target_Budget': 0.0,
                'Actual_Call_Rate': 0.0,
                'Overall_CER': round(cer_num / gt_len, 6) if gt_len else 0,
                'AER': 0.0, 'CVR': 0.0, 'N_valid': N,
            },
            'per_sample': per_sample,
            'backfill_log': [],
        }

    # 计算路由分数
    if strategy == 'GCR':
        # Global Confidence Router：仅用全局置信度
        scores = [(1.0 - r['conf']) for r in all_results]
    elif strategy == 'BAUR':
        # Budget-Aware Uncertainty Router：轻量不确定性，不含领域项
        scores = []
        for r in all_results:
            dec = router.route(
                boundary_stats=r['boundary_stats'],
                top2_info=r['top2_info'],
                r_d=0.0,
                agent_a_text=r['T_A'],
            )
            scores.append(max(dec.s_b, dec.s_a))
    elif strategy == 'DAR':
        # Domain-Augmented Router：在 BAUR 基础上显式加入领域项
        scores = []
        for r in all_results:
            dec = router.route(
                boundary_stats=r['boundary_stats'],
                top2_info=r['top2_info'],
                r_d=r['r_d'],
                agent_a_text=r['T_A'],
            )
            scores.append(dec.q)
    else:  # SH-DA++ or BAUR-only
        # BAUR-only / SH-DA++ 当前都基于 BAUR 路由（不含领域项）
        scores = []
        for r in all_results:
            dec = router.route(
                boundary_stats=r['boundary_stats'],
                top2_info=r['top2_info'],
                r_d=0.0,
                agent_a_text=r['T_A'],
            )
            scores.append(max(dec.s_b, dec.s_a))

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
            domain_label = {
                'geology': '地质勘探',
                'finance': '金融财会',
                'medicine': '医学',
            }.get(r.get('domain', 'geology')) if strategy == 'SH-DA++' else None
            prompt = prompter.generate_targeted_correction_prompt(
                T_A=r['T_A'], min_conf_idx=r['min_conf_idx'],
                domain=domain_label, image_path=r['img_path'],
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
                    idx, agent_b_result = future.result()
                    upgrade_results[idx] = agent_b_result
                except Exception as e:
                    idx = futures[future]
                    upgrade_results[idx] = {
                        'corrected_text': all_results[idx]['T_A'],
                        'latency_ms': None,
                        'token_usage': None,
                        'error_type': type(e).__name__,
                    }  # 失败降级
    
    # 回填与统计
    cer_num = 0
    gt_len = 0
    n_upgraded = 0
    n_accepted_edit = 0
    n_rejected = 0
    per_sample = []
    backfill_log = []
    
    for i, r in enumerate(all_results):
        T_A = r['T_A']
        T_GT = r['T_GT']
        router_score = float(scores[i]) if i < len(scores) else 0.0
        vlm_raw_output = ''
        final_text_if_upgraded = ''
        backfill_status = 'not_upgraded'
        backfill_reason = 'not_upgraded'
        latency_ms = None
        token_usage = None
        error_type = 'not_upgraded'
        
        if i in upgrade_set:
            n_upgraded += 1
            agent_b_result = upgrade_results.get(i, {
                'corrected_text': T_A,
                'latency_ms': None,
                'token_usage': None,
                'error_type': 'missing_agent_b_result',
            })
            T_cand = agent_b_result.get('corrected_text', T_A)
            vlm_raw_output = T_cand
            latency_ms = agent_b_result.get('latency_ms')
            token_usage = agent_b_result.get('token_usage')
            error_type = agent_b_result.get('error_type', 'none')

            if strategy == 'BAUR-only':
                # BAUR-only：不启用严格回填，直接接受 Agent B 输出
                T_final = T_cand if isinstance(T_cand, str) and T_cand else T_A
                final_text_if_upgraded = T_final
                backfill_status = 'skipped'
                backfill_reason = 'baur_only_no_backfill'
                if T_final != T_A:
                    n_accepted_edit += 1
            else:
                # SH-DA++ / 其他策略：启用严格回填
                bf = backfill_controller.apply_backfill(
                    T_A=T_A, T_cand=T_cand, route_type=RouteType.BOUNDARY,
                )
                T_final = bf.T_final
                final_text_if_upgraded = T_final
                if bf.is_rejected:
                    n_rejected += 1
                    backfill_status = 'rejected'
                    backfill_reason = bf.rejection_reason.value
                else:
                    backfill_status = 'accepted'
                    backfill_reason = bf.rejection_reason.value
                    if T_final != T_A:
                        n_accepted_edit += 1
        else:
            T_final = T_A
        
        cer_num += Levenshtein.distance(T_final, T_GT)
        gt_len += len(T_GT)
        per_sample.append({
            'sample_id': r.get('sample_id', ''),
            'image_path': r.get('image_path', ''),
            'source_image_id': r.get('source_image_id', ''),
            'domain': r.get('domain', 'geology'),
            'split': r.get('split', 'test'),
            'gt': T_GT,
            'ocr_text': T_A,
            'router_name': strategy,
            'router_score': round(router_score, 6),
            'budget': target_budget,
            'budget_mode': 'online',
            'selected_for_upgrade': i in upgrade_set,
            'vlm_model': agent_b_label,
            'prompt_version': prompt_version,
            'vlm_raw_output': vlm_raw_output,
            'latency_ms': latency_ms,
            'token_usage': token_usage,
            'error_type': error_type,
            'final_text_if_upgraded': final_text_if_upgraded,
            'final_text': T_final,
            'backfill_status': backfill_status,
            'backfill_reason': backfill_reason,
            'is_correct_ocr': T_A == T_GT,
            'is_correct_final': T_final == T_GT,
            'edit_distance_ocr': Levenshtein.distance(T_A, T_GT),
            'edit_distance_final': Levenshtein.distance(T_final, T_GT),
            'run_id': run_id,
        })
        if i in upgrade_set:
            backfill_log.append({
                'sample_id': r.get('sample_id', ''),
                'router_name': strategy,
                'budget': target_budget,
                'ocr_text': T_A,
                'vlm_raw_output': vlm_raw_output,
                'latency_ms': latency_ms,
                'token_usage': token_usage,
                'error_type': error_type,
                'final_text': T_final,
                'backfill_status': backfill_status,
                'backfill_reason': backfill_reason,
                'run_id': run_id,
            })

    actual_rate = n_upgraded / N if N > 0 else 0.0
    overall_cer = cer_num / gt_len if gt_len > 0 else 0.0
    aer = n_accepted_edit / n_upgraded if n_upgraded > 0 else 0.0
    cvr = n_rejected / n_upgraded if n_upgraded > 0 else 0.0
    
    return {
        'summary': {
            'Strategy': strategy, 'Target_Budget': target_budget,
            'Actual_Call_Rate': round(actual_rate, 4),
            'Overall_CER': round(overall_cer, 6),
            'AER': round(aer, 4), 'CVR': round(cvr, 4), 'N_valid': N,
        },
        'per_sample': per_sample,
        'backfill_log': backfill_log,
    }


def replay_from_full_budget(
    strategy: str,
    target_budget: float,
    full_budget_items: List[dict],
    run_id: str = '',
    prompt_version: str = 'prompt_v1.0',
):
    """基于 100% full-call 结果离线重建预算点。"""
    N = len(full_budget_items)
    if N == 0:
        return None

    n_call_target = int(round(N * target_budget))
    ranked_indices = sorted(
        range(N),
        key=lambda i: float(full_budget_items[i].get('router_score', 0.0)),
        reverse=True,
    )
    upgrade_set = set(ranked_indices[:n_call_target])

    per_sample = []
    backfill_log = []
    cer_num = 0
    gt_len = 0
    n_upgraded = 0
    n_accepted_edit = 0
    n_rejected = 0

    for i, item in enumerate(full_budget_items):
        T_A = item.get('ocr_text', '')
        T_GT = item.get('gt', '')
        is_upgraded = i in upgrade_set
        upgraded_text = item.get('final_text_if_upgraded') or item.get('final_text') or T_A
        final_text = upgraded_text if is_upgraded else T_A
        backfill_status = item.get('backfill_status', 'not_upgraded') if is_upgraded else 'not_upgraded'
        backfill_reason = item.get('backfill_reason', 'not_upgraded') if is_upgraded else 'not_upgraded'
        vlm_raw_output = item.get('vlm_raw_output', '') if is_upgraded else ''
        latency_ms = item.get('latency_ms') if is_upgraded else None
        token_usage = item.get('token_usage') if is_upgraded else None
        error_type = item.get('error_type', 'not_upgraded') if is_upgraded else 'not_upgraded'

        if is_upgraded:
            n_upgraded += 1
            if backfill_status == 'rejected':
                n_rejected += 1
            if final_text != T_A:
                n_accepted_edit += 1
                backfill_log.append({
                    'sample_id': item.get('sample_id', ''),
                    'router_name': strategy,
                    'budget': target_budget,
                    'ocr_text': T_A,
                    'vlm_raw_output': vlm_raw_output,
                    'final_text': final_text,
                    'backfill_status': backfill_status,
                    'backfill_reason': backfill_reason,
                    'run_id': run_id,
                    'budget_mode': 'offline',
                })

        cer_num += Levenshtein.distance(final_text, T_GT)
        gt_len += len(T_GT)
        per_item = dict(item)
        per_item.update({
            'router_name': strategy,
            'budget': target_budget,
            'budget_mode': 'offline',
            'selected_for_upgrade': is_upgraded,
            'vlm_raw_output': vlm_raw_output,
            'latency_ms': latency_ms,
            'token_usage': token_usage,
            'error_type': error_type,
            'final_text': final_text,
            'backfill_status': backfill_status,
            'backfill_reason': backfill_reason,
            'is_correct_final': final_text == T_GT,
            'edit_distance_final': Levenshtein.distance(final_text, T_GT),
            'run_id': run_id,
            'prompt_version': prompt_version,
        })
        per_sample.append(per_item)

    actual_rate = n_upgraded / N if N > 0 else 0.0
    overall_cer = cer_num / gt_len if gt_len > 0 else 0.0
    aer = n_accepted_edit / n_upgraded if n_upgraded > 0 else 0.0
    cvr = n_rejected / n_upgraded if n_upgraded > 0 else 0.0

    return {
        'summary': {
            'Strategy': strategy,
            'Target_Budget': target_budget,
            'Actual_Call_Rate': round(actual_rate, 4),
            'Overall_CER': round(overall_cer, 6),
            'AER': round(aer, 4),
            'CVR': round(cvr, 4),
            'N_valid': N,
            'Budget_Mode': 'offline',
        },
        'per_sample': per_sample,
        'backfill_log': backfill_log,
    }


def main():
    parser = argparse.ArgumentParser(description='SH-DA++ v5.1 Phase 3')
    parser.add_argument('--config', default='configs/router_config.yaml')
    parser.add_argument('--test_jsonl', default='data/l2w1data/test.jsonl')
    parser.add_argument('--image_root', default='data/l2w1data/images')
    parser.add_argument('--rec_model_dir', default='./models/agent_a_ppocr/PP-OCRv5_server_rec_infer')
    parser.add_argument('--rec_char_dict_path', default='ppocr/utils/ppocrv5_dict.txt')
    parser.add_argument('--geo_dict', default='data/dicts/Geology.txt')
    parser.add_argument('--finance_dict', default='data/dicts/Finance.txt')
    parser.add_argument('--medicine_dict', default='data/dicts/Medicine.txt')
    parser.add_argument('--output_dir', default='results/stage2_v51')
    parser.add_argument('--budgets', default='0.05,0.10,0.20,0.30')
    parser.add_argument('--offline_replay_budgets', default='0.05,0.10,0.20,0.30,0.50,1.00', help='Offline replay budgets from full-budget results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_samples', type=int, default=None, help='Limit to first N samples (for testing)')
    parser.add_argument('--use_gpu', action='store_true', default=False, help='Use GPU for Agent A (default: CPU)')
    parser.add_argument('--use_cache', action='store_true', default=False, help='Load Agent A results from cache instead of re-inferring')
    parser.add_argument('--rebuild_cache', action='store_true', default=False, help='Force rebuild Agent A cache even if it exists')
    parser.add_argument('--prompt_version', default=None, help='Override frozen prompt version')
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    prompt_version = args.prompt_version or config.get('prompt_version', 'prompt_v1.0')
    mainline_agent_b = config.get('mainline_agent_b', 'configured_agent_b')
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
    domain_engine = DomainKnowledgeEngine({
        'geology': args.geo_dict,
        'finance': args.finance_dict,
        'medicine': args.medicine_dict,
    })
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

    run_id = datetime.now().strftime('%Y%m%d_run%H%M%S')
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f'  Run dir: {run_dir}')

    with open(run_dir / 'config_snapshot.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump({
            'run_id': run_id,
            'args': vars(args),
            'config': config,
        }, f, allow_unicode=True, sort_keys=False)

    csv_path = run_dir / 'summary.csv'
    metrics_json_path = run_dir / 'metrics_summary.json'
    failure_cases_path = run_dir / 'failure_cases.csv'
    domain_breakdown_path = run_dir / 'domain_breakdown.csv'
    backfill_log_path = run_dir / 'backfill_log.jsonl'
    fieldnames = ['Strategy', 'Target_Budget', 'Actual_Call_Rate',
                  'Overall_CER', 'AER', 'CVR', 'N_valid']
    offline_fieldnames = ['Strategy', 'Target_Budget', 'Actual_Call_Rate',
                          'Overall_CER', 'AER', 'CVR', 'N_valid', 'Budget_Mode']
    metrics_rows = []
    all_failure_cases = []
    all_backfill_logs = []
    domain_stats = {}
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        budgets = [float(b) for b in args.budgets.split(',')]
        
        print('\n--- AgentA_Only (Baseline 0) ---')
        result = run_pipeline('AgentA_Only', 0.0, all_results, router,
                              backfill_controller, prompter, agent_b_callable,
                              run_id=run_id, prompt_version=prompt_version, agent_b_label=mainline_agent_b)
        row = result['summary']
        writer.writerow({k: row.get(k, '') for k in fieldnames})
        metrics_rows.append(row)
        all_backfill_logs.extend(result.get('backfill_log', []))
        for item in result['per_sample']:
            key = (item.get('router_name', 'unknown'), item.get('domain', 'unknown'), str(item.get('budget', 0.0)))
            stat = domain_stats.setdefault(key, {'total': 0, 'correct_final': 0})
            stat['total'] += 1
            stat['correct_final'] += 1 if item.get('is_correct_final') else 0
            if item.get('is_correct_ocr') is False and item.get('is_correct_final') is False:
                all_failure_cases.append({
                    'sample_id': item.get('sample_id', ''),
                    'router_name': item.get('router_name', ''),
                    'budget': item.get('budget', 0.0),
                    'domain': item.get('domain', ''),
                    'ocr_text': item.get('ocr_text', ''),
                    'final_text': item.get('final_text', ''),
                    'gt': item.get('gt', ''),
                    'backfill_status': item.get('backfill_status', ''),
                    'backfill_reason': item.get('backfill_reason', ''),
                    'edit_distance_final': item.get('edit_distance_final', 0),
                })
        with open(run_dir / 'online_budget_00_results.jsonl', 'w', encoding='utf-8') as f:
            for item in result['per_sample']:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        csvfile.flush()
        print(f"  CER={row['Overall_CER']:.4%}")
        
        # 并行处理正式策略
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        for B in budgets:
            print(f'\n=== Budget B={B:.2f} ===')
            tasks = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                for strategy in ['GCR', 'BAUR', 'DAR', 'BAUR-only', 'SH-DA++']:
                    future = executor.submit(
                        run_pipeline, strategy, B, all_results, router,
                        backfill_controller, prompter, agent_b_callable,
                        run_id, prompt_version, mainline_agent_b
                    )
                    tasks.append((strategy, future))
                
                for strategy, future in tasks:
                    try:
                        result = future.result()
                        row = result['summary']
                        writer.writerow({k: row.get(k, '') for k in fieldnames})
                        metrics_rows.append(row)
                        all_backfill_logs.extend(result.get('backfill_log', []))
                        for item in result['per_sample']:
                            key = (item.get('router_name', 'unknown'), item.get('domain', 'unknown'), str(item.get('budget', 0.0)))
                            stat = domain_stats.setdefault(key, {'total': 0, 'correct_final': 0})
                            stat['total'] += 1
                            stat['correct_final'] += 1 if item.get('is_correct_final') else 0
                            if item.get('is_correct_ocr') is False and item.get('is_correct_final') is False:
                                all_failure_cases.append({
                                    'sample_id': item.get('sample_id', ''),
                                    'router_name': item.get('router_name', ''),
                                    'budget': item.get('budget', 0.0),
                                    'domain': item.get('domain', ''),
                                    'ocr_text': item.get('ocr_text', ''),
                                    'final_text': item.get('final_text', ''),
                                    'gt': item.get('gt', ''),
                                    'backfill_status': item.get('backfill_status', ''),
                                    'backfill_reason': item.get('backfill_reason', ''),
                                    'edit_distance_final': item.get('edit_distance_final', 0),
                                })
                        budget_tag = f"{int(round(B * 100)):02d}"
                        jsonl_path = run_dir / f'online_budget_{budget_tag}_{strategy}.jsonl'
                        with open(jsonl_path, 'w', encoding='utf-8') as f:
                            for item in result['per_sample']:
                                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                        csvfile.flush()
                        print(
                            f"  [{strategy:10s}] CER={row['Overall_CER']:.4%}"
                            f"  AER={row['AER']:.2%}"
                            f"  CVR={row['CVR']:.2%}"
                            f"  ActualRate={row['Actual_Call_Rate']:.2%}"
                        )
                    except Exception as e:
                        print(f"  [{strategy:10s}] ERROR: {e}")
    
    # Offline replay：基于 100% full-call 结果离线重建预算点
    offline_replay_budgets = [float(b) for b in args.offline_replay_budgets.split(',')]
    print('\n=== Offline Replay from Full-Budget Results ===')
    for strategy in ['GCR', 'BAUR', 'DAR', 'BAUR-only', 'SH-DA++']:
        full_budget_path = run_dir / f'full_budget_results_{strategy}.jsonl'
        full_budget_result = run_pipeline(
            strategy, 1.0, all_results, router,
            backfill_controller, prompter, agent_b_callable,
            run_id, prompt_version, mainline_agent_b
        )
        with open(full_budget_path, 'w', encoding='utf-8') as f:
            for item in full_budget_result['per_sample']:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        for B in offline_replay_budgets:
            replay_result = replay_from_full_budget(
                strategy=strategy,
                target_budget=B,
                full_budget_items=full_budget_result['per_sample'],
                run_id=run_id,
                prompt_version='prompt_v1.0',
            )
            replay_row = replay_result['summary']
            metrics_rows.append(replay_row)
            all_backfill_logs.extend(replay_result.get('backfill_log', []))
            for item in replay_result['per_sample']:
                key = (item.get('router_name', 'unknown'), item.get('domain', 'unknown'), f"offline:{item.get('budget', 0.0)}")
                stat = domain_stats.setdefault(key, {'total': 0, 'correct_final': 0})
                stat['total'] += 1
                stat['correct_final'] += 1 if item.get('is_correct_final') else 0
                if item.get('is_correct_ocr') is False and item.get('is_correct_final') is False:
                    all_failure_cases.append({
                        'sample_id': item.get('sample_id', ''),
                        'router_name': item.get('router_name', ''),
                        'budget': item.get('budget', 0.0),
                        'domain': item.get('domain', ''),
                        'ocr_text': item.get('ocr_text', ''),
                        'final_text': item.get('final_text', ''),
                        'gt': item.get('gt', ''),
                        'backfill_status': item.get('backfill_status', ''),
                        'backfill_reason': item.get('backfill_reason', ''),
                        'edit_distance_final': item.get('edit_distance_final', 0),
                    })
            budget_tag = f"{int(round(B * 100)):02d}"
            replay_path = run_dir / f'offline_budget_{budget_tag}_{strategy}.jsonl'
            with open(replay_path, 'w', encoding='utf-8') as f:
                for item in replay_result['per_sample']:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            print(f"  [offline {strategy:10s} B={B:.2f}] CER={replay_row['Overall_CER']:.4%} ActualRate={replay_row['Actual_Call_Rate']:.2%}")

    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_rows, f, ensure_ascii=False, indent=2)

    with open(backfill_log_path, 'w', encoding='utf-8') as f:
        for item in all_backfill_logs:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(failure_cases_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['sample_id', 'router_name', 'budget', 'domain', 'ocr_text', 'final_text', 'gt',
                        'backfill_status', 'backfill_reason', 'edit_distance_final']
        )
        writer.writeheader()
        for row in all_failure_cases:
            writer.writerow(row)

    with open(domain_breakdown_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['router_name', 'domain', 'budget', 'n_samples', 'final_accuracy']
        )
        writer.writeheader()
        for (router_name, domain, budget), stat in sorted(domain_stats.items()):
            total = stat['total']
            acc = (stat['correct_final'] / total) if total else 0.0
            writer.writerow({
                'router_name': router_name,
                'domain': domain,
                'budget': budget,
                'n_samples': total,
                'final_accuracy': round(acc, 6),
            })

    offline_summary_csv_path = run_dir / 'offline_summary.csv'
    with open(offline_summary_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=offline_fieldnames)
        writer.writeheader()
        for row in metrics_rows:
            if row.get('Budget_Mode') == 'offline':
                writer.writerow({k: row.get(k, '') for k in offline_fieldnames})

    print(f'\nDone: {csv_path}')
    print(f'Done: {metrics_json_path}')
    print(f'Done: {failure_cases_path}')
    print(f'Done: {domain_breakdown_path}')
    print(f'Done: {backfill_log_path}')
    print(f'Done: {run_dir}')


if __name__ == '__main__':
    main()
