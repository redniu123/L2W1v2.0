#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v5.1 Phase 3: Efficiency Frontier Grand Loop

评测数据集: Test 集
目标预算: [0.05, 0.10, 0.20, 0.30]
策略: AgentA_Only / GCR / WUR / DGCR / DWUR / BAUR / DAR / Router-only / SH-DA++
输出: results/stage2_v51/efficiency_frontier.csv
"""

import argparse
import csv
import difflib
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


def normalize_eval_text(text: str) -> str:
    """评测前做轻量字符归一化，避免全/半角括号、句号等格式差异放大 CER。"""
    if not text:
        return ""
    translation = str.maketrans({
        '（': '(',
        '）': ')',
        '【': '[',
        '】': ']',
        '｛': '{',
        '｝': '}',
        '，': ',',
        '：': ':',
        '；': ';',
        '！': '!',
        '？': '?',
        '。': '.',
    })
    return text.translate(translation)


def compute_cer(T_final: str, T_GT: str) -> float:
    T_final = normalize_eval_text(T_final)
    T_GT = normalize_eval_text(T_GT)
    if not T_GT:
        return 0.0
    return Levenshtein.distance(T_final, T_GT) / len(T_GT)


def identify_boundary_deletion(agent_a_text: str, gt_text: str, k: int = 2) -> bool:
    if not gt_text:
        return False
    matcher = difflib.SequenceMatcher(None, agent_a_text, gt_text)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'insert':
            for gt_pos in range(j1, j2):
                if gt_pos < k or gt_pos >= len(gt_text) - k:
                    return True
        elif tag == 'replace':
            pred_len = i2 - i1
            gt_len = j2 - j1
            if gt_len > pred_len:
                for offset in range(gt_len - pred_len):
                    gt_pos = j1 + pred_len + offset
                    if gt_pos < k or gt_pos >= len(gt_text) - k:
                        return True
    return False


def count_substitutions(prediction: str, reference: str) -> int:
    matcher = difflib.SequenceMatcher(None, prediction, reference)
    substitutions = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            substitutions += min(i2 - i1, j2 - j1)
    return substitutions


def summarize_extended_metrics(per_sample: List[dict], boundary_k: int = 2) -> Dict[str, float]:
    boundary_total = 0
    boundary_upgraded = 0
    substitution_errors = 0
    gt_len_total = 0
    for item in per_sample:
        gt = item.get('gt', '')
        ocr_text = item.get('ocr_text', '')
        final_text = item.get('final_text', '')
        if identify_boundary_deletion(ocr_text, gt, k=boundary_k):
            boundary_total += 1
            if item.get('selected_for_upgrade'):
                boundary_upgraded += 1
        substitution_errors += count_substitutions(final_text, gt)
        gt_len_total += len(gt)
    return {
        'Boundary_Deletion_Recall@B': round((boundary_upgraded / boundary_total), 6) if boundary_total else 0.0,
        'Substitution_CER': round((substitution_errors / gt_len_total), 6) if gt_len_total else 0.0,
    }


def summarize_latency_and_token_usage(per_sample: List[dict]) -> Dict[str, float]:
    latencies = [float(item['latency_ms']) for item in per_sample if item.get('latency_ms') is not None]
    token_usages = [float(item['token_usage']) for item in per_sample if item.get('token_usage') is not None]
    return {
        'P95_Latency_MS': round(float(np.percentile(latencies, 95)), 3) if latencies else 0.0,
        'Avg_Token_Usage': round((sum(token_usages) / len(token_usages)), 3) if token_usages else 0.0,
        'Total_Token_Usage': round(sum(token_usages), 3) if token_usages else 0.0,
        'N_Latency_Valid': len(latencies),
        'N_Token_Valid': len(token_usages),
    }


def ensure_agent_a_result_schema(all_results: List[dict]) -> List[dict]:
    """兼容旧版 cache：补齐正式主线所需字段。"""
    normalized = []
    for item in all_results:
        r = dict(item)
        r.setdefault('sample_id', '')
        r.setdefault('source_image_id', '')
        r.setdefault('domain', 'geology')
        r.setdefault('split', 'test')
        r.setdefault('professional_terms', [])
        r.setdefault('has_professional_terms', False)
        r.setdefault('image_path', r.get('img_path', ''))
        r.setdefault('img_path', r.get('image_path', ''))
        r.setdefault('boundary_stats', {})
        r.setdefault('top2_info', {})
        if r.get('conf') is None:
            conf = r.get('mean_conf', r.get('min_conf', 0.0))
            r['conf'] = float(conf) if conf is not None else 0.0
        if r.get('mean_conf') is None:
            r['mean_conf'] = float(r.get('conf', 0.0))
        if r.get('min_conf') is None:
            r['min_conf'] = float(r.get('conf', 0.0))
        if 'min_conf_idx' not in r:
            r['min_conf_idx'] = None
        if r.get('r_d') is None:
            r['r_d'] = 0.0
        if r.get('b_edge') is None:
            r['b_edge'] = 0.0
        if r.get('drop') is None:
            r['drop'] = 0.0
        normalized.append(r)
    return normalized


def build_agent_b_callable(config: dict, max_retries: int = 3) -> Callable:
    """构建 Agent B 调用函数（含重试）"""
    import re

    def attach_callable_meta(fn: Callable, backend_name: str, model_label: str, supports_parallel: bool, max_concurrency: int = 1, key_count: int = 1) -> Callable:
        fn._backend = backend_name
        fn._model_label = model_label
        fn._supports_parallel = supports_parallel
        fn._max_concurrency = max_concurrency
        fn._key_count = key_count
        return fn

    agent_b_cfg = config.get("agent_b", {})
    skip = agent_b_cfg.get("skip", True)
    backend = agent_b_cfg.get("backend", "gemini")

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
        return attach_callable_meta(mock_fn, "mock", "mock", False, 1)

    if backend == "gemini":
        try:
            from modules.vlm_expert.gemini_expert import GeminiAgentB, GeminiConfig
            gemini_cfg = GeminiConfig(
                base_url=agent_b_cfg.get("base_url", "https://new.lemonapi.site/v1"),
                model_name=agent_b_cfg.get("model_name", "gemini-3-flash-preview"),
                key_file=agent_b_cfg.get("key_file", "key.txt"),
                provider_pool=agent_b_cfg.get("provider_pool", "gemini_1x"),
                temperature=agent_b_cfg.get("temperature", 0.1),
                max_tokens=agent_b_cfg.get("max_tokens", 256),
                max_retries=agent_b_cfg.get("max_retries", 3),
                timeout=agent_b_cfg.get("timeout", 60),
            )
            agent = GeminiAgentB(config=gemini_cfg)
            print(f"[Agent B] Gemini backend: {agent.config.model_name} @ {agent.config.base_url}")
            agent_model_label = f"gemini:{agent.config.model_name}"
            max_concurrency = int(agent_b_cfg.get("max_concurrency", 100) or 100)
            key_count = agent.config.key_manager.get_key_count()
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
            return attach_callable_meta(mock_fn, "mock", f"mock:gemini_load_failed:{type(e).__name__}", False, 1)
    elif backend == "local_vlm":
        try:
            from modules.vlm_expert import AgentBFactory
            agent = AgentBFactory.create(config)
            info = agent.get_model_info() if hasattr(agent, "get_model_info") else {"backend": "local_vlm"}
            print(f"[Agent B] Local backend ready: {info}")
            model_type = info.get("model_type", agent_b_cfg.get("model_type", "local_vlm"))
            model_path = info.get("model_path", agent_b_cfg.get("model_path", ""))
            model_name = Path(model_path).name if model_path else model_type
            agent_model_label = f"local_vlm:{model_type}:{model_name}"
        except Exception as e:
            print(f"[Agent B] local_vlm load failed: {e}, fallback to mock")
            def mock_fn(prompt: dict) -> dict:
                t0 = time.perf_counter()
                return {
                    "corrected_text": prompt.get("T_A", ""),
                    "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
                    "token_usage": None,
                    "error_type": "local_vlm_load_failed",
                }
            return attach_callable_meta(mock_fn, "mock", f"mock:local_vlm_load_failed:{type(e).__name__}", False, 1)
    else:
        print(f"[Agent B] Unknown backend: {backend}, fallback to mock")
        def mock_fn(prompt: dict) -> dict:
            t0 = time.perf_counter()
            return {
                "corrected_text": prompt.get("T_A", ""),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 3),
                "token_usage": None,
                "error_type": "unknown_backend",
            }
        return attach_callable_meta(mock_fn, "mock", "mock", False, 1)

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
            "domain": prompt.get("domain"),
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

    real_fn = attach_callable_meta(
        real_fn,
        backend_name=backend,
        model_label=locals().get("agent_model_label", backend),
        supports_parallel=(backend == "gemini"),
        max_concurrency=locals().get("max_concurrency", 1),
        key_count=locals().get("key_count", 1),
    )
    return real_fn


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
        cer_num = sum(Levenshtein.distance(normalize_eval_text(r['T_A']), normalize_eval_text(r['T_GT'])) for r in all_results)
        gt_len = sum(len(normalize_eval_text(r['T_GT'])) for r in all_results)
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
                'has_professional_terms': r.get('has_professional_terms', False),
                'professional_terms': r.get('professional_terms', []),
                'domain_risk_score': round(float(r.get('r_d', 0.0)), 6),
                'cvr_flag': False,
                'replay_rank': None,
                'final_text_if_upgraded': '',
                'final_text': r['T_A'],
                'backfill_status': 'not_applicable',
                'backfill_reason': 'not_upgraded',
                'run_id': run_id,
            })
        usage_metrics = summarize_latency_and_token_usage(per_sample)
        return {
            'summary': {
                'Strategy': 'AgentA_Only', 'Target_Budget': 0.0,
                'Actual_Call_Rate': 0.0,
                'Overall_CER': round(cer_num / gt_len, 6) if gt_len else 0,
                'Boundary_Deletion_Recall@B': 0.0,
                'Substitution_CER': round((sum(count_substitutions(r['T_A'], r['T_GT']) for r in all_results) / gt_len), 6) if gt_len else 0.0,
                'P95_Latency_MS': usage_metrics['P95_Latency_MS'],
                'Avg_Token_Usage': usage_metrics['Avg_Token_Usage'],
                'Total_Token_Usage': usage_metrics['Total_Token_Usage'],
                'N_Latency_Valid': usage_metrics['N_Latency_Valid'],
                'N_Token_Valid': usage_metrics['N_Token_Valid'],
                'AER': 0.0, 'CVR': 0.0, 'N_valid': N,
            },
            'per_sample': per_sample,
            'backfill_log': [],
        }

    # 主线路由：GCR / WUR / DGCR / DWUR
    # 补充路线：BAUR / DAR
    # 系统对照：Router-only / SH-DA++
    if strategy == 'GCR':
        scores = [1.0 - float(r.get('conf', r.get('mean_conf', 0.0))) for r in all_results]
    elif strategy == 'DGCR':
        scores = [
            (1.0 - float(r.get('conf', r.get('mean_conf', 0.0)))) + float(r.get('r_d', 0.0))
            for r in all_results
        ]
    elif strategy == 'WUR':
        scores = []
        for r in all_results:
            mean_conf = float(r.get('mean_conf', r.get('conf', 0.0)))
            min_conf = float(r.get('min_conf', r.get('conf', 0.0)))
            drop = float(r.get('drop', 0.0))
            q = 0.5 * (1.0 - mean_conf) + 0.3 * (1.0 - min_conf) + 0.2 * drop
            if min_conf < 0.35:
                q += 0.10
            if drop > 0.20:
                q += 0.10
            scores.append(float(q))
    elif strategy == 'DWUR':
        scores = []
        eta = float(getattr(router.config.rule_scorer, 'eta', 0.5))
        for r in all_results:
            mean_conf = float(r.get('mean_conf', r.get('conf', 0.0)))
            min_conf = float(r.get('min_conf', r.get('conf', 0.0)))
            drop = float(r.get('drop', 0.0))
            q = 0.5 * (1.0 - mean_conf) + 0.3 * (1.0 - min_conf) + 0.2 * drop
            if min_conf < 0.35:
                q += 0.10
            if drop > 0.20:
                q += 0.10
            q += eta * float(r.get('r_d', 0.0))
            scores.append(float(q))
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
    else:  # Router-only or SH-DA++
        # 系统对照默认使用 WUR 作为当前固定规则主路由
        scores = []
        for r in all_results:
            mean_conf = float(r.get('mean_conf', r.get('conf', 0.0)))
            min_conf = float(r.get('min_conf', r.get('conf', 0.0)))
            drop = float(r.get('drop', 0.0))
            q = 0.5 * (1.0 - mean_conf) + 0.3 * (1.0 - min_conf) + 0.2 * drop
            if min_conf < 0.35:
                q += 0.10
            if drop > 0.20:
                q += 0.10
            scores.append(float(q))

    upgrade_set = set(
        sorted(range(N), key=lambda i: scores[i], reverse=True)[:n_call_target]
    )

    # 批量并发调用 Agent B
    from modules.router.backfill import RouteType
    upgrade_results = {}  # {index: T_cand}
    
    if upgrade_set:
        supports_parallel = bool(getattr(agent_b_callable, '_supports_parallel', False))
        backend_name = getattr(agent_b_callable, '_backend', 'unknown')
        if supports_parallel:
            key_count = int(getattr(agent_b_callable, '_key_count', 10) or 10)
            max_concurrency = int(getattr(agent_b_callable, '_max_concurrency', 4) or 4)
            n_workers = max(1, min(key_count, max_concurrency))
        else:
            n_workers = 1
        print(f"  [{strategy}] Calling Agent B for {len(upgrade_set)} samples ({n_workers} concurrent, backend={backend_name})...")

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

        if supports_parallel and n_workers > 1:
            with ThreadPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(call_agent_b, i, all_results[i]): i
                    for i in upgrade_set
                }

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
                        }
        else:
            ordered_indices = sorted(upgrade_set)
            for idx in tqdm(ordered_indices, total=len(ordered_indices),
                            desc=f'{strategy} B={target_budget:.2f} [local]', leave=False):
                try:
                    _, agent_b_result = call_agent_b(idx, all_results[idx])
                    upgrade_results[idx] = agent_b_result
                except Exception as e:
                    upgrade_results[idx] = {
                        'corrected_text': all_results[idx]['T_A'],
                        'latency_ms': None,
                        'token_usage': None,
                        'error_type': type(e).__name__,
                    }
    
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

            if strategy == 'Router-only':
                # Router-only：不启用严格回填，直接接受 Agent B 输出
                T_final = T_cand if isinstance(T_cand, str) and T_cand else T_A
                final_text_if_upgraded = T_final
                backfill_status = 'skipped'
                backfill_reason = 'router_only_no_backfill'
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
        
        cer_num += Levenshtein.distance(normalize_eval_text(T_final), normalize_eval_text(T_GT))
        gt_len += len(normalize_eval_text(T_GT))
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
            'vlm_model': getattr(agent_b_callable, '_model_label', agent_b_label),
            'prompt_version': prompt_version,
            'vlm_raw_output': vlm_raw_output,
            'latency_ms': latency_ms,
            'token_usage': token_usage,
            'error_type': error_type,
            'has_professional_terms': r.get('has_professional_terms', False),
            'professional_terms': r.get('professional_terms', []),
            'domain_risk_score': round(float(r.get('r_d', 0.0)), 6),
            'cvr_flag': backfill_status == 'rejected',
            'replay_rank': None,
            'final_text_if_upgraded': final_text_if_upgraded,
            'final_text': T_final,
            'backfill_status': backfill_status,
            'backfill_reason': backfill_reason,
            'is_correct_ocr': T_A == T_GT,
            'is_correct_final': T_final == T_GT,
            'edit_distance_ocr': Levenshtein.distance(normalize_eval_text(T_A), normalize_eval_text(T_GT)),
            'edit_distance_final': Levenshtein.distance(normalize_eval_text(T_final), normalize_eval_text(T_GT)),
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
    
    extended_metrics = summarize_extended_metrics(per_sample)
    usage_metrics = summarize_latency_and_token_usage(per_sample)
    return {
        'summary': {
            'Strategy': strategy, 'Target_Budget': target_budget,
            'Actual_Call_Rate': round(actual_rate, 4),
            'Overall_CER': round(overall_cer, 6),
            'Boundary_Deletion_Recall@B': extended_metrics['Boundary_Deletion_Recall@B'],
            'Substitution_CER': extended_metrics['Substitution_CER'],
            'P95_Latency_MS': usage_metrics['P95_Latency_MS'],
            'Avg_Token_Usage': usage_metrics['Avg_Token_Usage'],
            'Total_Token_Usage': usage_metrics['Total_Token_Usage'],
            'N_Latency_Valid': usage_metrics['N_Latency_Valid'],
            'N_Token_Valid': usage_metrics['N_Token_Valid'],
            'AER': round(aer, 4), 'CVR': round(cvr, 4), 'N_valid': N,
        },
        'per_sample': per_sample,
        'backfill_log': backfill_log,
    }


def collect_case_rows(item: dict, failure_rows: List[dict], degradation_rows: List[dict]) -> None:
    if item.get('is_correct_ocr') is False and item.get('is_correct_final') is False:
        failure_rows.append({
            'sample_id': item.get('sample_id', ''),
            'router_name': item.get('router_name', ''),
            'budget': item.get('budget', 0.0),
            'domain': item.get('domain', ''),
            'ocr_text': item.get('ocr_text', ''),
            'final_text': item.get('final_text', ''),
            'gt': item.get('gt', ''),
            'backfill_status': item.get('backfill_status', ''),
            'backfill_reason': item.get('backfill_reason', ''),
            'edit_distance_ocr': item.get('edit_distance_ocr', 0),
            'edit_distance_final': item.get('edit_distance_final', 0),
        })

    if item.get('selected_for_upgrade') and item.get('final_text') != item.get('ocr_text'):
        ed_ocr = int(item.get('edit_distance_ocr', 0))
        ed_final = int(item.get('edit_distance_final', 0))
        if ed_final > ed_ocr:
            degradation_rows.append({
                'sample_id': item.get('sample_id', ''),
                'router_name': item.get('router_name', ''),
                'budget': item.get('budget', 0.0),
                'domain': item.get('domain', ''),
                'ocr_text': item.get('ocr_text', ''),
                'vlm_raw_output': item.get('vlm_raw_output', ''),
                'final_text': item.get('final_text', ''),
                'gt': item.get('gt', ''),
                'backfill_status': item.get('backfill_status', ''),
                'backfill_reason': item.get('backfill_reason', ''),
                'edit_distance_ocr': ed_ocr,
                'edit_distance_final': ed_final,
                'delta_edit_distance': ed_final - ed_ocr,
                'vlm_model': item.get('vlm_model', ''),
            })


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
    rank_map = {idx: rank for rank, idx in enumerate(ranked_indices)}
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

        cer_num += Levenshtein.distance(normalize_eval_text(final_text), normalize_eval_text(T_GT))
        gt_len += len(normalize_eval_text(T_GT))
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
            'cvr_flag': backfill_status == 'rejected',
            'replay_rank': rank_map.get(i),
            'is_correct_final': final_text == T_GT,
            'edit_distance_final': Levenshtein.distance(normalize_eval_text(final_text), normalize_eval_text(T_GT)),
            'run_id': run_id,
            'prompt_version': prompt_version,
        })
        per_sample.append(per_item)

    actual_rate = n_upgraded / N if N > 0 else 0.0
    overall_cer = cer_num / gt_len if gt_len > 0 else 0.0
    aer = n_accepted_edit / n_upgraded if n_upgraded > 0 else 0.0
    cvr = n_rejected / n_upgraded if n_upgraded > 0 else 0.0
    extended_metrics = summarize_extended_metrics(per_sample)

    return {
        'summary': {
            'Strategy': strategy,
            'Target_Budget': target_budget,
            'Actual_Call_Rate': round(actual_rate, 4),
            'Overall_CER': round(overall_cer, 6),
            'Boundary_Deletion_Recall@B': extended_metrics['Boundary_Deletion_Recall@B'],
            'Substitution_CER': extended_metrics['Substitution_CER'],
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
    parser.add_argument('--budgets', default=None, help='Comma-separated online budgets. Defaults to config.mainline.formal_budgets')
    parser.add_argument('--strategies', default='GCR,WUR,DGCR,DWUR,BAUR,DAR,Router-only,SH-DA++', help='Comma-separated strategies to run online')
    parser.add_argument('--offline_replay_budgets', default='0.05,0.10,0.20,0.30,0.50,1.00', help='Offline replay budgets from full-budget results')
    parser.add_argument('--offline_strategies', default='GCR,WUR,DGCR,DWUR,BAUR,DAR,Router-only,SH-DA++', help='Comma-separated strategies to run offline replay')
    parser.add_argument('--skip_offline_replay', action='store_true', help='Skip offline replay stage')
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
    prompt_version = args.prompt_version or config.get('prompt_version') or config.get('mainline', {}).get('prompt_version', 'prompt_v1.1')
    mainline_agent_b = config.get('mainline_agent_b') or config.get('mainline', {}).get('mainline_agent_b', 'configured_agent_b')
    formal_budgets = config.get('mainline', {}).get('formal_budgets', [0.10, 0.20, 0.30])
    budgets = [float(b) for b in (args.budgets.split(',') if args.budgets else formal_budgets)]
    online_strategies = [s.strip() for s in args.strategies.split(',') if s.strip()]
    offline_strategies = [s.strip() for s in args.offline_strategies.split(',') if s.strip()]
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
        all_results = ensure_agent_a_result_schema(all_results)
        print(f'  Loaded {len(all_results)} cached results')
    else:
        print('[4/4] Agent A full inference...')
        all_results = infer_all_samples(samples, recognizer, domain_engine, None, args.image_root)
        all_results = ensure_agent_a_result_schema(all_results)
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
    degradation_cases_path = run_dir / 'degradation_cases.csv'
    domain_breakdown_path = run_dir / 'domain_breakdown.csv'
    backfill_log_path = run_dir / 'backfill_log.jsonl'
    fieldnames = ['Strategy', 'Target_Budget', 'Actual_Call_Rate',
                  'Overall_CER', 'Boundary_Deletion_Recall@B', 'Substitution_CER',
                  'P95_Latency_MS', 'Avg_Token_Usage', 'Total_Token_Usage', 'N_Latency_Valid', 'N_Token_Valid',
                  'AER', 'CVR', 'N_valid']
    offline_fieldnames = ['Strategy', 'Target_Budget', 'Actual_Call_Rate',
                          'Overall_CER', 'Boundary_Deletion_Recall@B', 'Substitution_CER',
                          'P95_Latency_MS', 'Avg_Token_Usage', 'Total_Token_Usage', 'N_Latency_Valid', 'N_Token_Valid',
                          'AER', 'CVR', 'N_valid', 'Budget_Mode']
    metrics_rows = []
    all_failure_cases = []
    all_degradation_cases = []
    all_backfill_logs = []
    domain_stats = {}
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
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
            collect_case_rows(item, all_failure_cases, all_degradation_cases)
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
            with ThreadPoolExecutor(max_workers=len(online_strategies)) as executor:
                for strategy in online_strategies:
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
                            collect_case_rows(item, all_failure_cases, all_degradation_cases)
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
    if not args.skip_offline_replay:
        print('\n=== Offline Replay from Full-Budget Results ===')
        for strategy in offline_strategies:
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
                    prompt_version=prompt_version,
                )
                replay_row = replay_result['summary']
                metrics_rows.append(replay_row)
                all_backfill_logs.extend(replay_result.get('backfill_log', []))
                for item in replay_result['per_sample']:
                    key = (item.get('router_name', 'unknown'), item.get('domain', 'unknown'), f"offline:{item.get('budget', 0.0)}")
                    stat = domain_stats.setdefault(key, {'total': 0, 'correct_final': 0})
                    stat['total'] += 1
                    stat['correct_final'] += 1 if item.get('is_correct_final') else 0
                    collect_case_rows(item, all_failure_cases, all_degradation_cases)
                budget_tag = f"{int(round(B * 100)):02d}"
                replay_path = run_dir / f'offline_budget_{budget_tag}_{strategy}.jsonl'
                with open(replay_path, 'w', encoding='utf-8') as f:
                    for item in replay_result['per_sample']:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                print(f"  [offline {strategy:10s} B={B:.2f}] CER={replay_row['Overall_CER']:.4%} ActualRate={replay_row['Actual_Call_Rate']:.2%}")
    else:
        print('\n=== Offline Replay skipped by flag ===')

    with open(metrics_json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_rows, f, ensure_ascii=False, indent=2)

    with open(backfill_log_path, 'w', encoding='utf-8') as f:
        for item in all_backfill_logs:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    with open(failure_cases_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['sample_id', 'router_name', 'budget', 'domain', 'ocr_text', 'final_text', 'gt',
                        'backfill_status', 'backfill_reason', 'edit_distance_ocr', 'edit_distance_final']
        )
        writer.writeheader()
        for row in all_failure_cases:
            writer.writerow(row)

    with open(degradation_cases_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['sample_id', 'router_name', 'budget', 'domain', 'ocr_text', 'vlm_raw_output', 'final_text', 'gt',
                        'backfill_status', 'backfill_reason', 'edit_distance_ocr', 'edit_distance_final',
                        'delta_edit_distance', 'vlm_model']
        )
        writer.writeheader()
        for row in all_degradation_cases:
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
    print(f'Done: {degradation_cases_path}')
    print(f'Done: {domain_breakdown_path}')
    print(f'Done: {backfill_log_path}')
    print(f'Done: {run_dir}')


if __name__ == '__main__':
    main()
