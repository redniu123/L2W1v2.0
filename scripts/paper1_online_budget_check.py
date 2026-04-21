#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""paper1 RouterOnly online legality check with Main C cache reuse."""

import argparse
import csv
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import Levenshtein
import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.router.uncertainty_router import BudgetControllerConfig, OnlineBudgetController
from scripts.run_efficiency_frontier import ensure_agent_a_result_schema, infer_all_samples, summarize_extended_metrics, summarize_latency_and_token_usage

WUR_MEAN_WEIGHT = 0.5
WUR_MIN_WEIGHT = 0.3
WUR_DROP_WEIGHT = 0.2
WUR_MIN_CONF_GATE_THRESHOLD = 0.35
WUR_DROP_GATE_THRESHOLD = 0.20
WUR_GATE_BONUS = 0.10
MODEL_LABELS = {
    'V1': 'Qwen3-VL-8B',
    'V2': 'MiniCPM-V 4.5',
    'V3': 'Gemini 3 Flash Preview',
    'V4': 'gpt-5.4',
}


def load_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def build_router_score(strategy, row, eta=0.5):
    mean_conf = float(row.get('mean_conf', row.get('conf', 0.0)))
    min_conf = float(row.get('min_conf', row.get('conf', 0.0)))
    drop = float(row.get('drop', 0.0))
    conf = float(row.get('conf', mean_conf))
    r_d = float(row.get('r_d', 0.0))
    if strategy == 'GCR':
        return 1.0 - conf
    if strategy == 'DGCR':
        return (1.0 - conf) + r_d
    wur = WUR_MEAN_WEIGHT * (1.0 - mean_conf) + WUR_MIN_WEIGHT * (1.0 - min_conf) + WUR_DROP_WEIGHT * drop
    if min_conf < WUR_MIN_CONF_GATE_THRESHOLD:
        wur += WUR_GATE_BONUS
    if drop > WUR_DROP_GATE_THRESHOLD:
        wur += WUR_GATE_BONUS
    if strategy == 'WUR':
        return float(wur)
    if strategy == 'DWUR':
        return float(wur + eta * r_d)
    raise ValueError(f'Unsupported strategy: {strategy}')


def build_cached_result_lookup(full_call_rows):
    lookup = {}
    for row in full_call_rows:
        sid = row.get('sample_id', '')
        if sid:
            lookup[sid] = row
    return lookup


def run_online_routeronly(strategy, target_budget, all_results, cached_lookup, budget_cfg, run_id='', prompt_version='prompt_v1.1', agent_b_label='Gemini 3 Flash Preview', eta=0.5):
    ctrl = OnlineBudgetController(budget_cfg)
    per_sample, validation_logs = [], []
    cer_num = gt_len = n_upgraded = n_accepted = 0
    missing_cached_rows = 0

    for row in tqdm(all_results, desc=f'{strategy} online B={target_budget:.2f}', leave=False):
        t_a, t_gt = row['T_A'], row['T_GT']
        q = float(build_router_score(strategy, row, eta=eta))
        upgrade, budget_details = ctrl.step(q)
        cached = cached_lookup.get(row.get('sample_id', ''))
        vlm_raw_output = ''
        latency_ms = token_usage = None
        error_type = 'not_upgraded'
        final_text = t_a

        if upgrade:
            n_upgraded += 1
            if cached is None:
                missing_cached_rows += 1
                error_type = 'cached_result_missing'
            else:
                vlm_raw_output = cached.get('vlm_raw_output', cached.get('final_text_if_upgraded', t_a))
                final_text = cached.get('final_text_if_upgraded', t_a) or t_a
                latency_ms = cached.get('latency_ms')
                token_usage = cached.get('token_usage')
                error_type = cached.get('error_type', 'none')
                if final_text != t_a:
                    n_accepted += 1

        cer_num += Levenshtein.distance(final_text, t_gt)
        gt_len += len(t_gt)
        item = {
            'sample_id': row.get('sample_id', ''),
            'image_path': row.get('image_path', row.get('img_path', '')),
            'source_image_id': row.get('source_image_id', ''),
            'domain': row.get('domain', 'geology'),
            'split': row.get('split', 'test'),
            'gt': t_gt,
            'ocr_text': t_a,
            'router_name': strategy,
            'router_score': round(q, 6),
            'budget': target_budget,
            'budget_mode': 'online_control_reuse_cache',
            'selected_for_upgrade': upgrade,
            'vlm_model': agent_b_label,
            'prompt_version': prompt_version,
            'vlm_raw_output': vlm_raw_output,
            'latency_ms': latency_ms,
            'token_usage': token_usage,
            'error_type': error_type,
            'has_professional_terms': row.get('has_professional_terms', False),
            'professional_terms': row.get('professional_terms', []),
            'domain_risk_score': round(float(row.get('r_d', 0.0)), 6),
            'cvr_flag': False,
            'replay_rank': None,
            'final_text_if_upgraded': vlm_raw_output if upgrade else '',
            'final_text': final_text,
            'backfill_status': 'skipped' if upgrade else 'not_upgraded',
            'backfill_reason': 'paper1_routeronly_reuse_cache' if upgrade else 'not_upgraded',
            'is_correct_ocr': t_a == t_gt,
            'is_correct_final': final_text == t_gt,
            'edit_distance_ocr': Levenshtein.distance(t_a, t_gt),
            'edit_distance_final': Levenshtein.distance(final_text, t_gt),
            'run_id': run_id,
        }
        per_sample.append(item)
        validation_logs.append({
            'sample_id': item['sample_id'],
            'router_name': strategy,
            'target_budget': target_budget,
            'router_score': q,
            'selected_for_upgrade': upgrade,
            'budget_details': budget_details,
            'cached_result_found': cached is not None,
            'run_id': run_id,
        })

    extended_metrics = summarize_extended_metrics(per_sample)
    usage_metrics = summarize_latency_and_token_usage(per_sample)
    actual_call_rate = (n_upgraded / len(all_results)) if all_results else 0.0
    return {
        'summary': {
            'run_id': run_id,
            'router_name': strategy,
            'budget': target_budget,
            'target_call_rate': target_budget,
            'actual_call_rate': round(actual_call_rate, 4),
            'call_rate_valid': abs(actual_call_rate - target_budget) <= 0.005,
            'CER': round(cer_num / gt_len, 6) if gt_len else 0.0,
            'BoundaryDeletionRecallAtB': extended_metrics['Boundary_Deletion_Recall@B'],
            'SubstitutionCER': extended_metrics['Substitution_CER'],
            'AER': round(n_accepted / n_upgraded, 4) if n_upgraded else 0.0,
            'CVR': 0.0,
            'p95_latency_ms': usage_metrics['P95_Latency_MS'],
            'avg_token_usage': usage_metrics['Avg_Token_Usage'],
            'agentB_model': agent_b_label,
            'prompt_version': prompt_version,
            'n_valid': len(all_results),
            'missing_cached_rows': missing_cached_rows,
        },
        'per_sample': per_sample,
        'validation_logs': validation_logs,
    }


def main():
    p = argparse.ArgumentParser(description='paper1 RouterOnly online legality check (reuse Main C cache)')
    p.add_argument('--config', default='configs/router_config.yaml')
    p.add_argument('--test_jsonl', default='data/l2w1data/test.jsonl')
    p.add_argument('--image_root', default='data/l2w1data/images')
    p.add_argument('--rec_model_dir', default='./models/agent_a_ppocr/PP-OCRv5_server_rec_infer')
    p.add_argument('--rec_char_dict_path', default='ppocr/utils/ppocrv5_dict.txt')
    p.add_argument('--geo_dict', default='data/dicts/Geology.txt')
    p.add_argument('--finance_dict', default='data/dicts/Finance.txt')
    p.add_argument('--medicine_dict', default='data/dicts/Medicine.txt')
    p.add_argument('--output_dir', default='paper1_runs/online_budget_validation')
    p.add_argument('--mainc_root', default='paper1_runs/mainC')
    p.add_argument('--mainc_run_dir', default=None)
    p.add_argument('--full_call_cache_path', default=None)
    p.add_argument('--mainc_model_id', default='V3', choices=['V1', 'V2', 'V3', 'V4'])
    p.add_argument('--strategy', default='GCR', choices=['GCR', 'WUR', 'DGCR', 'DWUR'])
    p.add_argument('--target_budget', type=float, default=0.10)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n_samples', type=int, default=None)
    p.add_argument('--use_gpu', action='store_true', default=False)
    p.add_argument('--use_cache', action='store_true', default=False)
    p.add_argument('--rebuild_cache', action='store_true', default=False)
    p.add_argument('--prompt_version', default=None)
    a = p.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    config = yaml.safe_load(Path(a.config).read_text(encoding='utf-8'))
    prompt_version = a.prompt_version or config.get('prompt_version') or config.get('mainline', {}).get('prompt_version', 'prompt_v1.1')
    agent_b_label = MODEL_LABELS.get(a.mainc_model_id, a.mainc_model_id)

    import argparse as _ap
    rec_args = _ap.Namespace(rec_model_dir=a.rec_model_dir, rec_char_dict_path=a.rec_char_dict_path, rec_image_shape='3, 48, 320', rec_batch_num=6, rec_algorithm='SVTR_LCNet', use_space_char=True, use_gpu=a.use_gpu, use_xpu=False, use_npu=False, use_mlu=False, use_metax_gpu=False, use_gcu=False, ir_optim=True, use_tensorrt=False, min_subgraph_size=15, precision='fp32', gpu_mem=500, gpu_id=0, enable_mkldnn=None, cpu_threads=10, warmup=False, benchmark=False, save_log_path='./log_output/', show_log=False, use_onnx=False, max_batch_size=10, return_word_box=False, drop_score=0.5, max_text_length=25, rec_image_inverse=True, use_det=False, det_model_dir='')

    from modules.paddle_engine.predict_rec_modified import TextRecognizerWithLogits
    from modules.router.domain_knowledge import DomainKnowledgeEngine

    output_dir = Path(a.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    agent_a_cache_path = Path(a.mainc_root) / 'shared_agent_a_cache.json'

    if a.use_cache and not a.rebuild_cache and agent_a_cache_path.exists():
        all_results = ensure_agent_a_result_schema(json.loads(agent_a_cache_path.read_text(encoding='utf-8')))
    else:
        recognizer = TextRecognizerWithLogits(rec_args)
        domain_engine = DomainKnowledgeEngine({'geology': a.geo_dict, 'finance': a.finance_dict, 'medicine': a.medicine_dict})
        samples = [json.loads(x) for x in Path(a.test_jsonl).read_text(encoding='utf-8').splitlines() if x.strip()]
        all_results = ensure_agent_a_result_schema(infer_all_samples(samples, recognizer, domain_engine, None, a.image_root))
        agent_a_cache_path.write_text(json.dumps(all_results, ensure_ascii=False), encoding='utf-8')

    if a.n_samples and a.n_samples < len(all_results):
        all_results = all_results[:a.n_samples]

    if a.full_call_cache_path:
        full_call_path = Path(a.full_call_cache_path)
    elif a.mainc_run_dir:
        full_call_path = Path(a.mainc_run_dir) / f'{a.mainc_model_id}_full_call_cache.jsonl'
    else:
        full_call_path = Path(a.mainc_root) / f'{a.mainc_model_id}_full_call_cache.jsonl'
    if not full_call_path.exists():
        raise FileNotFoundError(f'Main C full-call cache not found: {full_call_path}')
    cached_lookup = build_cached_result_lookup(load_jsonl(full_call_path))

    bc = (config or {}).get('sh_da_v4', {}).get('budget_controller', {})
    budget_cfg = BudgetControllerConfig(target_budget=a.target_budget, lambda_init=bc.get('lambda_init', 0.5), lambda_min=bc.get('lambda_min', 0.0), lambda_max=bc.get('lambda_max', 1.0), k=bc.get('k', bc.get('alpha', 0.01)), window_size=bc.get('window_size', 500), warmup_samples=bc.get('warmup_samples'))
    eta = float((config or {}).get('sh_da_v4', {}).get('rule_scorer', {}).get('eta', 0.5))

    run_id = datetime.now().strftime('%Y%m%d_run%H%M%S')
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    result = run_online_routeronly(a.strategy, a.target_budget, all_results, cached_lookup, budget_cfg, run_id=run_id, prompt_version=prompt_version, agent_b_label=agent_b_label, eta=eta)

    with (run_dir / 'online_budget_validation.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(result['summary'].keys()))
        writer.writeheader()
        writer.writerow(result['summary'])
    with (run_dir / 'online_budget_validation_logs.jsonl').open('w', encoding='utf-8') as f:
        for row in result['validation_logs']:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    with (run_dir / f'online_budget_{int(round(a.target_budget * 100)):02d}_{a.strategy}.jsonl').open('w', encoding='utf-8') as f:
        for row in result['per_sample']:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    (run_dir / 'config_snapshot.yaml').write_text(yaml.safe_dump({'run_id': run_id, 'args': vars(a), 'config': config, 'mainc_full_call_path': str(full_call_path)}, allow_unicode=True, sort_keys=False), encoding='utf-8')
    print(run_dir)


if __name__ == '__main__':
    main()
