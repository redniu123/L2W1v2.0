#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, csv, json, random, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.router.backfill import BackfillConfig, StrictBackfillController
from modules.router.domain_knowledge import DomainKnowledgeEngine
from modules.paddle_engine.predict_rec_modified import TextRecognizerWithLogits
from modules.router.sh_da_router import SHDARouter
from modules.vlm_expert.constrained_prompter import ConstrainedPrompter
from scripts.run_efficiency_frontier import (
    build_agent_b_callable,
    collect_case_rows,
    ensure_agent_a_result_schema,
    infer_all_samples,
    replay_from_full_budget,
    run_pipeline,
)


def jdump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def jlines(path, rows):
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def wcsv(path, fieldnames, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})


def load_rows(args):
    cache = Path(args.cache_path)
    if args.use_cache and cache.exists() and not args.rebuild_cache:
        rows = ensure_agent_a_result_schema(json.loads(cache.read_text(encoding='utf-8')))
        return rows[:args.n_samples] if args.n_samples else rows
    samples = [json.loads(line) for line in Path(args.test_jsonl).read_text(encoding='utf-8').splitlines() if line.strip()]
    import argparse as _ap
    rec_args = _ap.Namespace(rec_model_dir=args.rec_model_dir, rec_char_dict_path=args.rec_char_dict_path, rec_image_shape='3, 48, 320', rec_batch_num=6, rec_algorithm='SVTR_LCNet', use_space_char=True, use_gpu=args.use_gpu, use_xpu=False, use_npu=False, use_mlu=False, use_metax_gpu=False, use_gcu=False, ir_optim=True, use_tensorrt=False, min_subgraph_size=15, precision='fp32', gpu_mem=500, gpu_id=0, enable_mkldnn=None, cpu_threads=10, warmup=False, benchmark=False, save_log_path='./log_output/', show_log=False, use_onnx=False, max_batch_size=10, return_word_box=False, drop_score=0.5, max_text_length=25, rec_image_inverse=True, use_det=False, det_model_dir='')
    recognizer = TextRecognizerWithLogits(rec_args)
    domain_engine = DomainKnowledgeEngine({'geology': args.geo_dict, 'finance': args.finance_dict, 'medicine': args.medicine_dict})
    rows = ensure_agent_a_result_schema(infer_all_samples(samples, recognizer, domain_engine, None, args.image_root))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
    return rows[:args.n_samples] if args.n_samples else rows


def default_budgets():
    return [round(i / 100.0, 2) for i in range(1, 101)]


def gain_rows(ocr_cer, metrics_rows):
    ordered = sorted(metrics_rows, key=lambda x: x['budget'])
    prev_budget, prev_cer = 0.0, ocr_cer
    best_idx, best_gain = -1, -1e9
    out = []
    for row in ordered:
        b = float(row['budget'])
        cer = float(row['overall_cer'])
        cer_gain_vs_ocr = ocr_cer - cer
        delta_cer_vs_prev = prev_cer - cer
        extra_budget = b - prev_budget
        gain_per_extra = (delta_cer_vs_prev / extra_budget) if extra_budget > 0 else 0.0
        cur = {
            **row,
            'cer_gain_vs_ocr': round(cer_gain_vs_ocr, 6),
            'delta_cer_vs_prev_budget': round(delta_cer_vs_prev, 6),
            'gain_per_extra_budget': round(gain_per_extra, 6),
            'is_best_gain_point': False,
        }
        out.append(cur)
        if gain_per_extra > best_gain:
            best_gain, best_idx = gain_per_extra, len(out) - 1
        prev_budget, prev_cer = b, cer
    if best_idx >= 0:
        out[best_idx]['is_best_gain_point'] = True
    return out


def domain_rows(per_sample, run_id, budget):
    stats = {}
    for item in per_sample:
        key = item.get('domain', 'unknown')
        stat = stats.setdefault(key, {'n_samples': 0, 'correct_final': 0})
        stat['n_samples'] += 1
        stat['correct_final'] += 1 if item.get('is_correct_final') else 0
    rows = []
    for domain, stat in sorted(stats.items()):
        rows.append({'run_id': run_id, 'budget': budget, 'domain': domain, 'n_samples': stat['n_samples'], 'final_accuracy': round(stat['correct_final'] / stat['n_samples'], 6) if stat['n_samples'] else 0.0})
    return rows


def case_stat_row(run_id, budget, per_sample):
    upgraded = [x for x in per_sample if x.get('selected_for_upgrade')]
    improved = sum(1 for x in upgraded if int(x.get('edit_distance_final', 0)) < int(x.get('edit_distance_ocr', 0)))
    unchanged = sum(1 for x in upgraded if int(x.get('edit_distance_final', 0)) == int(x.get('edit_distance_ocr', 0)))
    worsened = sum(1 for x in upgraded if int(x.get('edit_distance_final', 0)) > int(x.get('edit_distance_ocr', 0)))
    return {'run_id': run_id, 'budget': budget, 'n_upgraded': len(upgraded), 'n_improved': improved, 'n_unchanged': unchanged, 'n_worsened': worsened}


def load_or_make_full_call(path_str, rebuild, rows, router, backfill, prompter, agent_b, run_id, prompt_version, agent_b_label):
    path = Path(path_str)
    if path.exists() and not rebuild:
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    full = run_pipeline('Router-only', 1.0, rows, router, backfill, prompter, agent_b, run_id=run_id, prompt_version=prompt_version, agent_b_label=agent_b_label)
    path.parent.mkdir(parents=True, exist_ok=True)
    jlines(path, full['per_sample'])
    return full['per_sample']


def main():
    p = argparse.ArgumentParser(description='Phase A: M5 1% budget scan')
    p.add_argument('--config', default='configs/router_config.yaml')
    p.add_argument('--test_jsonl', default='data/l2w1data/test.jsonl')
    p.add_argument('--image_root', default='data/l2w1data/images')
    p.add_argument('--rec_model_dir', default='./models/agent_a_ppocr/PP-OCRv5_server_rec_infer')
    p.add_argument('--rec_char_dict_path', default='ppocr/utils/ppocrv5_dict.txt')
    p.add_argument('--geo_dict', default='data/dicts/Geology.txt')
    p.add_argument('--finance_dict', default='data/dicts/Finance.txt')
    p.add_argument('--medicine_dict', default='data/dicts/Medicine.txt')
    p.add_argument('--output_dir', default='results/expriments/exB_phaseA/02_runs')
    p.add_argument('--cache_path', default='results/expriments/exB_phaseA/02_runs/agent_a_cache.json')
    p.add_argument('--full_call_cache_path', default='results/expriments/exB_phaseA/02_runs/m5_full_budget_results.jsonl')
    p.add_argument('--reuse_full_call_cache', action='store_true', default=False)
    p.add_argument('--rebuild_full_call_cache', action='store_true', default=False)
    p.add_argument('--prepare_full_call_cache_only', action='store_true', default=False)
    p.add_argument('--n_samples', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--use_gpu', action='store_true', default=False)
    p.add_argument('--use_cache', action='store_true', default=True)
    p.add_argument('--rebuild_cache', action='store_true', default=False)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    prompt_version = config.get('prompt_version') or config.get('mainline', {}).get('prompt_version', 'prompt_v1.1')
    rows = load_rows(args)
    backfill = StrictBackfillController(BackfillConfig())
    router = SHDARouter.from_yaml(args.config)
    prompter = ConstrainedPrompter()
    agent_b = build_agent_b_callable(config)
    run_id = datetime.now().strftime('%Y%m%d_run%H%M%S')
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    budgets = default_budgets()
    (run_dir / 'config_snapshot.yaml').write_text(yaml.safe_dump({'run_id': run_id, 'strategy': 'Router-only', 'replay_budgets': budgets, 'prompt_version': prompt_version, 'full_call_cache_path': args.full_call_cache_path, 'phase': 'A'}, allow_unicode=True, sort_keys=False), encoding='utf-8')

    base = run_pipeline('AgentA_Only', 0.0, rows, router, backfill, prompter, agent_b, run_id=run_id, prompt_version=prompt_version)
    ocr_cer = float(base['summary']['Overall_CER'])

    full_items = load_or_make_full_call(args.full_call_cache_path, (args.rebuild_full_call_cache or not args.reuse_full_call_cache), rows, router, backfill, prompter, agent_b, run_id, prompt_version, config.get('mainline_agent_b', 'configured_agent_b'))
    if args.prepare_full_call_cache_only:
        print(f'Full-call cache ready: {args.full_call_cache_path}')
        print(run_dir)
        return

    jlines(run_dir / 'full_budget_results_M5.jsonl', full_items)
    metrics, case_stats = [], []
    for budget in budgets:
        replay = replay_from_full_budget('Router-only', budget, full_items, run_id=run_id, prompt_version=prompt_version)
        per_sample = replay['per_sample']
        pct = int(round(budget * 100))
        stem = f'M5_offline_budget_{pct:02d}'
        jlines(run_dir / f'{stem}_results.jsonl', per_sample)
        jdump(run_dir / f'{stem}_summary.json', replay['summary'])
        wcsv(run_dir / f'{stem}_domain_breakdown.csv', ['run_id', 'budget', 'domain', 'n_samples', 'final_accuracy'], domain_rows(per_sample, run_id, budget))
        failures, degradations = [], []
        for item in per_sample:
            collect_case_rows(item, failures, degradations)
        wcsv(run_dir / f'{stem}_failure_cases.csv', ['sample_id', 'router_name', 'budget', 'domain', 'ocr_text', 'final_text', 'gt', 'backfill_status', 'backfill_reason', 'edit_distance_ocr', 'edit_distance_final'], failures)
        metrics.append({'run_id': run_id, 'strategy': 'M5', 'budget': budget, 'actual_call_rate': replay['summary']['Actual_Call_Rate'], 'overall_cer': replay['summary']['Overall_CER'], 'boundary_deletion_recall_at_b': replay['summary']['Boundary_Deletion_Recall@B'], 'substitution_cer': replay['summary']['Substitution_CER'], 'aer': replay['summary']['AER'], 'cvr': replay['summary']['CVR'], 'num_samples': replay['summary']['N_valid']})
        case_stats.append(case_stat_row(run_id, budget, per_sample))

    gain_analysis = gain_rows(ocr_cer, metrics)
    wcsv(run_dir / 'tab_m5_budget_curve.csv', ['run_id', 'strategy', 'budget', 'actual_call_rate', 'overall_cer', 'boundary_deletion_recall_at_b', 'substitution_cer', 'aer', 'cvr', 'num_samples'], metrics)
    wcsv(run_dir / 'tab_m5_budget_gain_curve.csv', list(gain_analysis[0].keys()), gain_analysis)
    wcsv(run_dir / 'tab_m5_budget_case_stats.csv', list(case_stats[0].keys()), case_stats)
    jdump(run_dir / 'metrics_summary.json', {'run_id': run_id, 'strategy': 'M5', 'ocr_cer': ocr_cer, 'full_call_cache_path': args.full_call_cache_path, 'results': metrics, 'gain_analysis': gain_analysis, 'case_stats': case_stats})
    print(run_dir)


if __name__ == '__main__':
    main()
