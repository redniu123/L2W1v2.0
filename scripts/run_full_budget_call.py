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
from scripts.run_efficiency_frontier import (
    build_agent_b_callable,
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


def gain_rows(ocr_cer, metrics_rows):
    ordered = sorted(metrics_rows, key=lambda x: x['budget'])
    best_idx, best_gain = -1, -1e9
    prev_budget, prev_cer = 0.0, ocr_cer
    out = []
    for row in ordered:
        b, cer = float(row['budget']), float(row['overall_cer'])
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


def main():
    p = argparse.ArgumentParser(description='Run 100% full-call and budget gain analysis')
    p.add_argument('--config', default='configs/router_config.yaml')
    p.add_argument('--test_jsonl', default='data/l2w1data/test.jsonl')
    p.add_argument('--image_root', default='data/l2w1data/images')
    p.add_argument('--rec_model_dir', default='./models/agent_a_ppocr/PP-OCRv5_server_rec_infer')
    p.add_argument('--rec_char_dict_path', default='ppocr/utils/ppocrv5_dict.txt')
    p.add_argument('--geo_dict', default='data/dicts/Geology.txt')
    p.add_argument('--finance_dict', default='data/dicts/Finance.txt')
    p.add_argument('--medicine_dict', default='data/dicts/Medicine.txt')
    p.add_argument('--output_dir', default='results/expriments/full_budget/02_runs')
    p.add_argument('--cache_path', default='results/expriments/full_budget/02_runs/agent_a_cache.json')
    p.add_argument('--strategy', default='GCR', choices=['GCR', 'WUR', 'DGCR', 'DWUR', 'BAUR', 'DAR', 'Router-only', 'SH-DA++'])
    p.add_argument('--replay_budgets', nargs='+', type=float, default=[0.05, 0.10, 0.20, 0.30, 0.50, 0.80, 1.00])
    p.add_argument('--n_samples', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--use_gpu', action='store_true', default=False)
    p.add_argument('--use_cache', action='store_true', default=True)
    p.add_argument('--rebuild_cache', action='store_true', default=False)
    args = p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed)

    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    prompt_version = config.get('prompt_version') or config.get('mainline', {}).get('prompt_version', 'prompt_v1.1')
    rows = load_rows(args)
    backfill = StrictBackfillController(BackfillConfig())
    from modules.router.sh_da_router import SHDARouter
    from modules.vlm_expert.constrained_prompter import ConstrainedPrompter
    router = SHDARouter.from_yaml(args.config)
    prompter = ConstrainedPrompter()
    agent_b = build_agent_b_callable(config)

    run_id = datetime.now().strftime('%Y%m%d_run%H%M%S')
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'config_snapshot.yaml').write_text(yaml.safe_dump({'run_id': run_id, 'strategy': args.strategy, 'replay_budgets': args.replay_budgets, 'prompt_version': prompt_version}, allow_unicode=True, sort_keys=False), encoding='utf-8')

    base = run_pipeline('AgentA_Only', 0.0, rows, router, backfill, prompter, agent_b, run_id=run_id, prompt_version=prompt_version)
    ocr_cer = float(base['summary']['Overall_CER'])
    full = run_pipeline(args.strategy, 1.0, rows, router, backfill, prompter, agent_b, run_id=run_id, prompt_version=prompt_version, agent_b_label=config.get('mainline_agent_b', 'configured_agent_b'))
    jlines(run_dir / f'full_budget_results_{args.strategy}.jsonl', full['per_sample'])

    metrics = []
    for budget in args.replay_budgets:
        replay = replay_from_full_budget(args.strategy, budget, full['per_sample'], run_id=run_id, prompt_version=prompt_version)
        row = {
            'run_id': run_id,
            'strategy': args.strategy,
            'budget': budget,
            'actual_call_rate': replay['summary']['Actual_Call_Rate'],
            'overall_cer': replay['summary']['Overall_CER'],
            'boundary_deletion_recall_at_b': replay['summary']['Boundary_Deletion_Recall@B'],
            'substitution_cer': replay['summary']['Substitution_CER'],
            'aer': replay['summary']['AER'],
            'cvr': replay['summary']['CVR'],
            'num_samples': replay['summary']['N_valid'],
        }
        metrics.append(row)
        jlines(run_dir / f"offline_budget_{int(round(budget * 100)):02d}_{args.strategy}.jsonl", replay['per_sample'])

    gain_analysis = gain_rows(ocr_cer, metrics)
    wcsv(run_dir / 'budget_gain_analysis.csv', list(gain_analysis[0].keys()) if gain_analysis else ['run_id'], gain_analysis)
    wcsv(run_dir / 'summary.csv', ['run_id', 'strategy', 'budget', 'actual_call_rate', 'overall_cer', 'boundary_deletion_recall_at_b', 'substitution_cer', 'aer', 'cvr', 'num_samples'], metrics)
    jdump(run_dir / 'metrics_summary.json', {'run_id': run_id, 'strategy': args.strategy, 'ocr_cer': ocr_cer, 'results': metrics, 'gain_analysis': gain_analysis})
    print(run_dir)


if __name__ == '__main__':
    main()
