#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_efficiency_frontier import (
    collect_case_rows,
    ensure_agent_a_result_schema,
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


def default_budgets():
    return [round(i / 100.0, 2) for i in range(1, 101)]


def load_agent_a_rows(cache_path, n_samples=None):
    rows = ensure_agent_a_result_schema(json.loads(Path(cache_path).read_text(encoding='utf-8')))
    return rows[:n_samples] if n_samples else rows


def load_vlm_cache(cache_path):
    items = [json.loads(line) for line in Path(cache_path).read_text(encoding='utf-8').splitlines() if line.strip()]
    by_sample = {}
    for item in items:
        sample_id = item.get('sample_id')
        if sample_id:
            by_sample[sample_id] = item
    return by_sample


def wur_score(row):
    mean_conf = float(row.get('mean_conf', row.get('conf', 0.0)))
    min_conf = float(row.get('min_conf', row.get('conf', 0.0)))
    drop = float(row.get('drop', 0.0))
    q = 0.5 * (1.0 - mean_conf) + 0.3 * (1.0 - min_conf) + 0.2 * drop
    if min_conf < 0.35:
        q += 0.10
    if drop > 0.20:
        q += 0.10
    return float(q)


def build_full_budget_items(agent_rows, vlm_cache_by_sample, run_id, prompt_version):
    full_items = []
    missing = []
    for row in agent_rows:
        sample_id = row.get('sample_id', '')
        cache_item = vlm_cache_by_sample.get(sample_id)
        if cache_item is None:
            missing.append(sample_id)
            continue

        T_A = row.get('T_A', '')
        T_GT = row.get('T_GT', '')
        vlm_raw_output = cache_item.get('vlm_raw_output', T_A)
        router_score = wur_score(row)
        edit_distance_ocr = Levenshtein.distance(T_A, T_GT)
        edit_distance_final = Levenshtein.distance(vlm_raw_output, T_GT)

        full_items.append({
            'sample_id': sample_id,
            'image_path': row.get('image_path', ''),
            'source_image_id': row.get('source_image_id', ''),
            'domain': row.get('domain', 'geology'),
            'split': row.get('split', 'test'),
            'gt': T_GT,
            'ocr_text': T_A,
            'router_name': 'Router-only',
            'router_score': round(router_score, 6),
            'budget': 1.0,
            'budget_mode': 'offline_cache_merge',
            'selected_for_upgrade': True,
            'vlm_model': cache_item.get('vlm_model', ''),
            'prompt_version': cache_item.get('prompt_version', prompt_version),
            'vlm_raw_output': vlm_raw_output,
            'latency_ms': cache_item.get('latency_ms'),
            'token_usage': cache_item.get('token_usage'),
            'error_type': cache_item.get('error_type', 'none'),
            'has_professional_terms': row.get('has_professional_terms', False),
            'professional_terms': row.get('professional_terms', []),
            'domain_risk_score': round(float(row.get('r_d', 0.0)), 6),
            'cvr_flag': False,
            'replay_rank': None,
            'final_text_if_upgraded': vlm_raw_output,
            'final_text': vlm_raw_output,
            'backfill_status': 'skipped',
            'backfill_reason': 'router_only_no_backfill',
            'is_correct_ocr': T_A == T_GT,
            'is_correct_final': vlm_raw_output == T_GT,
            'edit_distance_ocr': edit_distance_ocr,
            'edit_distance_final': edit_distance_final,
            'run_id': run_id,
        })

    if missing:
        raise ValueError(f'Missing M5 cache items for {len(missing)} samples, first few: {missing[:10]}')
    return full_items


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
        rows.append({
            'run_id': run_id,
            'budget': budget,
            'domain': domain,
            'n_samples': stat['n_samples'],
            'final_accuracy': round(stat['correct_final'] / stat['n_samples'], 6) if stat['n_samples'] else 0.0,
        })
    return rows


def case_stat_row(run_id, budget, per_sample):
    upgraded = [x for x in per_sample if x.get('selected_for_upgrade')]
    improved = sum(1 for x in upgraded if int(x.get('edit_distance_final', 0)) < int(x.get('edit_distance_ocr', 0)))
    unchanged = sum(1 for x in upgraded if int(x.get('edit_distance_final', 0)) == int(x.get('edit_distance_ocr', 0)))
    worsened = sum(1 for x in upgraded if int(x.get('edit_distance_final', 0)) > int(x.get('edit_distance_ocr', 0)))
    return {
        'run_id': run_id,
        'budget': budget,
        'n_upgraded': len(upgraded),
        'n_improved': improved,
        'n_unchanged': unchanged,
        'n_worsened': worsened,
    }


def main():
    p = argparse.ArgumentParser(description='Phase A: valid M5 1% budget scan from Agent A + M5 cache')
    p.add_argument('--config', default='configs/router_config.yaml')
    p.add_argument('--output_dir', default='cloud_result_sync/exB_phaseA/02_runs')
    p.add_argument('--cache_path', default='cloud_result_sync/exB/02_runs/agent_a_cache.json')
    p.add_argument('--full_call_cache_path', default='cloud_result_sync/exB/02_runs/m5_full_call_cache.jsonl')
    p.add_argument('--n_samples', type=int, default=None)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    config = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    prompt_version = config.get('prompt_version') or config.get('mainline', {}).get('prompt_version', 'prompt_v1.1')

    agent_rows = load_agent_a_rows(args.cache_path, args.n_samples)
    vlm_cache_by_sample = load_vlm_cache(args.full_call_cache_path)

    run_id = datetime.now().strftime('%Y%m%d_run%H%M%S')
    run_dir = Path(args.output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    budgets = default_budgets()

    (run_dir / 'config_snapshot.yaml').write_text(yaml.safe_dump({
        'run_id': run_id,
        'strategy': 'M5',
        'replay_budgets': budgets,
        'prompt_version': prompt_version,
        'agent_a_cache_path': args.cache_path,
        'm5_cache_path': args.full_call_cache_path,
        'phase': 'A',
        'replay_source': 'agent_a_cache_plus_m5_vlm_cache',
    }, allow_unicode=True, sort_keys=False), encoding='utf-8')

    base = run_pipeline('AgentA_Only', 0.0, agent_rows, None, None, None, None, run_id=run_id, prompt_version=prompt_version)
    ocr_cer = float(base['summary']['Overall_CER'])
    full_items = build_full_budget_items(agent_rows, vlm_cache_by_sample, run_id, prompt_version)

    jlines(run_dir / 'full_budget_results_M5.jsonl', full_items)
    metrics, case_stats = [], []
    for budget in budgets:
        replay = replay_from_full_budget('Router-only', budget, full_items, run_id=run_id, prompt_version=prompt_version)
        per_sample = replay['per_sample']
        pct = int(round(budget * 100))
        stem = f'M5_offline_budget_{pct:02d}'
        jlines(run_dir / f'{stem}_results.jsonl', per_sample)
        jdump(run_dir / f'{stem}_summary.json', replay['summary'])
        wcsv(
            run_dir / f'{stem}_domain_breakdown.csv',
            ['run_id', 'budget', 'domain', 'n_samples', 'final_accuracy'],
            domain_rows(per_sample, run_id, budget),
        )
        failures, degradations = [], []
        for item in per_sample:
            collect_case_rows(item, failures, degradations)
        wcsv(
            run_dir / f'{stem}_failure_cases.csv',
            ['sample_id', 'router_name', 'budget', 'domain', 'ocr_text', 'final_text', 'gt', 'backfill_status', 'backfill_reason', 'edit_distance_ocr', 'edit_distance_final'],
            failures,
        )
        metrics.append({
            'run_id': run_id,
            'strategy': 'M5',
            'budget': budget,
            'actual_call_rate': replay['summary']['Actual_Call_Rate'],
            'overall_cer': replay['summary']['Overall_CER'],
            'boundary_deletion_recall_at_b': replay['summary']['Boundary_Deletion_Recall@B'],
            'substitution_cer': replay['summary']['Substitution_CER'],
            'aer': replay['summary']['AER'],
            'cvr': replay['summary']['CVR'],
            'num_samples': replay['summary']['N_valid'],
        })
        case_stats.append(case_stat_row(run_id, budget, per_sample))

    gain_analysis = gain_rows(ocr_cer, metrics)
    wcsv(
        run_dir / 'tab_m5_budget_curve.csv',
        ['run_id', 'strategy', 'budget', 'actual_call_rate', 'overall_cer', 'boundary_deletion_recall_at_b', 'substitution_cer', 'aer', 'cvr', 'num_samples'],
        metrics,
    )
    wcsv(run_dir / 'tab_m5_budget_gain_curve.csv', list(gain_analysis[0].keys()), gain_analysis)
    wcsv(run_dir / 'tab_m5_budget_case_stats.csv', list(case_stats[0].keys()), case_stats)
    jdump(run_dir / 'metrics_summary.json', {
        'run_id': run_id,
        'strategy': 'M5',
        'ocr_cer': ocr_cer,
        'agent_a_cache_path': args.cache_path,
        'm5_cache_path': args.full_call_cache_path,
        'results': metrics,
        'gain_analysis': gain_analysis,
        'case_stats': case_stats,
    })
    print(run_dir)


if __name__ == '__main__':
    main()
