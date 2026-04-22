#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Merge split Main C runs into one formal final run."""
import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Levenshtein

from scripts.run_efficiency_frontier import summarize_extended_metrics, summarize_latency_and_token_usage


def norm(t):
    return '' if not t else t.translate(str.maketrans({'（':'(','）':')','【':'[','】':']','｛':'{','｝':'}','，':',','：':':','；':';','！':'!','？':'?','。':'.'}))


def rjsonl(p: Path):
    return [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]


def wjsonl(p: Path, rows):
    with p.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def wcsv(p: Path, rows):
    if not rows:
        p.write_text('', encoding='utf-8')
        return
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def normalize_cache(rows, name, pv, run_id):
    out = []
    for r in rows:
        d = dict(r)
        ta = d.get('ocr_text', d.get('T_A', ''))
        tg = d.get('gt', d.get('T_GT', ''))
        up = d.get('final_text_if_upgraded', d.get('final_text', ta))
        d.update({
            'ocr_text': ta,
            'gt': tg,
            'final_text_if_upgraded': up,
            'vlm_raw_output': d.get('vlm_raw_output', up),
            'latency_ms': d.get('latency_ms'),
            'token_usage': d.get('token_usage'),
            'error_type': d.get('error_type', 'none'),
            'has_professional_terms': d.get('has_professional_terms', False),
            'professional_terms': d.get('professional_terms', []),
            'is_correct_ocr': d.get('is_correct_ocr', ta == tg),
            'edit_distance_ocr': d.get('edit_distance_ocr', Levenshtein.distance(norm(ta), norm(tg))),
            'vlm_model': d.get('vlm_model', name),
            'prompt_version': d.get('prompt_version', pv),
            'run_id': d.get('run_id', run_id),
        })
        out.append(d)
    return out


def build_budget_rows(full, ranked, model_name, run_id, prompt_version):
    budgets = [0.05, 0.10, 0.20, 0.30, 0.50, 0.80]
    for b in budgets:
        n = int(round(len(full) * b))
        up = set(ranked[:n])
        rmap = {i: k + 1 for k, i in enumerate(ranked)}
        ps = []
        cer = 0
        gtl = 0
        nup = 0
        nacc = 0
        for i, it in enumerate(full):
            ta, tg = it['ocr_text'], it['gt']
            sel = i in up
            ft = it['final_text_if_upgraded'] if sel else ta
            if sel:
                nup += 1
                nacc += 1 if ft != ta else 0
            cer += Levenshtein.distance(norm(ft), norm(tg))
            gtl += len(norm(tg))
            row = dict(it)
            row.update({
                'model_name': model_name,
                'router_name': 'GCR',
                'budget': b,
                'target_call_rate': b,
                'selected_for_upgrade': sel,
                'replay_rank': rmap[i],
                'final_text': ft,
                'latency_ms': it['latency_ms'] if sel else None,
                'token_usage': it['token_usage'] if sel else None,
                'error_type': it['error_type'] if sel else 'not_upgraded',
                'backfill_status': 'skipped' if sel else 'not_upgraded',
                'backfill_reason': 'paper1_routeronly',
                'is_correct_final': ft == tg,
                'edit_distance_final': Levenshtein.distance(norm(ft), norm(tg)),
            })
            ps.append(row)
        ex = summarize_extended_metrics(ps)
        us = summarize_latency_and_token_usage(ps)
        ar = (nup / len(full)) if full else 0.0
        summary = {
            'run_id': run_id,
            'model_name': model_name,
            'router_name': 'GCR',
            'budget': b,
            'target_call_rate': b,
            'actual_call_rate': round(ar, 4),
            'call_rate_valid': abs(ar - b) <= 0.005,
            'CER': round(cer / gtl, 6) if gtl else 0.0,
            'BoundaryDeletionRecallAtB': ex['Boundary_Deletion_Recall@B'],
            'SubstitutionCER': ex['Substitution_CER'],
            'AER': round(nacc / nup, 4) if nup else 0.0,
            'CVR': 0.0,
            'p95_latency_ms': us['P95_Latency_MS'],
            'avg_token_usage': us['Avg_Token_Usage'],
            'prompt_version': prompt_version,
            'n_valid': len(full),
        }
        yield b, ps, summary


def main():
    p = argparse.ArgumentParser(description='Merge split Main C runs into one final run')
    p.add_argument('--output_dir', default='paper1_runs/mainC_fixed')
    p.add_argument('--v1_run', required=True)
    p.add_argument('--v2_run', required=True)
    p.add_argument('--v34_run', required=True)
    p.add_argument('--maina_run_dir', required=True)
    p.add_argument('--prompt_version', default='prompt_v1.1')
    a = p.parse_args()

    out_root = Path(a.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime('%Y%m%d_run%H%M%S')
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    maina_run = Path(a.maina_run_dir)
    full_agentb = rjsonl(maina_run / 'shared_repmodel_full_call_cache.jsonl')
    ranked = sorted(range(len(full_agentb)), key=lambda i: float(full_agentb[i].get('router_score', 0.0)), reverse=True)

    sources = {
        'V1': (Path(a.v1_run), 'Qwen3-VL-8B'),
        'V2': (Path(a.v2_run), 'MiniCPM-V 4.5'),
        'V3': (Path(a.v34_run), 'Gemini 3 Flash Preview'),
        'V4': (Path(a.v34_run), 'gpt-5.4'),
    }

    main_rows = []
    manifest = {
        'maina_run_dir': str(maina_run),
        'sources': {k: str(v[0]) for k, v in sources.items()},
        'models': {},
    }

    for prefix, (src_dir, model_name) in sources.items():
        src_cache = src_dir / f'{prefix}_full_call_cache.jsonl'
        if not src_cache.exists():
            raise FileNotFoundError(f'missing source cache: {src_cache}')
        full = normalize_cache(rjsonl(src_cache), model_name, a.prompt_version, run_id)
        wjsonl(run_dir / f'{prefix}_full_call_cache.jsonl', full)
        for b, ps, summary in build_budget_rows(full, ranked, model_name, run_id, a.prompt_version):
            main_rows.append(summary)
            wjsonl(run_dir / f'{prefix}_offline_budget_{int(round(b * 100)):02d}.jsonl', ps)
        manifest['models'][prefix] = {
            'model_name': model_name,
            'src_cache': str(src_cache),
            'n_valid': len(full),
        }

    wcsv(run_dir / 'tab_mainC_results.csv', main_rows)
    wcsv(run_dir / 'tab_mainC_budget_check.csv', [r for r in main_rows if round(float(r['budget']), 2) in {0.10, 0.20, 0.30}])
    (run_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(run_dir)


if __name__ == '__main__':
    main()
