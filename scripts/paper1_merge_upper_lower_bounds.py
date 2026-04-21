#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse,csv,json
from pathlib import Path
import Levenshtein


def norm(t):
    return '' if not t else t.translate(str.maketrans({'（':'(','）':')','【':'[','】':']','｛':'{','｝':'}','，':',','：':':','；':';','！':'!','？':'?','。':'.'}))

def rjsonl(p):
    return [json.loads(x) for x in Path(p).read_text(encoding='utf-8').splitlines() if x.strip()]

def wjsonl(p, rows):
    with Path(p).open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

def wcsv(p, rows):
    p = Path(p)
    if not rows:
        p.write_text('', encoding='utf-8')
        return
    fields, seen = [], set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k); fields.append(k)
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

def cer_rows(rows, text_key):
    valid = [r for r in rows if r.get(text_key, '') != '']
    num = sum(Levenshtein.distance(norm(r.get(text_key,'')), norm(r.get('gt',''))) for r in valid)
    den = sum(len(norm(r.get('gt',''))) for r in valid)
    return round(num/den, 6) if den else 0.0

def acc_rows(rows, text_key):
    valid = [r for r in rows if r.get(text_key, '') != '']
    return round(sum(1 for r in valid if r.get(text_key,'') == r.get('gt','')) / len(valid), 6) if valid else 0.0

def merge_by_sample(base_rows, patch_rows):
    out = {r['sample_id']: dict(r) for r in base_rows}
    for r in patch_rows:
        out[r['sample_id']] = dict(r)
    return [out[k] for k in sorted(out.keys())]

def main():
    p = argparse.ArgumentParser(description='merge upper/lower bound runs')
    p.add_argument('--base_run_dir', required=True)
    p.add_argument('--patch_run_dir', required=True)
    p.add_argument('--output_dir', required=True)
    a = p.parse_args()

    base = Path(a.base_run_dir)
    patch = Path(a.patch_run_dir)
    out = Path(a.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    a_only = rjsonl(base / 'A_only_outputs.jsonl')
    a_plus_b = rjsonl(base / 'A_plus_B_correction_outputs.jsonl')
    b_base = rjsonl(base / 'B_only_recognition_outputs.jsonl')
    b_patch = rjsonl(patch / 'B_only_recognition_outputs.jsonl')
    b_only = merge_by_sample(b_base, b_patch)

    failed_rows = [r for r in b_only if r.get('error_type') != 'none']
    error_stats = {}
    for r in b_only:
        err = r.get('error_type', 'none')
        error_stats[err] = error_stats.get(err, 0) + 1

    for r in b_only:
        txt = r.get('final_text', '')
        r['edit_distance_final'] = Levenshtein.distance(norm(txt), norm(r.get('gt',''))) if txt else None

    wjsonl(out / 'A_only_outputs.jsonl', a_only)
    wjsonl(out / 'A_plus_B_correction_outputs.jsonl', a_plus_b)
    wjsonl(out / 'B_only_recognition_outputs.jsonl', b_only)
    wcsv(out / 'b_only_error_stats.csv', [{'error_type': k, 'count': v} for k, v in sorted(error_stats.items())])
    wcsv(out / 'b_only_failed_samples.csv', failed_rows)
    wcsv(out / 'tab_upper_lower_bounds.csv', [
        {'system_name':'A-Only', 'CER': cer_rows(a_only,'final_text'), 'accuracy': acc_rows(a_only,'final_text'), 'n_samples': len(a_only), 'n_eval': len(a_only), 'backend_label':'agent_a'},
        {'system_name':'B-Only Recognition', 'CER': cer_rows(b_only,'final_text'), 'accuracy': acc_rows(b_only,'final_text'), 'n_samples': len(b_only), 'n_eval': sum(1 for r in b_only if r.get('final_text','') != ''), 'backend_label': (b_only[0].get('backend_label','') if b_only else '')},
        {'system_name':'A+B Correction', 'CER': cer_rows(a_plus_b,'final_text'), 'accuracy': acc_rows(a_plus_b,'final_text'), 'n_samples': len(a_plus_b), 'n_eval': len(a_plus_b), 'backend_label':'reuse_maina_gemini_cache'},
    ])

    b_map = {r['sample_id']: r for r in b_only}
    ab_map = {r['sample_id']: r for r in a_plus_b}
    case_pool = []
    for ao in a_only:
        sid = ao['sample_id']; bo = b_map[sid]; ab = ab_map[sid]
        case_pool.append({'sample_id': sid,'domain': ao['domain'],'ocr_text': ao['ocr_text'],'b_only_text': bo['final_text'],'a_plus_b_text': ab['final_text'],'gt': ao['gt'],'edit_distance_a_only': ao['edit_distance_final'],'edit_distance_b_only': bo['edit_distance_final'],'edit_distance_a_plus_b': ab['edit_distance_final'],'a_plus_b_better_than_b_only': (bo['edit_distance_final'] is not None) and ab['edit_distance_final'] < bo['edit_distance_final'],'a_plus_b_better_than_a_only': ab['edit_distance_final'] < ao['edit_distance_final'],'b_only_error_type': bo.get('error_type','none')})
    wcsv(out / 'upper_lower_case_pool.csv', case_pool)
    (out / 'manifest.json').write_text(json.dumps({'base_run_dir': str(base), 'patch_run_dir': str(patch), 'error_stats': error_stats}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(out)

if __name__ == '__main__':
    main()
