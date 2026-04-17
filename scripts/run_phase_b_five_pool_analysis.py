#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_BUDGETS = [0.05, 0.10, 0.20, 0.60, 1.00]
TERM_MIN_LEN = 2
NUMBERING_CHARS = set('0123456789①②③④⑤⑥⑦⑧⑨⑩ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ一二三四五六七八九十ABCDEF')
STRUCTURAL_SYMBOLS = set('()（）[]【】{}<>《》→->—-=:：;；,.，。!?！？/\\|·•○●▲△※☆★')
BOUNDARY_WINDOW = 2


def jdump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def wcsv(path, fieldnames, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, '') for k in fieldnames})


def budget_to_tag(budget):
    return f'{int(round(budget * 100)):02d}'


def normalize_format_text(text):
    table = str.maketrans({
        '（': '(', '）': ')', '【': '[', '】': ']', '｛': '{', '｝': '}',
        '，': ',', '：': ':', '；': ';', '！': '!', '？': '?', '。': '.',
        '—': '-', '–': '-', '－': '-', '→': '->', '　': ' ',
    })
    return text.translate(table).replace(' ', '').strip()


def char_set(text, allowed):
    return {ch for ch in text if ch in allowed}


def common_terms(terms, texts):
    found = []
    for term in terms or []:
        if len(term) >= TERM_MIN_LEN and any(term in t for t in texts):
            found.append(term)
    return found


def first_last_diff_positions(a, b):
    left = 0
    min_len = min(len(a), len(b))
    while left < min_len and a[left] == b[left]:
        left += 1
    if left == len(a) == len(b):
        return None, None
    right_a = len(a) - 1
    right_b = len(b) - 1
    while right_a >= left and right_b >= left and a[right_a] == b[right_b]:
        right_a -= 1
        right_b -= 1
    return left, max(right_a, right_b)


def classify_edit_span_type(ocr_text, vlm_text):
    if ocr_text == vlm_text:
        return 'no_change'
    left, right = first_last_diff_positions(ocr_text, vlm_text)
    if left is None:
        return 'no_change'
    changed_chars = max(1, right - left + 1)
    if len(ocr_text) != len(vlm_text):
        return 'length_changed'
    if changed_chars == 1:
        return 'single_char'
    if changed_chars <= 3:
        return 'local_multi_char'
    return 'long_span'


def is_boundary_edit(ocr_text, vlm_text):
    if ocr_text == vlm_text:
        return False
    left, right = first_last_diff_positions(ocr_text, vlm_text)
    if left is None:
        return False
    max_len = max(len(ocr_text), len(vlm_text))
    return left < BOUNDARY_WINDOW or right >= max_len - BOUNDARY_WINDOW


def is_format_normalization(ocr_text, vlm_text):
    return ocr_text != vlm_text and normalize_format_text(ocr_text) == normalize_format_text(vlm_text)


def is_semantic_substitution(ocr_text, vlm_text, gt_text):
    if ocr_text == vlm_text:
        return False
    if len(ocr_text) != len(vlm_text):
        return False
    if char_set(ocr_text + vlm_text, NUMBERING_CHARS) or char_set(ocr_text + vlm_text, STRUCTURAL_SYMBOLS):
        return False
    return vlm_text != gt_text and classify_edit_span_type(ocr_text, vlm_text) in {'local_multi_char', 'long_span'}


def derive_risk_tag(row):
    tags = []
    if row['is_boundary_edit']:
        tags.append('boundary_edit')
    if row['is_numbering_changed']:
        tags.append('numbering_change')
    if row['is_symbol_changed']:
        tags.append('symbol_change')
    if row['is_term_changed']:
        tags.append('term_change')
    if row['is_format_normalization']:
        tags.append('format_norm')
    if row['is_semantic_substitution']:
        tags.append('semantic_sub')
    if row['is_length_changed']:
        tags.append('length_change')
    if not tags:
        tags.append('plain_local_edit')
    return '|'.join(tags)


def derive_edit_type(row):
    if row['is_numbering_changed']:
        return 'numbering_rewrite'
    if row['is_format_normalization']:
        return 'format_normalization'
    if row['is_term_changed']:
        return 'term_edit'
    if row['is_boundary_edit'] and row['edit_span_type'] == 'single_char':
        return 'boundary_single_char_fix'
    if row['is_boundary_edit']:
        return 'boundary_edit'
    if row['is_symbol_changed']:
        return 'symbol_edit'
    if row['is_semantic_substitution']:
        return 'semantic_substitution'
    if row['edit_span_type'] == 'single_char':
        return 'single_char_fix'
    if row['is_length_changed']:
        return 'length_change'
    return row['edit_span_type']


def gain_type(edit_distance_ocr, edit_distance_final):
    if edit_distance_final < edit_distance_ocr:
        return 'beneficial'
    if edit_distance_final == edit_distance_ocr:
        return 'neutral'
    return 'harmful'


def enrich_row(item, budget):
    ocr_text = item.get('ocr_text', '')
    vlm_text = item.get('final_text_if_upgraded') or item.get('vlm_raw_output') or item.get('final_text', '')
    gt_text = item.get('gt', '')
    terms = item.get('professional_terms', []) or []
    ocr_ed = int(item.get('edit_distance_ocr', 0))
    vlm_ed = int(item.get('edit_distance_final', 0))
    numbering_changed = char_set(ocr_text, NUMBERING_CHARS) != char_set(vlm_text, NUMBERING_CHARS)
    symbol_changed = char_set(ocr_text, STRUCTURAL_SYMBOLS) != char_set(vlm_text, STRUCTURAL_SYMBOLS)
    term_changed = set(common_terms(terms, [ocr_text, gt_text])) != set(common_terms(terms, [vlm_text, gt_text]))
    row = {
        'run_id': item.get('run_id', ''),
        'analysis_split': item.get('split', 'test'),
        'analysis_scope': 'exploratory_test_based',
        'rule_freeze_allowed': False,
        'budget': budget,
        'budget_tag': budget_to_tag(budget),
        'sample_id': item.get('sample_id', ''),
        'domain': item.get('domain', ''),
        'ocr_text': ocr_text,
        'vlm_raw_output': item.get('vlm_raw_output', ''),
        'vlm_text': vlm_text,
        'gt': gt_text,
        'router_score': item.get('router_score', 0.0),
        'edit_distance_ocr_to_gt': ocr_ed,
        'edit_distance_vlm_to_gt': vlm_ed,
        'edit_distance_delta': vlm_ed - ocr_ed,
        'is_length_changed': len(ocr_text) != len(vlm_text),
        'is_numbering_changed': numbering_changed,
        'is_symbol_changed': symbol_changed,
        'is_term_changed': term_changed,
        'is_boundary_edit': is_boundary_edit(ocr_text, vlm_text),
        'is_format_normalization': is_format_normalization(ocr_text, vlm_text),
        'is_semantic_substitution': is_semantic_substitution(ocr_text, vlm_text, gt_text),
        'edit_span_type': classify_edit_span_type(ocr_text, vlm_text),
        'has_professional_terms': item.get('has_professional_terms', False),
        'professional_terms': '|'.join(terms),
        'backfill_status': item.get('backfill_status', ''),
        'backfill_reason': item.get('backfill_reason', ''),
    }
    row['edit_gain_type'] = gain_type(ocr_ed, vlm_ed)
    row['risk_tag'] = derive_risk_tag(row)
    row['edit_type'] = derive_edit_type(row)
    return row


def summarize_type_stats(rows):
    stats = defaultdict(lambda: Counter())
    for row in rows:
        key = (row['budget_tag'], row['edit_type'])
        stats[key]['count'] += 1
        stats[key][row['edit_gain_type']] += 1
        stats[key][row['risk_tag']] += 0
    out = []
    for (budget_tag, edit_type), counter in sorted(stats.items()):
        count = counter['count']
        out.append({
            'budget_tag': budget_tag,
            'edit_type': edit_type,
            'count': count,
            'beneficial_count': counter['beneficial'],
            'neutral_count': counter['neutral'],
            'harmful_count': counter['harmful'],
            'beneficial_rate': round(counter['beneficial'] / count, 6) if count else 0.0,
            'harmful_rate': round(counter['harmful'] / count, 6) if count else 0.0,
        })
    return out


def summarize_rule_table(rows):
    feature_keys = [
        'is_boundary_edit', 'is_numbering_changed', 'is_symbol_changed',
        'is_term_changed', 'is_format_normalization', 'is_semantic_substitution',
        'is_length_changed',
    ]
    out = []
    for key in feature_keys:
        subset = [r for r in rows if r[key]]
        if not subset:
            continue
        beneficial = sum(1 for r in subset if r['edit_gain_type'] == 'beneficial')
        neutral = sum(1 for r in subset if r['edit_gain_type'] == 'neutral')
        harmful = sum(1 for r in subset if r['edit_gain_type'] == 'harmful')
        if harmful == 0 and beneficial > 0:
            action = 'prefer_accept'
        elif harmful > beneficial:
            action = 'hard_reject_or_strict_review'
        else:
            action = 'strict_review'
        out.append({
            'feature_name': key,
            'count': len(subset),
            'beneficial_count': beneficial,
            'neutral_count': neutral,
            'harmful_count': harmful,
            'beneficial_rate': round(beneficial / len(subset), 6),
            'harmful_rate': round(harmful / len(subset), 6),
            'suggested_action': action,
            'evidence_scope': 'exploratory_test_based_only',
        })
    return out


def export_pool(run_dir, budget, output_dir):
    budget_tag = budget_to_tag(budget)
    src = run_dir / f'M5_offline_budget_{budget_tag}_results.jsonl'
    items = read_jsonl(src)
    upgraded = [item for item in items if item.get('selected_for_upgrade')]
    rows = [enrich_row(item, budget) for item in upgraded]
    base = output_dir / f'm5_budget_{budget_tag}'
    all_path = base.with_name(base.name + '_edit_behavior_all.csv')
    beneficial_path = base.with_name(base.name + '_edit_behavior_beneficial.csv')
    neutral_path = base.with_name(base.name + '_edit_behavior_neutral.csv')
    harmful_path = base.with_name(base.name + '_edit_behavior_harmful.csv')
    casebook_path = base.with_name(base.name + '_case_pool_for_backfill.csv')
    fieldnames = list(rows[0].keys()) if rows else ['budget_tag']
    wcsv(all_path, fieldnames, rows)
    wcsv(beneficial_path, fieldnames, [r for r in rows if r['edit_gain_type'] == 'beneficial'])
    wcsv(neutral_path, fieldnames, [r for r in rows if r['edit_gain_type'] == 'neutral'])
    wcsv(harmful_path, fieldnames, [r for r in rows if r['edit_gain_type'] == 'harmful'])
    wcsv(casebook_path, fieldnames, rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description='Phase B five-pool M5 edit behavior export')
    parser.add_argument('--phase_a_run_dir', required=True)
    parser.add_argument('--output_subdir', default='phase_b_five_pool')
    parser.add_argument('--budgets', default='0.05,0.10,0.20,0.60,1.00')
    args = parser.parse_args()

    run_dir = Path(args.phase_a_run_dir)
    output_dir = run_dir / args.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    budgets = [float(x) for x in args.budgets.split(',') if x.strip()]

    all_rows = []
    for budget in budgets:
        all_rows.extend(export_pool(run_dir, budget, output_dir))

    if not all_rows:
        raise ValueError('No upgraded samples found in selected pools')

    fieldnames = list(all_rows[0].keys())
    wcsv(output_dir / 'm5_edit_behavior_all.csv', fieldnames, all_rows)
    wcsv(output_dir / 'm5_edit_behavior_beneficial.csv', fieldnames, [r for r in all_rows if r['edit_gain_type'] == 'beneficial'])
    wcsv(output_dir / 'm5_edit_behavior_neutral.csv', fieldnames, [r for r in all_rows if r['edit_gain_type'] == 'neutral'])
    wcsv(output_dir / 'm5_edit_behavior_harmful.csv', fieldnames, [r for r in all_rows if r['edit_gain_type'] == 'harmful'])

    type_stats = summarize_type_stats(all_rows)
    rule_table = summarize_rule_table(all_rows)
    wcsv(output_dir / 'm5_edit_type_stats.csv', list(type_stats[0].keys()) if type_stats else ['budget_tag'], type_stats)
    wcsv(output_dir / 'backfill_v2_rule_table.csv', list(rule_table[0].keys()) if rule_table else ['feature_name'], rule_table)
    jdump(output_dir / 'phase_b_scope.json', {
        'analysis_scope': 'exploratory_test_based',
        'rule_freeze_allowed': False,
        'note': 'Use for hypothesis generation only; freeze backfill_v2 on val or dedicated analysis split.',
        'budgets': budgets,
    })
    print(output_dir)


if __name__ == '__main__':
    main()
