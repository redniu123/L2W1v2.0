"""Extended metrics extracted from ``scripts/experiments/efficiency_frontier.py``.

These helpers preserve the experiment script's metric semantics exactly. In
particular, ``normalize_eval_text`` only maps selected full-width punctuation to
half-width punctuation, and is intentionally different from
``l2w1.metrics.text.normalize_text_for_eval``.
"""

from __future__ import annotations

import difflib
from typing import Any

import Levenshtein
import numpy as np


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


def summarize_extended_metrics(per_sample: list[dict[str, Any]], boundary_k: int = 2) -> dict[str, float]:
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


def summarize_latency_and_token_usage(per_sample: list[dict[str, Any]]) -> dict[str, float]:
    latencies = [float(item['latency_ms']) for item in per_sample if item.get('latency_ms') is not None]
    token_usages = [float(item['token_usage']) for item in per_sample if item.get('token_usage') is not None]
    return {
        'P95_Latency_MS': round(float(np.percentile(latencies, 95)), 3) if latencies else 0.0,
        'Avg_Token_Usage': round((sum(token_usages) / len(token_usages)), 3) if token_usages else 0.0,
        'Total_Token_Usage': round(sum(token_usages), 3) if token_usages else 0.0,
        'N_Latency_Valid': len(latencies),
        'N_Token_Valid': len(token_usages),
    }
