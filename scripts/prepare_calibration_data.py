#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 (S2.1) - 自动化标签构造与校准集准备

目标：
1) 对齐 agent_a_text 与 gt_text，构造边界漏字与高 CER 标签
2) 从 router_features.jsonl 提取特征向量，输出 train_samples.csv
3) 打印类别分布与边界漏字统计
"""

import argparse
import csv
import json
import math
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    # DP, O(len(a)*len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(
                min(
                    prev[j] + 1,      # delete
                    curr[j - 1] + 1,  # insert
                    prev[j - 1] + cost,  # substitute
                )
            )
        prev = curr
    return prev[-1]


def compute_cer(pred: str, gt: str) -> float:
    if gt is None:
        return 1.0
    gt = gt or ""
    if len(gt) == 0:
        return 0.0 if (pred or "") == "" else 1.0
    ed = levenshtein_distance(pred or "", gt)
    return ed / max(1, len(gt))


def has_boundary_deletion(
    pred: str,
    gt: str,
    boundary_k: int,
) -> bool:
    """
    Boundary Deletion 定义：
    GT 文本的前 K 或后 K 个字符，在 pred 中缺失。
    """
    if not gt:
        return False
    if boundary_k <= 0:
        return False

    k = min(boundary_k, len(gt))
    left_range = (0, k)
    right_range = (len(gt) - k, len(gt))

    matcher = SequenceMatcher(None, gt, pred)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "delete":
            continue
        # deletion covers gt[i1:i2]
        if i2 <= left_range[0] or i1 >= left_range[1]:
            left_overlap = False
        else:
            left_overlap = True
        if i2 <= right_range[0] or i1 >= right_range[1]:
            right_overlap = False
        else:
            right_overlap = True
        if left_overlap or right_overlap:
            return True
    return False


def compute_b_edge(record: Dict) -> float:
    blank_mean_L = float(record.get("blank_mean_L", 0.0) or 0.0)
    blank_mean_R = float(record.get("blank_mean_R", 0.0) or 0.0)
    blank_peak_L = float(record.get("blank_peak_L", 0.0) or 0.0)
    blank_peak_R = float(record.get("blank_peak_R", 0.0) or 0.0)
    b_edge_L = 0.6 * blank_mean_L + 0.4 * blank_peak_L
    b_edge_R = 0.6 * blank_mean_R + 0.4 * blank_peak_R
    return max(b_edge_L, b_edge_R)


def compute_drop(pred: str, gt: str) -> float:
    if not gt:
        return 0.0
    expected = len(gt)
    actual = len(pred or "")
    if expected <= 0:
        return 0.0
    return max(0.0, (expected - actual) / expected)


def build_feature_row(record: Dict, cer: float, b_edge: float, drop: float) -> Dict:
    top2_status = record.get("top2_status", "missing") or "missing"
    top2_status_code = {"available": 2, "available_no_chars": 1, "missing": 0}.get(
        top2_status, 0
    )

    return {
        "v_edge_raw": record.get("v_edge_raw", 0.0) or 0.0,
        "v_edge_norm": record.get("v_edge_norm", 0.0) or 0.0,
        "blank_mean_L": record.get("blank_mean_L", 0.0) or 0.0,
        "blank_mean_R": record.get("blank_mean_R", 0.0) or 0.0,
        "blank_peak_L": record.get("blank_peak_L", 0.0) or 0.0,
        "blank_peak_R": record.get("blank_peak_R", 0.0) or 0.0,
        "b_edge": b_edge,
        "v_edge_x_b_edge": (record.get("v_edge_raw", 0.0) or 0.0) * b_edge,
        "drop": drop,
        "s_b": record.get("s_b", 0.0) or 0.0,
        "s_a": record.get("s_a", 0.0) or 0.0,
        "q": record.get("q", 0.0) or 0.0,
        "agent_a_confidence": record.get("agent_a_confidence", 0.0) or 0.0,
        "top1_conf_mean": record.get("top1_conf_mean", 0.0) or 0.0,
        "top2_conf_mean": record.get("top2_conf_mean", 0.0) or 0.0,
        "conf_gap_mean": record.get("conf_gap_mean", 0.0) or 0.0,
        "top2_status_code": top2_status_code,
        "cer": cer,
        "gt_len": len(record.get("gt_text", "") or ""),
        "pred_len": len(record.get("agent_a_text", "") or ""),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare calibration dataset for Stage 2 (LogReg)."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/router_features.jsonl",
        help="Path to router_features.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/calibration/train_samples.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--boundary_k",
        type=int,
        default=2,
        help="Boundary window K (front/back). Default: 2",
    )
    parser.add_argument(
        "--cer_threshold",
        type=float,
        default=0.1,
        help="CER threshold for hard samples",
    )
    parser.add_argument(
        "--conf_threshold",
        type=float,
        default=0.8,
        help="Confidence threshold (low confidence => positive)",
    )
    parser.add_argument(
        "--keep_nonperfect",
        action="store_true",
        help="Keep samples with 0 < CER <= threshold as label=0",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pos_count = 0
    neg_count = 0
    boundary_pos_count = 0
    total_records = 0
    skipped = 0

    rows: List[Dict] = []
    for record in iter_jsonl(input_path):
        total_records += 1
        gt_text = record.get("gt_text", "")
        pred_text = record.get("agent_a_text", "")
        if gt_text is None or gt_text == "":
            skipped += 1
            continue

        cer = compute_cer(pred_text, gt_text)
        boundary_del = has_boundary_deletion(
            pred=pred_text,
            gt=gt_text,
            boundary_k=args.boundary_k,
        )
        if boundary_del:
            boundary_pos_count += 1

        # 标签规则
        label = None
        agent_a_conf = float(record.get("agent_a_confidence", 0.0) or 0.0)
        if boundary_del:
            label = 1
        elif cer > args.cer_threshold and agent_a_conf < args.conf_threshold:
            label = 1
        elif cer == 0.0:
            label = 0
        elif args.keep_nonperfect:
            label = 0
        else:
            skipped += 1
            continue

        b_edge = compute_b_edge(record)
        drop = compute_drop(pred_text, gt_text)
        feature_row = build_feature_row(record, cer, b_edge, drop)
        feature_row["label"] = label
        rows.append(feature_row)

        if label == 1:
            pos_count += 1
        else:
            neg_count += 1

    if not rows:
        print("[Error] No valid samples produced. Check input/gt_text.")
        return

    # 输出 CSV
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    total_labeled = pos_count + neg_count
    pos_ratio = pos_count / total_labeled if total_labeled else 0.0
    neg_ratio = neg_count / total_labeled if total_labeled else 0.0

    print("=" * 70)
    print("Stage 2 - Calibration Data Preparation")
    print("=" * 70)
    print(f"Input records: {total_records}")
    print(f"Labeled samples: {total_labeled}")
    print(f"Positive (label=1): {pos_count} ({pos_ratio:.2%})")
    print(f"Negative (label=0): {neg_count} ({neg_ratio:.2%})")
    print(f"Boundary deletion positives: {boundary_pos_count}")
    print(f"Skipped records: {skipped}")
    print(f"Output CSV: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
