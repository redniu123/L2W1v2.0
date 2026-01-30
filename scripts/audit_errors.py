#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ 错误审计报表生成脚本

功能：
1) 从 router_features.jsonl 读取样本
2) 重新对齐 agent_a_text 与 gt_text，识别边界漏字
3) 控制台打印对齐结果，生成 Markdown 审计报表
"""

import argparse
import json
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


def classify_boundary_deletion(
    pred: str, gt: str, boundary_k: int
) -> Tuple[bool, str]:
    """
    返回 (is_boundary_deletion, error_type)
    error_type: "Left Deletion" | "Right Deletion" | "Both Deletion"
    """
    if not gt or boundary_k <= 0:
        return False, ""

    k = min(boundary_k, len(gt))
    left_range = (0, k)
    right_range = (len(gt) - k, len(gt))

    matcher = SequenceMatcher(None, gt, pred)
    left_hit = False
    right_hit = False
    for tag, i1, i2, _, _ in matcher.get_opcodes():
        if tag != "delete":
            continue
        if i2 > left_range[0] and i1 < left_range[1]:
            left_hit = True
        if i2 > right_range[0] and i1 < right_range[1]:
            right_hit = True
    if left_hit and right_hit:
        return True, "Both Deletion"
    if left_hit:
        return True, "Left Deletion"
    if right_hit:
        return True, "Right Deletion"
    return False, ""


def build_alignment_view(pred: str, gt: str) -> Tuple[str, str]:
    """
    输出两行对齐视图：
    - GT: 边界漏字字符用 [ ] 标记
    - A : 缺失位置用 [] 表示
    """
    matcher = SequenceMatcher(None, gt, pred)
    gt_out = []
    pred_out = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            gt_out.append(gt[i1:i2])
            pred_out.append(pred[j1:j2])
        elif tag == "delete":
            deleted = gt[i1:i2]
            gt_out.append(f"[{deleted}]")
            pred_out.append("[]")
        elif tag == "insert":
            inserted = pred[j1:j2]
            gt_out.append("[]")
            pred_out.append(f"[{inserted}]")
        else:  # replace
            replaced_gt = gt[i1:i2]
            replaced_pred = pred[j1:j2]
            gt_out.append(f"{{{replaced_gt}}}")
            pred_out.append(f"{{{replaced_pred}}}")

    return "".join(gt_out), "".join(pred_out)


def main():
    parser = argparse.ArgumentParser(description="Generate boundary deletion audit report.")
    parser.add_argument(
        "--input",
        type=str,
        default="results/router_features.jsonl",
        help="Input router_features.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/error_audit_list.md",
        help="Output markdown report path",
    )
    parser.add_argument(
        "--boundary_k",
        type=int,
        default=2,
        help="Boundary window K for deletion detection (default: 2)",
    )
    parser.add_argument(
        "--max_print",
        type=int,
        default=0,
        help="Max number of samples to print to console (0 = all)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    boundary_count = 0
    left_count = 0
    right_count = 0
    both_count = 0
    printed = 0

    table_rows: List[Dict] = []

    for record in iter_jsonl(input_path):
        total += 1
        gt_text = record.get("gt_text", "") or ""
        pred_text = record.get("agent_a_text", "") or ""
        if not gt_text:
            continue

        is_boundary, error_type = classify_boundary_deletion(
            pred=pred_text, gt=gt_text, boundary_k=args.boundary_k
        )
        if not is_boundary:
            continue

        boundary_count += 1
        if error_type == "Left Deletion":
            left_count += 1
        elif error_type == "Right Deletion":
            right_count += 1
        else:
            both_count += 1

        s_b = record.get("s_b", 0.0) or 0.0
        sample_id = record.get("id", f"sample_{total:06d}")

        # 控制台对齐输出
        if args.max_print == 0 or printed < args.max_print:
            gt_view, pred_view = build_alignment_view(pred_text, gt_text)
            print(f"[{sample_id}] Error: {error_type}")
            print(f"GT: {gt_view}")
            print(f"A : {pred_view}")
            print("-" * 60)
            printed += 1

        table_rows.append(
            {
                "id": sample_id,
                "error_type": error_type,
                "agent_a_text": pred_text,
                "gt_text": gt_text,
                "s_b": f"{float(s_b):.4f}",
            }
        )

    # 写 Markdown 报表
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# SH-DA++ 错误审计报表\n\n")
        f.write(f"- 输入样本数: {total}\n")
        f.write(f"- 边界漏字样本: {boundary_count}\n")
        f.write(f"- 左边界漏字: {left_count}\n")
        f.write(f"- 右边界漏字: {right_count}\n")
        f.write(f"- 双侧漏字: {both_count}\n\n")
        f.write("| ID | Error Type | Agent A Text | GT Text | Score (s_b) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for row in table_rows:
            f.write(
                f"| {row['id']} | {row['error_type']} | {row['agent_a_text']} | "
                f"{row['gt_text']} | {row['s_b']} |\n"
            )

    print("=" * 70)
    print("SH-DA++ Error Audit Summary")
    print("=" * 70)
    print(f"Input records: {total}")
    print(f"Boundary deletions: {boundary_count}")
    print(f"Left deletions: {left_count}")
    print(f"Right deletions: {right_count}")
    print(f"Both sides: {both_count}")
    print(f"Markdown report: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
