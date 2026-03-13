#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v5.1 Phase 2 Task 1: Top-2 Coverage Check

遍历 Validation Set 中所有真实错误样本，
统计 Agent A 的 Top-1 与 Top-2 预测是否覆盖了 Ground Truth 字符。

输出：Coverage@2 百分比
"""

import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import Levenshtein


def check_top2_coverage(
    val_jsonl: str,
    metadata_jsonl: str,
    image_root: str = "data/geo",
    K: int = 2,
) -> None:
    """
    检查 Val 集中错误样本的 Top-2 字符覆盖率

    Args:
        val_jsonl:     val.jsonl 路径
        metadata_jsonl: features_val 对应的 metadata.jsonl（含 T_A）
        image_root:    图像基准目录
        K:             边界窗口大小
    """
    from tools.infer.utility import init_args
    from modules.paddle_engine.predict_rec_modified import TextRecognizerWithLogits

    # 读取 metadata（含 T_A 和 y_deletion）
    meta_map = {}  # image_path -> {T_A, T_GT, y_deletion}
    with open(metadata_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                m = json.loads(line)
                meta_map[m["image"]] = m

    # 读取 val.jsonl
    samples = []
    with open(val_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"Val 集总样本: {len(samples)}")

    # 初始化 Agent A
    import argparse
    parser = argparse.ArgumentParser()
    # 借用 utility 的参数体系
    from tools.infer.utility import init_args as _init
    parser2 = _init()
    parser2.add_argument("--val_jsonl", type=str)
    parser2.add_argument("--metadata_jsonl", type=str)
    parser2.add_argument("--image_root", type=str, default="data/geo")
    parser2.add_argument("--K", type=int, default=2)
    parser2.add_argument("--use_det", action="store_true")
    parser2.add_argument("--det_model_dir", type=str, default="")
    args = parser2.parse_args()

    if args.rec_model_dir is None:
        args.rec_model_dir = "./models/agent_a_ppocr/PP-OCRv5_server_rec_infer"

    print("初始化 Agent A...")
    recognizer = TextRecognizerWithLogits(args)

    import cv2
    data_root = Path(image_root).resolve()

    total_error_chars = 0    # 所有错误样本中需要被覆盖的边界字符总数
    covered_by_top2 = 0      # 被 Top-2 覆盖的边界字符数
    error_samples = 0        # 有真实错误的样本数
    no_top2_samples = 0      # top2_info 不可用的样本数

    for sample in tqdm(samples, desc="Checking Coverage@2"):
        image_path = sample.get("image") or sample.get("image_path")
        T_GT = sample.get("gt_text") or sample.get("text") or sample.get("label", "")

        if not image_path or not T_GT:
            continue

        img_path = Path(image_path)
        if not img_path.is_absolute():
            img_path = data_root / img_path

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        try:
            output = recognizer([img])
            if not output or not output.get("results"):
                continue
            T_A, conf = output["results"][0]
        except Exception:
            continue

        # 判断是否有真实错误（y_deletion 或 y_ambiguity）
        ops = Levenshtein.editops(T_A, T_GT)
        boundary_errors = [
            (op, p_a, p_gt) for op, p_a, p_gt in ops
            if (op in ("delete", "replace")) and
               (p_gt < K or p_gt >= len(T_GT) - K)
        ]
        if not boundary_errors:
            continue

        error_samples += 1

        # 获取 top2_info
        top2_info_list = output.get("top2_info", [])
        top2_info = top2_info_list[0] if top2_info_list else None

        if not top2_info or top2_info.get("top2_status") != "available":
            no_top2_samples += 1
            total_error_chars += len(boundary_errors)
            continue

        top1_chars = top2_info.get("top1_chars", [])
        top2_chars = top2_info.get("top2_chars", [])

        # 对每个边界错误位置，检查 GT 字符是否在 Top-2 内
        for op, p_a, p_gt in boundary_errors:
            total_error_chars += 1
            gt_char = T_GT[p_gt] if p_gt < len(T_GT) else ""

            if op == "replace" and p_a < len(top1_chars):
                t1 = top1_chars[p_a] if p_a < len(top1_chars) else ""
                t2 = top2_chars[p_a] if p_a < len(top2_chars) else ""
                if gt_char in (t1, t2):
                    covered_by_top2 += 1
            elif op == "delete":
                # 漏字：GT 有字但 T_A 没有，无法用 Top-2 覆盖
                pass

    print(f"\n{'='*50}")
    print(f"[Coverage@2 统计结果]")
    print(f"  Val 集总样本数:         {len(samples)}")
    print(f"  有边界错误的样本数:      {error_samples}")
    print(f"  Top-2 不可用的样本数:    {no_top2_samples}")
    print(f"  边界错误字符总数:        {total_error_chars}")
    print(f"  被 Top-2 覆盖的字符数:   {covered_by_top2}")
    if total_error_chars > 0:
        coverage = covered_by_top2 / total_error_chars
        print(f"  Coverage@2:              {coverage:.2%}")
    else:
        print(f"  Coverage@2:              N/A (无边界错误字符)")
    print(f"{'='*50}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SH-DA++ v5.1: Top-2 Coverage Check")
    parser.add_argument("--val_jsonl", type=str, default="data/raw/hctr_riskbench/val.jsonl")
    parser.add_argument("--metadata_jsonl", type=str, default="results/stage2_v51/metadata_val.jsonl")
    parser.add_argument("--image_root", type=str, default="data/geo")
    parser.add_argument("--K", type=int, default=2)
    args = parser.parse_args()

    check_top2_coverage(
        val_jsonl=args.val_jsonl,
        metadata_jsonl=args.metadata_jsonl,
        image_root=args.image_root,
        K=args.K,
    )


if __name__ == "__main__":
    main()
