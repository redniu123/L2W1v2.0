#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ Stage 2: Calibration Data Preparation

目标：
1. 从 Validation Set 提取 Router 特征 (v_edge, b_edge, drop)
2. 自动构造边界漏字标签 y_deletion
3. 输出校准数据集 (features.npy, labels.npy)

公式：
y_deletion = I[∃ op ∈ Ops s.t. op.type='delete' ∧ (op.pos ≤ K ∨ op.pos ≥ N-K)]
"""

import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import Levenshtein
from modules.paddle_engine.predict_rec_modified import TextRecognizerWithLogits


def generate_deletion_label(T_A: str, T_GT: str, K: int = 2) -> int:
    """
    生成边界漏字标签

    Args:
        T_A: Agent A 输出文本
        T_GT: Ground Truth 文本
        K: 边界窗口大小（默认 2）

    Returns:
        y_deletion: 0 或 1
    """
    if not T_A or not T_GT:
        return 0

    ops = Levenshtein.editops(T_A, T_GT)

    for op_type, pos_A, pos_GT in ops:
        if op_type == "delete":
            if pos_GT < K or pos_GT >= len(T_GT) - K:
                return 1

    return 0


def prepare_calibration_dataset(
    data_jsonl: str,
    output_dir: str,
    recognizer_args,
    K: int = 2,
) -> None:
    """
    准备校准数据集

    Args:
        data_jsonl: 验证集 JSONL 文件路径
        output_dir: 输出目录
        recognizer_args: TextRecognizerWithLogits 参数对象
        K: 边界窗口大小
    """
    import cv2

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 Agent A
    print("[1/4] 初始化 Agent A (TextRecognizerWithLogits)...")
    recognizer = TextRecognizerWithLogits(recognizer_args)

    # 读取验证集
    print(f"[2/4] 读取验证集: {data_jsonl}")
    samples = []
    # 以 data_jsonl 所在目录作为图像路径的基准目录
    data_root = Path(data_jsonl).resolve().parent
    with open(data_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"总样本数: {len(samples)}")

    # 提取特征和标签
    print("[3/4] 提取特征和生成标签...")
    features_list = []
    labels_list = []
    metadata_list = []
    skip_count = 0

    for sample in tqdm(samples, desc="Processing"):
        # 兼容多种字段名
        image_path = sample.get("image") or sample.get("image_path")
        T_GT = sample.get("gt_text") or sample.get("text") or sample.get("label", "")

        if not image_path or not T_GT:
            skip_count += 1
            continue

        # 将相对路径解析为绝对路径（相对于 data_jsonl 所在目录）
        image_path = Path(image_path)
        if not image_path.is_absolute():
            image_path = data_root / image_path

        # 读取图像
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                skip_count += 1
                continue
        except Exception:
            skip_count += 1
            continue

        # Agent A 推理
        # 返回格式: {"results": [(text, conf), ...], "boundary_stats": [...], "top2_info": [...]}
        try:
            output = recognizer([img])
            if not output or "results" not in output:
                skip_count += 1
                continue

            results = output["results"]
            if not results:
                skip_count += 1
                continue

            T_A, conf = results[0]

            # 从 boundary_stats 提取边界特征（SH-DA++ v4.0 真实接口）
            boundary_stats_list = output.get("boundary_stats", [])
            boundary_stats = boundary_stats_list[0] if boundary_stats_list else None

        except Exception:
            skip_count += 1
            continue

        # 从 boundary_stats 计算 v_edge, b_edge, drop
        if boundary_stats and boundary_stats.get("valid", False):
            blank_mean_L = float(boundary_stats.get("blank_mean_L", 0.0))
            blank_mean_R = float(boundary_stats.get("blank_mean_R", 0.0))
            blank_peak_L = float(boundary_stats.get("blank_peak_L", 0.0))
            blank_peak_R = float(boundary_stats.get("blank_peak_R", 0.0))
            b_edge_L = 0.6 * blank_mean_L + 0.4 * blank_peak_L
            b_edge_R = 0.6 * blank_mean_R + 0.4 * blank_peak_R
            b_edge = float(max(b_edge_L, b_edge_R))
            # v_edge: 用边界 blank 峰值反推视觉熵代理（高 blank 概率 → 低熵 → 取反）
            v_edge = float(np.clip(max(blank_peak_L, blank_peak_R), 0.0, 1.0))
            # drop: 用左右边界 blank 均值的不对称性作为代理
            drop = float(np.clip(abs(blank_mean_L - blank_mean_R), 0.0, 1.0))
        else:
            # boundary_stats 无效时用置信度反推
            b_edge = float(np.clip(1.0 - conf, 0.0, 1.0))
            v_edge = float(np.clip(1.0 - conf, 0.0, 1.0))
            drop = 0.0

        features = np.array(
            [v_edge, b_edge, v_edge * b_edge, drop], dtype=np.float32
        )

        # 生成标签
        y_deletion = generate_deletion_label(T_A, T_GT, K=K)

        features_list.append(features)
        labels_list.append(y_deletion)
        metadata_list.append(
            {
                "image": str(image_path),
                "T_A": T_A,
                "T_GT": T_GT,
                "y_deletion": y_deletion,
                "conf": float(conf),
                "v_edge": float(v_edge),
                "b_edge": float(b_edge),
                "drop": float(drop),
            }
        )

    if not features_list:
        print(f"[错误] 没有成功处理的样本！跳过数: {skip_count}")
        return

    # 转换为 numpy 数组
    X = np.array(features_list, dtype=np.float32)
    Y = np.array(labels_list, dtype=np.int32)

    print(f"[4/4] 保存数据集...")
    print(f"  有效样本: {len(features_list)} / {len(samples)} (跳过: {skip_count})")
    print(f"  特征矩阵 X: {X.shape}")
    print(f"  标签向量 Y: {Y.shape}")
    print(f"  正样本比例: {Y.mean():.2%}")

    np.save(output_dir / "features.npy", X)
    np.save(output_dir / "labels.npy", Y)

    with open(output_dir / "metadata.jsonl", "w", encoding="utf-8") as f:
        for meta in metadata_list:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")

    print(f"✓ 数据集已保存到: {output_dir}")
    print(f"  - features.npy: {X.shape}")
    print(f"  - labels.npy: {Y.shape}")
    print(f"  - metadata.jsonl: {len(metadata_list)} 行")


def main():
    # 使用 utility.init_args() 获取完整的参数列表（包含所有 Paddle 推理参数）
    from tools.infer.utility import init_args

    parser = init_args()

    # 追加 Stage 2 专属参数
    parser.add_argument("--data_jsonl", type=str, required=True, help="验证集 JSONL 路径")
    parser.add_argument("--output_dir", type=str, default="./results/stage2", help="输出目录")
    parser.add_argument("--K", type=int, default=2, help="边界窗口大小")
    parser.add_argument("--use_det", action="store_true", help="是否启用检测器")
    parser.add_argument("--det_model_dir", type=str, default="", help="检测模型目录")

    args = parser.parse_args()

    # 如果用户没有显式指定 rec_model_dir，使用项目实际模型路径
    if args.rec_model_dir is None:
        args.rec_model_dir = "./models/agent_a_ppocr/PP-OCRv5_server_rec_infer"

    prepare_calibration_dataset(
        data_jsonl=args.data_jsonl,
        output_dir=args.output_dir,
        recognizer_args=args,
        K=args.K,
    )


if __name__ == "__main__":
    main()
