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

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


def extract_boundary_features(
    logits: np.ndarray,
    char_conf: List[float],
    rho: float = 0.1,
    K_drop: int = 2,
) -> Tuple[float, float, float]:
    """
    从 logits 中提取 [v_edge, b_edge, drop] 特征

    Args:
        logits: Emission 矩阵 (T, C)，已经是概率
        char_conf: 字符级置信度序列
        rho: 边界窗口比例
        K_drop: 陡降检测窗口

    Returns:
        v_edge, b_edge, drop
    """
    T, C = logits.shape
    blank_id = 0

    # --- b_edge: 边界 blank 均值 ---
    L_end = max(1, int(rho * T))
    R_start = min(T - 1, int((1 - rho) * T))

    blank_probs = logits[:, blank_id]  # (T,)
    blank_mean_L = float(blank_probs[:L_end].mean())
    blank_mean_R = float(blank_probs[R_start:].mean())
    b_edge = max(blank_mean_L, blank_mean_R)

    # --- v_edge: 边界区域视觉熵均值 ---
    eps = 1e-10
    entropy = -np.sum(logits * np.log(logits + eps), axis=-1)  # (T,)
    v_edge_raw = (entropy[:L_end].mean() + entropy[R_start:].mean()) / 2.0
    # Min-max 归一化到 [0, 1]（以 [0, 5] 为典型范围）
    v_edge = float(np.clip(v_edge_raw / 5.0, 0.0, 1.0))

    # --- drop: 边界字符置信度陡降 ---
    N = len(char_conf)
    if N > 2 * K_drop:
        p_left = np.mean(char_conf[:K_drop])
        p_right = np.mean(char_conf[N - K_drop:])
        p_mid = np.mean(char_conf[K_drop:N - K_drop])
        drop = float(max(0.0, p_mid - min(p_left, p_right)))
    else:
        drop = 0.0

    return v_edge, b_edge, drop


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
    from PIL import Image
    import cv2

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 Agent A
    print("[1/4] 初始化 Agent A (TextRecognizerWithLogits)...")
    recognizer = TextRecognizerWithLogits(recognizer_args)

    # 读取验证集
    print(f"[2/4] 读取验证集: {data_jsonl}")
    samples = []
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
        try:
            output = recognizer([img])
            if not output or "results" not in output:
                skip_count += 1
                continue

            results = output["results"]
            logits_list = output.get("logits", [])

            if not results or not logits_list:
                skip_count += 1
                continue

            T_A, conf = results[0]
            logits = logits_list[0]  # (T, C)

            if logits is None or len(logits.shape) != 2:
                skip_count += 1
                continue

        except Exception as e:
            skip_count += 1
            continue

        # 提取特征
        char_conf = [conf]  # 简化版：只有整体置信度
        v_edge, b_edge, drop = extract_boundary_features(logits, char_conf)

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
    parser = argparse.ArgumentParser(description="SH-DA++ Stage 2: 准备校准数据集")
    parser.add_argument("--data_jsonl", type=str, required=True, help="验证集 JSONL 路径")
    parser.add_argument("--output_dir", type=str, default="./results/stage2", help="输出目录")
    parser.add_argument("--rec_model_dir", type=str, default="./models/ppocrv5_rec", help="Agent A 模型目录")
    parser.add_argument("--rec_char_dict_path", type=str, default="./ppocr/utils/ppocrv5_dict.txt", help="字符字典路径")
    parser.add_argument("--K", type=int, default=2, help="边界窗口大小")
    parser.add_argument("--rec_image_shape", type=str, default="3, 48, 320", help="识别图像尺寸")
    parser.add_argument("--rec_batch_num", type=int, default=6, help="批处理大小")
    parser.add_argument("--rec_algorithm", type=str, default="SVTR_LCNet", help="识别算法")
    parser.add_argument("--use_space_char", type=bool, default=True, help="是否使用空格")
    parser.add_argument("--use_gpu", type=bool, default=True, help="是否使用 GPU")
    parser.add_argument("--use_det", action="store_true", help="是否启用检测器")
    parser.add_argument("--det_model_dir", type=str, default="", help="检测模型目录")

    args = parser.parse_args()

    prepare_calibration_dataset(
        data_jsonl=args.data_jsonl,
        output_dir=args.output_dir,
        recognizer_args=args,
        K=args.K,
    )


if __name__ == "__main__":
    main()
