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
from typing import Dict, List, Tuple

import Levenshtein
import numpy as np
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.paddle_engine.predict_rec_modified import TextRecognizer
from modules.router.uncertainty_router import UncertaintyRouter, RouterConfig


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

    # 计算编辑操作序列
    ops = Levenshtein.editops(T_A, T_GT)
    N = len(T_A)

    # 扫描 deletion 操作
    for op_type, pos_A, pos_GT in ops:
        if op_type == "delete":
            # 判断是否在边界区域（以 GT 索引为准）
            if pos_GT < K or pos_GT >= len(T_GT) - K:
                return 1

    return 0


def extract_router_features(
    image_path: str, recognizer: TextRecognizer, router: UncertaintyRouter
) -> Tuple[np.ndarray, Dict]:
    """
    提取 Router 特征向量

    Args:
        image_path: 图像路径
        recognizer: Agent A 识别器
        router: 不确定性路由器

    Returns:
        features: [v_edge, b_edge, v_edge*b_edge, drop] (shape: 4,)
        metadata: 包含 T_A, char_conf 等信息
    """
    # Agent A 推理
    result = recognizer.predict_single(image_path)
    if not result:
        return None, None

    T_A = result["text"]
    char_conf = result.get("char_conf", [])
    logits = result.get("logits", None)

    if logits is None:
        return None, None

    # 计算 Router 特征
    routing_result = router.route(logits, char_conf, T_A)

    # 提取特征（需要从 router 内部获取）
    # 这里需要修改 UncertaintyRouter 暴露内部特征
    # 暂时返回占位符
    features = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    metadata = {
        "text": T_A,
        "char_conf": char_conf,
        "visual_entropy": routing_result.visual_entropy,
        "max_char_entropy": routing_result.max_char_entropy,
    }

    return features, metadata


def prepare_calibration_dataset(
    data_jsonl: str,
    output_dir: str,
    recognizer_config: Dict,
    router_config: RouterConfig,
    K: int = 2,
) -> None:
    """
    准备校准数据集

    Args:
        data_jsonl: 验证集 JSONL 文件路径 (每行: {"image": "path/to/img.jpg", "label": "真实文本"})
        output_dir: 输出目录
        recognizer_config: Agent A 配置
        router_config: Router 配置
        K: 边界窗口大小
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化 Agent A 和 Router
    print("[1/4] 初始化 Agent A 和 Router...")
    recognizer = TextRecognizer(recognizer_config)
    router = UncertaintyRouter(router_config)

    # 读取验证集
    print(f"[2/4] 读取验证集: {data_jsonl}")
    samples = []
    with open(data_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line.strip())
            samples.append(sample)

    print(f"总样本数: {len(samples)}")

    # 提取特征和标签
    print("[3/4] 提取特征和生成标签...")
    features_list = []
    labels_list = []
    metadata_list = []

    for sample in tqdm(samples, desc="Processing"):
        image_path = sample["image"]
        T_GT = sample["label"]

        # 提取特征
        features, metadata = extract_router_features(image_path, recognizer, router)
        if features is None:
            continue

        T_A = metadata["text"]

        # 生成标签
        y_deletion = generate_deletion_label(T_A, T_GT, K=K)

        features_list.append(features)
        labels_list.append(y_deletion)
        metadata_list.append(
            {
                "image": image_path,
                "T_A": T_A,
                "T_GT": T_GT,
                "y_deletion": y_deletion,
                **metadata,
            }
        )

    # 转换为 numpy 数组
    X = np.array(features_list, dtype=np.float32)
    Y = np.array(labels_list, dtype=np.int32)

    print(f"[4/4] 保存数据集...")
    print(f"  特征矩阵 X: {X.shape}")
    print(f"  标签向量 Y: {Y.shape}")
    print(f"  正样本比例: {Y.mean():.2%}")

    # 保存
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
    parser.add_argument(
        "--data_jsonl",
        type=str,
        required=True,
        help="验证集 JSONL 文件路径",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/calibration",
        help="输出目录",
    )
    parser.add_argument(
        "--rec_model_dir",
        type=str,
        default="./models/ppocrv5_rec",
        help="Agent A 模型目录",
    )
    parser.add_argument(
        "--rec_dict_path",
        type=str,
        default="./ppocr/utils/ppocr_keys_v1.txt",
        help="字符字典路径",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=2,
        help="边界窗口大小",
    )

    args = parser.parse_args()

    # Agent A 配置
    recognizer_config = {
        "rec_model_dir": args.rec_model_dir,
        "rec_char_dict_path": args.rec_dict_path,
        "use_gpu": True,
        "gpu_mem": 500,
    }

    # Router 配置
    router_config = RouterConfig(
        entropy_threshold_low=2.0,
        entropy_threshold_high=4.0,
        blank_idx=0,
        epsilon=1e-10,
    )

    prepare_calibration_dataset(
        data_jsonl=args.data_jsonl,
        output_dir=args.output_dir,
        recognizer_config=recognizer_config,
        router_config=router_config,
        K=args.K,
    )


if __name__ == "__main__":
    main()
