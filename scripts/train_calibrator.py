#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ Stage 2: Calibrator Training

目标：
1. 加载校准数据集 (features.npy, labels.npy)
2. 训练 Logistic Regression 模型
3. 保存权重到 router_config.yaml

公式：
s_b = σ(w^T x + b) = 1 / (1 + exp(-(w^T x + b)))
其中 x = [v_edge, b_edge, v_edge*b_edge, drop]^T
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_calibration_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载校准数据集

    Args:
        data_dir: 数据目录

    Returns:
        X: 特征矩阵 (N, 4)
        Y: 标签向量 (N,)
    """
    data_dir = Path(data_dir)

    X = np.load(data_dir / "features.npy")
    Y = np.load(data_dir / "labels.npy")

    print(f"✓ 加载数据集:")
    print(f"  特征矩阵 X: {X.shape}")
    print(f"  标签向量 Y: {Y.shape}")
    print(f"  正样本比例: {Y.mean():.2%}")

    return X, Y


def train_logistic_regression(
    X: np.ndarray, Y: np.ndarray, C: float = 1.0, max_iter: int = 1000
) -> Tuple[LogisticRegression, Dict]:
    """
    训练 Logistic Regression 模型

    Args:
        X: 特征矩阵 (N, 4)
        Y: 标签向量 (N,)
        C: 正则化强度的倒数
        max_iter: 最大迭代次数

    Returns:
        model: 训练好的模型
        metrics: 评估指标
    """
    print("\n[训练 Logistic Regression]")
    print(f"  正则化参数 C: {C}")
    print(f"  最大迭代次数: {max_iter}")

    # 训练模型
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        random_state=42,
        class_weight="balanced",  # 处理类别不平衡
    )

    model.fit(X, Y)

    # 预测
    Y_pred = model.predict(X)
    Y_prob = model.predict_proba(X)[:, 1]

    # 计算指标
    accuracy = accuracy_score(Y, Y_pred)
    roc_auc = roc_auc_score(Y, Y_prob)
    pr_auc = average_precision_score(Y, Y_prob)

    metrics = {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }

    print(f"\n✓ 训练完成:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC: {pr_auc:.4f}")

    # 打印权重
    weights = model.coef_[0]
    bias = model.intercept_[0]

    print(f"\n权重向量 w:")
    feature_names = ["v_edge", "b_edge", "v_edge*b_edge", "drop"]
    for name, w in zip(feature_names, weights):
        print(f"  {name:15s}: {w:+.6f}")
    print(f"  bias            : {bias:+.6f}")

    return model, metrics


def save_weights_to_config(
    model: LogisticRegression, config_path: str, metrics: Dict
) -> None:
    """
    保存权重到 router_config.yaml

    Args:
        model: 训练好的模型
        config_path: 配置文件路径
        metrics: 评估指标
    """
    config_path = Path(config_path)

    # 读取现有配置
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 提取权重
    weights = model.coef_[0]
    bias = model.intercept_[0]

    # 更新配置（添加 calibrated_scorer 部分）
    if "sh_da_v4" not in config:
        config["sh_da_v4"] = {}

    config["sh_da_v4"]["calibrated_scorer"] = {
        "enabled": True,
        "weights": {
            "v_edge": float(weights[0]),
            "b_edge": float(weights[1]),
            "v_edge_x_b_edge": float(weights[2]),
            "drop": float(weights[3]),
        },
        "bias": float(bias),
        "metrics": {
            "accuracy": float(metrics["accuracy"]),
            "roc_auc": float(metrics["roc_auc"]),
            "pr_auc": float(metrics["pr_auc"]),
        },
        "note": "Trained by train_calibrator.py on validation set",
    }

    # 保存配置
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, indent=2)

    print(f"\n✓ 权重已保存到: {config_path}")
    print(f"  路径: sh_da_v4.calibrated_scorer")


def compute_optimal_threshold(
    model: LogisticRegression, X: np.ndarray, Y: np.ndarray, target_budget: float
) -> float:
    """
    计算最优阈值 λ_0

    根据目标预算 B，在验证集上计算 q 的 (1-B) 分位数作为初始阈值

    Args:
        model: 训练好的模型
        X: 特征矩阵
        Y: 标签向量
        target_budget: 目标预算 B (e.g., 0.2 for 20%)

    Returns:
        lambda_0: 初始阈值
    """
    # 计算校准后的评分 s_b
    s_b = model.predict_proba(X)[:, 1]

    # 计算 (1-B) 分位数
    quantile = 1 - target_budget
    lambda_0 = np.quantile(s_b, quantile)

    print(f"\n[最优阈值计算]")
    print(f"  目标预算 B: {target_budget:.1%}")
    print(f"  分位数: {quantile:.1%}")
    print(f"  λ_0: {lambda_0:.4f}")

    # 验证实际调用率
    actual_call_rate = (s_b >= lambda_0).mean()
    print(f"  实际调用率: {actual_call_rate:.2%}")

    return float(lambda_0)


def main():
    parser = argparse.ArgumentParser(description="SH-DA++ Stage 2: 校准训练器")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/calibration",
        help="校准数据集目录",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="./configs/router_config.yaml",
        help="Router 配置文件路径",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="正则化强度的倒数",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="最大迭代次数",
    )
    parser.add_argument(
        "--target_budget",
        type=float,
        default=0.2,
        help="目标预算 B (0.2 = 20%)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SH-DA++ Stage 2: Calibrator Training")
    print("=" * 60)

    # 1. 加载数据
    X, Y = load_calibration_data(args.data_dir)

    # 2. 训练模型
    model, metrics = train_logistic_regression(X, Y, C=args.C, max_iter=args.max_iter)

    # 3. 计算最优阈值
    lambda_0 = compute_optimal_threshold(model, X, Y, args.target_budget)

    # 4. 保存权重到配置文件
    save_weights_to_config(model, args.config_path, metrics)

    # 5. 更新 lambda_init
    config_path = Path(args.config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["sh_da_v4"]["budget_controller"]["lambda_init"] = lambda_0

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, indent=2)

    print(f"\n✓ λ_0 已更新到配置文件: {lambda_0:.4f}")

    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
