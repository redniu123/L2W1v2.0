#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v5.1 Phase 1.2 & 1.3: 校准训练 + b_edge 消融实验

严格验证原则：
  - 仅在 features_train.npy 上执行 .fit()
  - 在 features_val.npy 上搜索 lambda_0 并计算 AUC 指标
  - Test 集严格封存，本脚本禁止触碰

特征顺序（5维）：
  [Mean_Confidence, Min_Confidence, b_edge, drop, r_d]
  b_edge 位于第 2 列（index=2）
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FEATURE_NAMES = ["Mean_Confidence", "Min_Confidence", "b_edge", "drop", "r_d"]
B_EDGE_IDX = 2  # b_edge 在特征矩阵中的列索引


def load_split(
    data_dir: str, split: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载指定 split 的特征与标签

    Args:
        data_dir: 特征目录
        split:    'train' | 'val' | 'test'

    Returns:
        X: (N, 5), Y: (N,)
    """
    data_dir = Path(data_dir)
    X = np.load(data_dir / f"features_{split}.npy")
    Y = np.load(data_dir / f"labels_{split}.npy")
    print(f"  [{split:5s}] X={X.shape}, Y={Y.shape}, 正样本={Y.mean():.2%}")
    return X, Y


def train_lr(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
) -> LogisticRegression:
    """在 Train 集上训练 Logistic Regression"""
    model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, Y_train)
    return model


def evaluate_on_val(
    model: LogisticRegression,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    feature_names: list,
    label: str = "完整特征",
) -> Dict:
    """在 Val 集上计算 ROC-AUC 和 PR-AUC"""
    Y_prob = model.predict_proba(X_val)[:, 1]
    roc_auc = roc_auc_score(Y_val, Y_prob)
    pr_auc = average_precision_score(Y_val, Y_prob)

    print(f"\n[{label}]")
    if feature_names:
        weights = model.coef_[0]
        bias = model.intercept_[0]
        print(f"  权重向量 w:")
        for name, w in zip(feature_names, weights):
            print(f"    {name:20s}: {w:+.6f}")
        print(f"    {'bias':20s}: {bias:+.6f}")
    print(f"  Val ROC-AUC : {roc_auc:.4f}")
    print(f"  Val PR-AUC  : {pr_auc:.4f}")

    return {"roc_auc": roc_auc, "pr_auc": pr_auc, "Y_prob": Y_prob}


def compute_lambda(
    Y_prob: np.ndarray,
    target_budget: float,
) -> Tuple[float, float]:
    """在 Val 集上通过分位数搜索 lambda_0"""
    quantile = 1.0 - target_budget
    lambda_0 = float(np.quantile(Y_prob, quantile))
    actual_rate = float((Y_prob >= lambda_0).mean())
    return lambda_0, actual_rate


def save_to_config(
    model: LogisticRegression,
    feature_names: list,
    lambda_0: float,
    metrics: Dict,
    config_path: str,
) -> None:
    """将权重和 lambda_0 写入 router_config.yaml"""
    config_path = Path(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    weights = model.coef_[0]
    bias = model.intercept_[0]

    if "sh_da_v4" not in config:
        config["sh_da_v4"] = {}

    config["sh_da_v4"]["calibrated_scorer"] = {
        "enabled": True,
        "feature_version": "v5.1",
        "feature_names": feature_names,
        "weights": {name: float(w) for name, w in zip(feature_names, weights)},
        "bias": float(bias),
        "metrics": {
            "val_roc_auc": float(metrics["roc_auc"]),
            "val_pr_auc": float(metrics["pr_auc"]),
        },
        "note": "SH-DA++ v5.1: 5-dim feature, trained on Train split only",
    }
    config["sh_da_v4"]["budget_controller"]["lambda_init"] = lambda_0

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, indent=2)

    print(f"\n✓ 权重已写入: {config_path}")


def main():
    parser = argparse.ArgumentParser(description="SH-DA++ v5.1: 校准训练器")
    parser.add_argument("--data_dir", type=str, default="./results/stage2_v51",
                        help="特征目录（含 features_train/val.npy）")
    parser.add_argument("--config_path", type=str, default="./configs/router_config.yaml")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max_iter", type=int, default=1000)
    parser.add_argument("--target_budget", type=float, default=0.2)
    args = parser.parse_args()

    print("=" * 60)
    print("SH-DA++ v5.1: Calibrator Training (Phase 1.2 & 1.3)")
    print("=" * 60)

    # 加载 Train & Val（Test 严格封存）
    print("\n加载数据集:")
    X_train, Y_train = load_split(args.data_dir, "train")
    X_val, Y_val = load_split(args.data_dir, "val")

    # =========================================================
    # Phase 1.2: 主训练流程（5 维特征）
    # =========================================================
    print("\n" + "-" * 40)
    print("Phase 1.2: 主训练流程（5 维特征）")
    print("-" * 40)
    model_full = train_lr(X_train, Y_train, C=args.C, max_iter=args.max_iter)
    metrics_full = evaluate_on_val(model_full, X_val, Y_val, FEATURE_NAMES, label="完整特征 (5维)")

    # 计算 lambda_0（在 Val 集上搜索）
    lambda_0, actual_rate = compute_lambda(metrics_full["Y_prob"], args.target_budget)
    print(f"\n[预算控制]")
    print(f"  目标调用率 B : {args.target_budget:.1%}")
    print(f"  λ_0          : {lambda_0:.4f}")
    print(f"  实际调用率   : {actual_rate:.2%}")

    # 保存到配置
    save_to_config(model_full, FEATURE_NAMES, lambda_0, metrics_full, args.config_path)

    # =========================================================
    # Phase 1.3: 强制 b_edge 消融实验
    # =========================================================
    print("\n" + "-" * 40)
    print("Phase 1.3: b_edge 消融实验")
    print("-" * 40)

    # 剔除 b_edge 列（index=2）
    ablation_cols = [i for i in range(X_train.shape[1]) if i != B_EDGE_IDX]
    ablation_names = [FEATURE_NAMES[i] for i in ablation_cols]

    X_train_abl = X_train[:, ablation_cols]
    X_val_abl = X_val[:, ablation_cols]

    model_abl = train_lr(X_train_abl, Y_train, C=args.C, max_iter=args.max_iter)
    metrics_abl = evaluate_on_val(model_abl, X_val_abl, Y_val, ablation_names, label="移除 b_edge (4维)")

    # 输出消融诊断
    delta = metrics_full["pr_auc"] - metrics_abl["pr_auc"]
    sign = "+" if delta >= 0 else ""
    print(f"\n[消融诊断] b_edge 的独立贡献:")
    print(f"  - 完整特征 (5维) PR-AUC  : {metrics_full['pr_auc']:.4f}")
    print(f"  - 移除 b_edge (4维) PR-AUC: {metrics_abl['pr_auc']:.4f}")
    print(f"  - Δ PR-AUC               : {sign}{delta:.4f}")

    print("\n" + "=" * 60)
    print("训练完成！（Test 集已严格封存，未被触碰）")
    print("=" * 60)


if __name__ == "__main__":
    main()
