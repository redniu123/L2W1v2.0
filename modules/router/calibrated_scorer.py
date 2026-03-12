#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ Stage 2: Calibrated Scorer

目标：
使用训练好的 Logistic Regression 权重计算校准后的边界风险评分 s_b

公式：
s_b = σ(w^T x + b) = 1 / (1 + exp(-(w^T x + b)))
其中 x = [v_edge, b_edge, v_edge*b_edge, drop]^T
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class CalibratedScorerConfig:
    """校准评分器配置"""

    enabled: bool = False  # 是否启用校准评分器
    weights: Dict[str, float] = None  # 权重字典
    bias: float = 0.0  # 偏置项

    def __post_init__(self):
        if self.weights is None:
            # 默认权重（等权）
            self.weights = {
                "v_edge": 0.25,
                "b_edge": 0.25,
                "v_edge_x_b_edge": 0.25,
                "drop": 0.25,
            }


class CalibratedScorer:
    """校准评分器"""

    def __init__(self, config: CalibratedScorerConfig):
        self.config = config

        # 提取权重向量
        self.w = np.array(
            [
                config.weights["v_edge"],
                config.weights["b_edge"],
                config.weights["v_edge_x_b_edge"],
                config.weights["drop"],
            ],
            dtype=np.float32,
        )
        self.b = config.bias

    def compute_score(
        self, v_edge: float, b_edge: float, drop: float
    ) -> Dict[str, float]:
        """
        计算校准后的边界风险评分

        Args:
            v_edge: 视觉边缘探针强度
            b_edge: CTC 边界 blank 强度
            drop: 置信度陡降

        Returns:
            dict: 包含 s_b 和中间特征
        """
        # 构造特征向量 x = [v_edge, b_edge, v_edge*b_edge, drop]
        x = np.array([v_edge, b_edge, v_edge * b_edge, drop], dtype=np.float32)

        # 计算 logit: z = w^T x + b
        z = np.dot(self.w, x) + self.b

        # 计算 sigmoid: s_b = 1 / (1 + exp(-z))
        s_b = self._sigmoid(z)

        return {
            "s_b": float(s_b),
            "logit": float(z),
            "features": {
                "v_edge": float(v_edge),
                "b_edge": float(b_edge),
                "v_edge_x_b_edge": float(v_edge * b_edge),
                "drop": float(drop),
            },
        }

    @staticmethod
    def _sigmoid(z: float) -> float:
        """Sigmoid 函数"""
        return 1.0 / (1.0 + np.exp(-z))

    @classmethod
    def from_config_dict(cls, config_dict: Dict) -> "CalibratedScorer":
        """从配置字典创建评分器"""
        config = CalibratedScorerConfig(
            enabled=config_dict.get("enabled", False),
            weights=config_dict.get("weights", None),
            bias=config_dict.get("bias", 0.0),
        )
        return cls(config)


class RuleOnlyScorer:
    """Rule-only 评分器（Stage 1 基线）"""

    def __init__(self, a1: float = 0.333, a2: float = 0.333, a3: float = 0.333):
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3

    def compute_score(
        self, v_edge: float, b_edge: float, drop: float
    ) -> Dict[str, float]:
        """
        计算 Rule-only 评分

        公式：
        s_b = clip(a1*(v_edge*b_edge) + a2*b_edge + a3*drop, 0, 1)
        """
        s_b = self.a1 * (v_edge * b_edge) + self.a2 * b_edge + self.a3 * drop
        s_b = np.clip(s_b, 0.0, 1.0)

        return {
            "s_b": float(s_b),
            "features": {
                "v_edge": float(v_edge),
                "b_edge": float(b_edge),
                "v_edge_x_b_edge": float(v_edge * b_edge),
                "drop": float(drop),
            },
        }
