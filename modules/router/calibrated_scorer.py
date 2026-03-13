#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v5.1: Calibrated Scorer

公式：
  s_b = σ(w^T x + b)
  其中 x = [Mean_Confidence, Min_Confidence, b_edge, drop, r_d]^T

变更说明（v4.0 → v5.1）：
  - 删除 v_edge 及 v_edge_x_b_edge
  - 新增 Mean_Confidence、Min_Confidence
  - 激活 r_d（领域语义安全锚点）
  - 特征顺序与 train_calibrator.py 严格对齐
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# v5.1 特征名（顺序必须与训练时一致）
FEATURE_NAMES_V51 = ["Mean_Confidence", "Min_Confidence", "b_edge", "drop", "r_d"]

# 向后兼容：v4.0 旧特征名（仅用于识别旧配置）
FEATURE_NAMES_V40 = ["v_edge", "b_edge", "v_edge_x_b_edge", "drop"]


@dataclass
class CalibratedScorerConfig:
    """校准评分器配置"""

    enabled: bool = False
    weights: Dict[str, float] = None
    bias: float = 0.0
    feature_version: str = "v5.1"  # 特征版本标识
    feature_names: List[str] = None  # 特征名列表（从配置文件读取）

    def __post_init__(self):
        if self.weights is None:
            # v5.1 默认等权
            self.weights = {name: 0.2 for name in FEATURE_NAMES_V51}
        if self.feature_names is None:
            # 根据 weights 的 key 自动推断特征版本
            if "v_edge" in self.weights:
                self.feature_names = FEATURE_NAMES_V40
                self.feature_version = "v4.0"
            else:
                self.feature_names = FEATURE_NAMES_V51
                self.feature_version = "v5.1"


class CalibratedScorer:
    """
    SH-DA++ v5.1 校准评分器

    支持 v5.1（5维）和 v4.0（4维）特征，自动从配置文件推断。
    """

    def __init__(self, config: CalibratedScorerConfig):
        self.config = config
        self.feature_names = config.feature_names
        self.feature_version = config.feature_version

        # 按特征名顺序构造权重向量
        self.w = np.array(
            [config.weights.get(name, 0.0) for name in self.feature_names],
            dtype=np.float32,
        )
        self.b = float(config.bias)

    def compute_score_v51(
        self,
        mean_conf: float,
        min_conf: float,
        b_edge: float,
        drop: float,
        r_d: float,
    ) -> Dict[str, float]:
        """
        v5.1 评分接口

        Args:
            mean_conf: 全局置信度均值
            min_conf:  局部最低置信度
            b_edge:    CTC 边界 blank 强度
            drop:      左右边界不对称性
            r_d:       领域语义安全锚点

        Returns:
            dict: 包含 s_b 和特征值
        """
        x = np.array([mean_conf, min_conf, b_edge, drop, r_d], dtype=np.float32)
        z = np.dot(self.w, x) + self.b
        s_b = self._sigmoid(z)

        return {
            "s_b": float(s_b),
            "logit": float(z),
            "features": {
                "Mean_Confidence": float(mean_conf),
                "Min_Confidence": float(min_conf),
                "b_edge": float(b_edge),
                "drop": float(drop),
                "r_d": float(r_d),
            },
        }

    def compute_score(
        self,
        mean_conf: float = 0.0,
        min_conf: float = 0.0,
        b_edge: float = 0.0,
        drop: float = 0.0,
        r_d: float = 0.0,
        # v4.0 兼容参数（已废弃，保留接口不报错）
        v_edge: float = None,
    ) -> Dict[str, float]:
        """
        统一评分接口（v5.1 优先，兼容 v4.0）
        """
        if self.feature_version == "v4.0" and v_edge is not None:
            # v4.0 旧路径（仅向后兼容）
            x = np.array(
                [v_edge, b_edge, v_edge * b_edge, drop], dtype=np.float32
            )
        else:
            # v5.1 新路径
            x = np.array(
                [mean_conf, min_conf, b_edge, drop, r_d], dtype=np.float32
            )

        # 如果维度不匹配，截断或补零
        if len(x) > len(self.w):
            x = x[:len(self.w)]
        elif len(x) < len(self.w):
            x = np.pad(x, (0, len(self.w) - len(x)))

        z = np.dot(self.w, x) + self.b
        s_b = self._sigmoid(z)

        return {
            "s_b": float(s_b),
            "logit": float(z),
            "feature_version": self.feature_version,
        }

    @staticmethod
    def _sigmoid(z: float) -> float:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    @classmethod
    def from_config_dict(cls, config_dict: Dict) -> "CalibratedScorer":
        """从配置字典创建评分器（自动检测特征版本）"""
        weights = config_dict.get("weights", {})
        feature_names = config_dict.get("feature_names", None)
        feature_version = config_dict.get("feature_version", "v5.1")

        config = CalibratedScorerConfig(
            enabled=config_dict.get("enabled", False),
            weights=weights,
            bias=config_dict.get("bias", 0.0),
            feature_version=feature_version,
            feature_names=feature_names,
        )
        return cls(config)


class RuleOnlyScorer:
    """Rule-only 评分器（随机基线 / ConfOnly 基线复用）"""

    def __init__(self, a1: float = 0.5, a2: float = 0.3, a3: float = 0.2):
        self.a1 = a1  # Mean_Confidence 权重
        self.a2 = a2  # b_edge 权重
        self.a3 = a3  # drop 权重

    def compute_score(
        self,
        mean_conf: float = 0.0,
        b_edge: float = 0.0,
        drop: float = 0.0,
        # v4.0 兼容
        v_edge: float = None,
        v_edge_x_b_edge: float = None,
    ) -> Dict[str, float]:
        """
        Rule-only 评分：基于置信度和 blank 边界
        """
        # 风险 = 低置信度 + 高 blank 边界 + 高不对称性
        risk = self.a1 * (1.0 - mean_conf) + self.a2 * b_edge + self.a3 * drop
        s_b = float(np.clip(risk, 0.0, 1.0))
        return {"s_b": s_b}
