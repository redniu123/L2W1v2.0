#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v4.0 Router (Stage 2 Integration)

集成 CalibratedScorer 和 RuleOnlyScorer，根据配置自动切换。

核心功能：
1. 根据 router_config.yaml 配置选择评分器
2. 计算边界风险 s_b 和识别歧义 s_a
3. 在线预算控制（动态调整 λ）
4. 战略分诊（NONE/BOUNDARY/AMBIGUITY/BOTH）
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import yaml

from modules.router.calibrated_scorer import CalibratedScorer, CalibratedScorerConfig
from modules.router.uncertainty_router import (
    BudgetControllerConfig,
    OnlineBudgetController,
    RouteType,
    RuleOnlyScorer,
    RuleScorerConfig,
    ScoringResult,
)

logger = logging.getLogger(__name__)


@dataclass
class SHDARouterConfig:
    """SH-DA++ Router 配置"""

    # 评分器模式
    use_calibrated_scorer: bool = False  # 是否使用校准评分器

    # RuleOnlyScorer 配置
    rule_scorer: RuleScorerConfig = None

    # CalibratedScorer 配置
    calibrated_scorer: CalibratedScorerConfig = None

    # 预算控制器配置
    budget_controller: BudgetControllerConfig = None

    def __post_init__(self):
        if self.rule_scorer is None:
            self.rule_scorer = RuleScorerConfig()
        if self.calibrated_scorer is None:
            self.calibrated_scorer = CalibratedScorerConfig()
        if self.budget_controller is None:
            self.budget_controller = BudgetControllerConfig()


@dataclass
class RouterDecision:
    """路由决策结果"""

    upgrade: bool  # 是否升级到 Agent B
    route_type: RouteType  # 分诊类型
    s_b: float  # 边界风险评分
    s_a: float  # 识别歧义评分
    q: float  # 综合优先级
    lambda_current: float  # 当前阈值
    idx_susp: Optional[int] = None  # 存疑字符位置（AMBIGUITY 路径）
    top2_chars: Optional[list] = None  # Top-2 候选字符（AMBIGUITY 路径）
    details: Dict = None  # 详细信息

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "upgrade": self.upgrade,
            "route_type": self.route_type.value,
            "s_b": round(self.s_b, 6),
            "s_a": round(self.s_a, 6),
            "q": round(self.q, 6),
            "lambda": round(self.lambda_current, 6),
            "idx_susp": self.idx_susp,
            "top2_chars": self.top2_chars,
            "details": self.details,
        }


class SHDARouter:
    """
    SH-DA++ v4.0 Router (Stage 2 Integration)

    根据配置自动选择 RuleOnlyScorer 或 CalibratedScorer。
    """

    def __init__(self, config: SHDARouterConfig):
        """
        初始化 Router

        Args:
            config: Router 配置
        """
        self.config = config

        # 初始化评分器
        if config.use_calibrated_scorer:
            logger.info("使用 CalibratedScorer（校准模式）")
            self.scorer = CalibratedScorer(config.calibrated_scorer)
            self.scorer_mode = "calibrated"
        else:
            logger.info("使用 RuleOnlyScorer（规则模式）")
            self.scorer = RuleOnlyScorer(config.rule_scorer)
            self.scorer_mode = "rule_only"

        # 初始化预算控制器
        self.budget_controller = OnlineBudgetController(config.budget_controller)

        logger.info(f"Router 初始化完成，模式={self.scorer_mode}")

    def route(
        self,
        boundary_stats: Dict,
        top2_info: Dict,
        r_d: float = 0.0,
        v_edge: float = None,
        char_count: int = 0,
        expected_char_count: int = 0,
        agent_a_text: str = "",
    ) -> RouterDecision:
        """
        执行路由决策

        Args:
            boundary_stats: Stage 0 边界统计量
            top2_info: Stage 0 Top-2 信息
            r_d: 额外风险因子
            v_edge: 边界区域视觉熵
            char_count: 实际识别字符数
            expected_char_count: 预期字符数
            agent_a_text: Agent A 输出文本

        Returns:
            RouterDecision: 路由决策结果
        """
        # Step 1: 计算评分
        if self.scorer_mode == "calibrated":
            # 使用校准评分器
            scoring_result = self._score_with_calibrated(
                boundary_stats, top2_info, v_edge
            )
        else:
            # 使用规则评分器
            scoring_result = self.scorer.score(
                boundary_stats=boundary_stats,
                top2_info=top2_info,
                r_d=r_d,
                v_edge=v_edge,
                char_count=char_count,
                expected_char_count=expected_char_count,
                agent_a_text=agent_a_text,
            )

        # Step 2: 预算控制决策
        upgrade, budget_details = self.budget_controller.step(scoring_result.q)

        # Step 3: 提取 idx_susp 和 top2_chars（用于 AMBIGUITY 路径）
        idx_susp, top2_chars = self._extract_ambiguity_info(top2_info)

        # Step 4: 构造决策结果
        decision = RouterDecision(
            upgrade=upgrade,
            route_type=scoring_result.route_type,
            s_b=scoring_result.s_b,
            s_a=scoring_result.s_a,
            q=scoring_result.q,
            lambda_current=self.budget_controller.current_lambda,
            idx_susp=idx_susp,
            top2_chars=top2_chars,
            details={
                "scoring_details": scoring_result.details,
                "budget_details": budget_details,
                "scorer_mode": self.scorer_mode,
            },
        )

        return decision

    def _score_with_calibrated(
        self, boundary_stats: Dict, top2_info: Dict, v_edge: float
    ) -> ScoringResult:
        """
        使用 CalibratedScorer 计算评分

        Args:
            boundary_stats: 边界统计量
            top2_info: Top-2 信息
            v_edge: 边界视觉熵

        Returns:
            ScoringResult: 评分结果
        """
        # 提取特征
        b_edge = self._extract_b_edge(boundary_stats)
        drop = self._extract_drop(boundary_stats)

        # 计算 s_b（使用校准评分器）
        # v5.1: 从 top2_info 提取 Mean/Min Confidence
        top1_probs = (top2_info or {}).get("top1_probs") or []
        if top1_probs:
            mean_conf = float(np.mean(top1_probs))
            min_conf = float(np.min(top1_probs))
        else:
            mean_conf = 0.5
            min_conf = 0.5

        s_b_result = self.scorer.compute_score(
            mean_conf=mean_conf,
            min_conf=min_conf,
            b_edge=b_edge,
            drop=drop,
            r_d=0.0,
        )
        s_b = s_b_result["s_b"]

        # 计算 s_a（使用 RuleOnlyScorer 的方法）
        rule_scorer = RuleOnlyScorer(self.config.rule_scorer)
        s_a, s_a_details = rule_scorer.compute_ambiguity_score(top2_info)

        # 计算 q（暂不考虑 r_d）
        q = max(s_b, s_a)

        # 判定路由类型
        route_type = rule_scorer.determine_route_type(
            s_b, s_a, self.budget_controller.current_lambda
        )

        return ScoringResult(
            s_b=s_b,
            s_a=s_a,
            q=q,
            route_type=route_type,
            details={
                "s_b_details": s_b_result,
                "s_a_details": s_a_details,
                "scorer_mode": "calibrated",
            },
        )

    def _extract_b_edge(self, boundary_stats: Dict) -> float:
        """提取 b_edge"""
        if not boundary_stats or not boundary_stats.get("valid", False):
            return 0.0

        blank_mean_L = boundary_stats.get("blank_mean_L", 0.0)
        blank_mean_R = boundary_stats.get("blank_mean_R", 0.0)
        blank_peak_L = boundary_stats.get("blank_peak_L", 0.0)
        blank_peak_R = boundary_stats.get("blank_peak_R", 0.0)

        b_edge_L = 0.6 * blank_mean_L + 0.4 * blank_peak_L
        b_edge_R = 0.6 * blank_mean_R + 0.4 * blank_peak_R
        return max(b_edge_L, b_edge_R)

    def _extract_drop(self, boundary_stats: Dict) -> float:
        """提取 drop（暂时返回 0，需要从 boundary_stats 中获取）"""
        # TODO: 从 boundary_stats 中提取 drop 信息
        return 0.0

    def _extract_ambiguity_info(self, top2_info: Dict) -> Tuple[Optional[int], Optional[list]]:
        """
        提取 AMBIGUITY 路径所需的信息

        Args:
            top2_info: Top-2 信息

        Returns:
            Tuple[idx_susp, top2_chars]: 存疑位置和 Top-2 候选字符
        """
        if not top2_info or top2_info.get("top2_status") != "available":
            return None, None

        # 找到最小 margin 的位置
        top1_probs = top2_info.get("top1_probs")
        top2_probs = top2_info.get("top2_probs")
        top1_chars = top2_info.get("top1_chars")
        top2_chars_list = top2_info.get("top2_chars")

        if (
            top1_probs is None
            or top2_probs is None
            or top1_chars is None
            or top2_chars_list is None
        ):
            return None, None

        top1_probs = np.array(top1_probs)
        top2_probs = np.array(top2_probs)
        margins = top1_probs - top2_probs
        idx_susp = int(np.argmin(margins))

        # 提取 Top-2 字符
        if idx_susp < len(top1_chars) and idx_susp < len(top2_chars_list):
            top2_chars = [top1_chars[idx_susp], top2_chars_list[idx_susp]]
        else:
            top2_chars = None

        return idx_susp, top2_chars

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "scorer_mode": self.scorer_mode,
            "budget_controller": self.budget_controller.get_stats(),
        }

    def reset(self):
        """重置 Router 状态"""
        self.budget_controller.reset()

    @classmethod
    def from_yaml(cls, config_path: str) -> "SHDARouter":
        """
        从 YAML 配置文件创建 Router

        Args:
            config_path: 配置文件路径

        Returns:
            SHDARouter: Router 实例
        """
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)

        # 解析配置
        sh_da_config = config_dict.get("sh_da_v4", {})

        # 检查是否启用校准评分器
        calibrated_config = sh_da_config.get("calibrated_scorer", {})
        use_calibrated = calibrated_config.get("enabled", False)

        # 构造配置对象
        rule_scorer_config = RuleScorerConfig(
            v_min=sh_da_config.get("rule_scorer", {}).get("v_min", 0.0),
            v_max=sh_da_config.get("rule_scorer", {}).get("v_max", 5.0),
            lambda_threshold=sh_da_config.get("rule_scorer", {}).get(
                "lambda_threshold", 0.5
            ),
            eta=sh_da_config.get("rule_scorer", {}).get("eta", 0.5),
            geology_dict_path=sh_da_config.get("rule_scorer", {}).get(
                "geology_dict_path", "data/dicts/Geology.txt"
            ),
            geology_min_len=sh_da_config.get("rule_scorer", {}).get(
                "geology_min_len", 2
            ),
            geology_risk_weight=sh_da_config.get("rule_scorer", {}).get(
                "geology_risk_weight", 1.0
            ),
            wur_mean_weight=sh_da_config.get("rule_scorer", {}).get(
                "wur_mean_weight", 0.5
            ),
            wur_min_weight=sh_da_config.get("rule_scorer", {}).get(
                "wur_min_weight", 0.3
            ),
            wur_drop_weight=sh_da_config.get("rule_scorer", {}).get(
                "wur_drop_weight", 0.2
            ),
            wur_min_conf_gate_threshold=sh_da_config.get("rule_scorer", {}).get(
                "wur_min_conf_gate_threshold", 0.35
            ),
            wur_drop_gate_threshold=sh_da_config.get("rule_scorer", {}).get(
                "wur_drop_gate_threshold", 0.20
            ),
            wur_gate_bonus=sh_da_config.get("rule_scorer", {}).get(
                "wur_gate_bonus", 0.10
            ),
        )

        calibrated_scorer_config = CalibratedScorerConfig(
            enabled=use_calibrated,
            weights=calibrated_config.get("weights", {}),
            bias=calibrated_config.get("bias", 0.0),
        )

        budget_controller_config = BudgetControllerConfig(
            window_size=sh_da_config.get("budget_controller", {}).get(
                "window_size", 500
            ),
            warmup_samples=sh_da_config.get("budget_controller", {}).get(
                "warmup_samples"
            ),
            k=sh_da_config.get("budget_controller", {}).get(
                "k", sh_da_config.get("budget_controller", {}).get("alpha", 0.01)
            ),
            lambda_min=sh_da_config.get("budget_controller", {}).get("lambda_min", 0.0),
            lambda_max=sh_da_config.get("budget_controller", {}).get("lambda_max", 2.0),
            lambda_init=sh_da_config.get("budget_controller", {}).get(
                "lambda_init", 0.5
            ),
            target_budget=sh_da_config.get("budget_controller", {}).get(
                "target_budget", 0.2
            ),
        )

        router_config = SHDARouterConfig(
            use_calibrated_scorer=use_calibrated,
            rule_scorer=rule_scorer_config,
            calibrated_scorer=calibrated_scorer_config,
            budget_controller=budget_controller_config,
        )

        return cls(router_config)
