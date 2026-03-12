#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v4.0 Integrated Pipeline (Stage 2)

集成 Router、Agent B、BackfillController 和熔断机制。

核心流程：
1. Agent A 推理 → 提取特征
2. Router 决策 → 判断是否升级
3. Agent B 修正 → 生成候选文本
4. BackfillController → 严格回填与拒改
5. 熔断监控 → CVR > 30% 触发降级
"""

import json
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from modules.router.backfill import (
    BackfillConfig,
    BackfillResult,
    RouteType,
    StrictBackfillController,
)
from modules.router.sh_da_router import RouterDecision, SHDARouter
from modules.vlm_expert.constrained_prompter import ConstrainedPrompter

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Pipeline 配置"""

    # 回填配置
    backfill_config: BackfillConfig = None

    # 熔断配置
    meltdown_enabled: bool = True
    meltdown_cvr_threshold: float = 0.3  # CVR > 30% 触发熔断
    meltdown_window_size: int = 100  # 熔断监控窗口大小

    # 日志配置
    enable_logging: bool = True
    backfill_log_path: str = "./results/backfill_log.jsonl"

    def __post_init__(self):
        if self.backfill_config is None:
            self.backfill_config = BackfillConfig()


@dataclass
class PipelineResult:
    """Pipeline 处理结果"""

    T_final: str  # 最终文本
    upgrade: bool  # 是否升级
    route_type: str  # 路由类型
    is_rejected: bool  # 是否被拒改
    rejection_reason: str  # 拒改原因
    meltdown_active: bool  # 是否处于熔断状态
    router_decision: Dict  # Router 决策详情
    backfill_result: Dict  # 回填结果详情

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "T_final": self.T_final,
            "upgrade": self.upgrade,
            "route_type": self.route_type,
            "is_rejected": self.is_rejected,
            "rejection_reason": self.rejection_reason,
            "meltdown_active": self.meltdown_active,
            "router_decision": self.router_decision,
            "backfill_result": self.backfill_result,
        }


class MeltdownMonitor:
    """
    熔断监控器

    监控 CVR（Constraint Violation Rate），当 CVR > 30% 时触发熔断。
    """

    def __init__(self, window_size: int = 100, cvr_threshold: float = 0.3):
        """
        初始化熔断监控器

        Args:
            window_size: 滑动窗口大小
            cvr_threshold: CVR 阈值
        """
        self.window_size = window_size
        self.cvr_threshold = cvr_threshold

        # 滑动窗口：记录最近的拒改事件
        self._rejection_history: deque = deque(maxlen=window_size)

        # 熔断状态
        self._meltdown_active = False

        # 统计信息
        self._total_upgrades = 0
        self._total_rejections = 0

    @property
    def is_meltdown_active(self) -> bool:
        """是否处于熔断状态"""
        return self._meltdown_active

    @property
    def current_cvr(self) -> float:
        """当前 CVR（窗口内）"""
        if not self._rejection_history:
            return 0.0
        return sum(self._rejection_history) / len(self._rejection_history)

    @property
    def total_cvr(self) -> float:
        """总体 CVR"""
        if self._total_upgrades == 0:
            return 0.0
        return self._total_rejections / self._total_upgrades

    def record(self, is_rejected: bool) -> bool:
        """
        记录一次升级事件

        Args:
            is_rejected: 是否被拒改

        Returns:
            bool: 是否触发熔断
        """
        self._total_upgrades += 1
        if is_rejected:
            self._total_rejections += 1

        # 记录到滑动窗口
        self._rejection_history.append(1 if is_rejected else 0)

        # 检查是否触发熔断
        if len(self._rejection_history) >= self.window_size:
            cvr = self.current_cvr
            if cvr > self.cvr_threshold:
                if not self._meltdown_active:
                    logger.critical(
                        f"熔断触发！CVR={cvr:.2%} > {self.cvr_threshold:.2%}"
                    )
                    self._meltdown_active = True
                return True

        return False

    def reset(self):
        """重置熔断监控器"""
        self._rejection_history.clear()
        self._meltdown_active = False
        self._total_upgrades = 0
        self._total_rejections = 0

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "meltdown_active": self._meltdown_active,
            "current_cvr": self.current_cvr,
            "total_cvr": self.total_cvr,
            "total_upgrades": self._total_upgrades,
            "total_rejections": self._total_rejections,
            "window_size": self.window_size,
            "cvr_threshold": self.cvr_threshold,
        }


class SHDAPipeline:
    """
    SH-DA++ v4.0 集成 Pipeline

    完整流程：Agent A → Router → Agent B → BackfillController → 熔断监控
    """

    def __init__(
        self,
        router: SHDARouter,
        config: PipelineConfig = None,
    ):
        """
        初始化 Pipeline

        Args:
            router: SH-DA++ Router
            config: Pipeline 配置
        """
        self.router = router
        self.config = config or PipelineConfig()

        # 初始化组件
        self.backfill_controller = StrictBackfillController(self.config.backfill_config)
        self.prompter = ConstrainedPrompter()

        # 初始化熔断监控器
        if self.config.meltdown_enabled:
            self.meltdown_monitor = MeltdownMonitor(
                window_size=self.config.meltdown_window_size,
                cvr_threshold=self.config.meltdown_cvr_threshold,
            )
        else:
            self.meltdown_monitor = None

        # 日志文件
        if self.config.enable_logging:
            self.backfill_log_path = Path(self.config.backfill_log_path)
            self.backfill_log_path.parent.mkdir(parents=True, exist_ok=True)
            self.backfill_log_file = open(
                self.backfill_log_path, "a", encoding="utf-8"
            )
        else:
            self.backfill_log_file = None

        logger.info("SH-DA++ Pipeline 初始化完成")

    def process(
        self,
        T_A: str,
        boundary_stats: Dict,
        top2_info: Dict,
        image_path: str = None,
        agent_b_callable: Optional[callable] = None,
        **kwargs,
    ) -> PipelineResult:
        """
        处理单个样本

        Args:
            T_A: Agent A 输出文本
            boundary_stats: 边界统计量
            top2_info: Top-2 信息
            image_path: 图像路径
            agent_b_callable: Agent B 调用函数（可选）
            **kwargs: 其他参数（r_d, v_edge 等）

        Returns:
            PipelineResult: 处理结果
        """
        # Step 1: Router 决策
        router_decision = self.router.route(
            boundary_stats=boundary_stats,
            top2_info=top2_info,
            agent_a_text=T_A,
            **kwargs,
        )

        # Step 2: 检查熔断状态
        meltdown_active = (
            self.meltdown_monitor.is_meltdown_active
            if self.meltdown_monitor
            else False
        )

        if meltdown_active:
            logger.warning("熔断激活，跳过 Agent B 调用")
            return PipelineResult(
                T_final=T_A,
                upgrade=False,
                route_type=router_decision.route_type.value,
                is_rejected=False,
                rejection_reason="meltdown_active",
                meltdown_active=True,
                router_decision=router_decision.to_dict(),
                backfill_result={},
            )

        # Step 3: 判断是否升级
        if not router_decision.upgrade:
            return PipelineResult(
                T_final=T_A,
                upgrade=False,
                route_type=router_decision.route_type.value,
                is_rejected=False,
                rejection_reason="no_upgrade",
                meltdown_active=False,
                router_decision=router_decision.to_dict(),
                backfill_result={},
            )

        # Step 4: 生成受限提示词
        prompt = self._generate_prompt(router_decision, T_A, image_path)

        # Step 5: 调用 Agent B（如果提供）
        if agent_b_callable is None:
            # 模拟模式：直接返回 T_A
            T_cand = T_A
            logger.warning("未提供 agent_b_callable，使用模拟模式")
        else:
            T_cand = agent_b_callable(prompt)

        # Step 6: 严格回填
        backfill_result = self.backfill_controller.apply_backfill(
            T_A=T_A,
            T_cand=T_cand,
            route_type=router_decision.route_type,
            idx_susp=router_decision.idx_susp,
            top2_chars=router_decision.top2_chars,
        )

        # Step 7: 记录熔断监控
        if self.meltdown_monitor:
            self.meltdown_monitor.record(backfill_result.is_rejected)

        # Step 8: 记录日志
        if self.backfill_log_file:
            self._log_backfill(
                T_A=T_A,
                T_cand=T_cand,
                backfill_result=backfill_result,
                router_decision=router_decision,
                prompt=prompt,
            )

        # Step 9: 构造结果
        result = PipelineResult(
            T_final=backfill_result.T_final,
            upgrade=True,
            route_type=router_decision.route_type.value,
            is_rejected=backfill_result.is_rejected,
            rejection_reason=backfill_result.rejection_reason.value,
            meltdown_active=False,
            router_decision=router_decision.to_dict(),
            backfill_result={
                "T_cand": T_cand,
                "edit_distance": backfill_result.edit_distance,
                "length_change_ratio": backfill_result.length_change_ratio,
            },
        )

        return result

    def _generate_prompt(
        self, router_decision: RouterDecision, T_A: str, image_path: str
    ) -> Dict:
        """生成受限提示词"""
        route_type = router_decision.route_type.value

        if route_type == "boundary":
            return self.prompter.generate_boundary_prompt(T_A, image_path)
        elif route_type == "ambiguity":
            return self.prompter.generate_ambiguity_prompt(
                T_A,
                router_decision.idx_susp,
                router_decision.top2_chars,
                image_path,
            )
        elif route_type == "both":
            return self.prompter.generate_both_prompts(
                T_A,
                router_decision.idx_susp,
                router_decision.top2_chars,
                image_path,
            )
        else:
            return {}

    def _log_backfill(
        self,
        T_A: str,
        T_cand: str,
        backfill_result: BackfillResult,
        router_decision: RouterDecision,
        prompt: Dict,
    ):
        """记录回填日志"""
        log_entry = {
            "T_A": T_A,
            "T_cand": T_cand,
            "T_final": backfill_result.T_final,
            "route_type": router_decision.route_type.value,
            "is_rejected": backfill_result.is_rejected,
            "rejection_reason": backfill_result.rejection_reason.value,
            "edit_distance": backfill_result.edit_distance,
            "length_change_ratio": backfill_result.length_change_ratio,
            "idx_susp": router_decision.idx_susp,
            "top2_chars": router_decision.top2_chars,
            "prompt_type": prompt.get("prompt_type"),
        }

        self.backfill_log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        self.backfill_log_file.flush()

    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = {
            "router": self.router.get_stats(),
        }

        if self.meltdown_monitor:
            stats["meltdown"] = self.meltdown_monitor.get_stats()

        return stats

    def reset(self):
        """重置 Pipeline 状态"""
        self.router.reset()
        if self.meltdown_monitor:
            self.meltdown_monitor.reset()

    def close(self):
        """关闭 Pipeline"""
        if self.backfill_log_file:
            self.backfill_log_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
