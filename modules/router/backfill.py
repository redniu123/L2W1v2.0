#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ Stage 2: Strict Backfill Controller

目标：
1. 对 VLM 输出施加物理级约束
2. 实现路径专属回填规则（BOUNDARY / AMBIGUITY）
3. 全局拒改红线（ED > 2 或长度变化 > 20%）

公式：
Reject(T_cand) = I[ED(T_A, T_cand) > 2 ∨ |len(T_cand) - len(T_A)| / len(T_A) > 0.2]
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

import Levenshtein


class RouteType(Enum):
    """路由类型"""

    NONE = "NONE"
    BOUNDARY = "BOUNDARY"
    AMBIGUITY = "AMBIGUITY"
    BOTH = "BOTH"


class RejectionReason(Enum):
    """拒改原因"""

    ACCEPTED = "accepted"
    GLOBAL_ED_EXCEEDED = "rejected_global_ed_exceeded"
    GLOBAL_LENGTH_CHANGE = "rejected_global_length_change"
    BOUNDARY_VIOLATION = "rejected_boundary_violation"
    AMBIGUITY_VIOLATION = "rejected_ambiguity_violation"
    TOP2_MISMATCH = "rejected_top2_mismatch"
    MULTIPLE_CHANGES = "rejected_multiple_changes"


@dataclass
class BackfillConfig:
    """回填配置"""

    strict_mode: bool = True  # 启用严格回填
    max_edit_distance: int = 2  # 最大编辑距离
    max_length_change_ratio: float = 0.2  # 最大长度变化比例
    boundary_K: int = 2  # 边界窗口大小


@dataclass
class BackfillResult:
    """回填结果"""

    T_final: str  # 最终文本
    is_rejected: bool  # 是否被拒改
    rejection_reason: RejectionReason  # 拒改原因
    edit_distance: int  # 编辑距离
    length_change_ratio: float  # 长度变化比例


class StrictBackfillController:
    """严格回填控制器"""

    def __init__(self, config: BackfillConfig):
        self.config = config

    def apply_backfill(
        self,
        T_A: str,
        T_cand: str,
        route_type: RouteType,
        idx_susp: Optional[int] = None,
        top2_chars: Optional[List[str]] = None,
    ) -> BackfillResult:
        """
        应用严格回填规则

        Args:
            T_A: Agent A 原始文本
            T_cand: VLM 候选修正文本
            route_type: 路由类型
            idx_susp: 存疑字符位置（AMBIGUITY 路径必需）
            top2_chars: Top-2 候选字符（AMBIGUITY 路径必需）

        Returns:
            BackfillResult: 回填结果
        """
        if not self.config.strict_mode:
            # 非严格模式：直接接受
            return BackfillResult(
                T_final=T_cand,
                is_rejected=False,
                rejection_reason=RejectionReason.ACCEPTED,
                edit_distance=Levenshtein.distance(T_A, T_cand),
                length_change_ratio=self._compute_length_change_ratio(T_A, T_cand),
            )

        # Rule 1: 全局拒改红线
        rejection = self._check_global_rejection(T_A, T_cand)
        if rejection is not None:
            return rejection

        # Rule 2: 路径专属约束
        if route_type == RouteType.BOUNDARY:
            rejection = self._check_boundary_constraint(T_A, T_cand)
        elif route_type == RouteType.AMBIGUITY:
            rejection = self._check_ambiguity_constraint(
                T_A, T_cand, idx_susp, top2_chars
            )
        elif route_type == RouteType.BOTH:
            # BOTH 路径需要分阶段验证（暂时使用宽松策略）
            rejection = None
        else:
            rejection = None

        if rejection is not None:
            return rejection

        # 通过所有检查：接受修正
        return BackfillResult(
            T_final=T_cand,
            is_rejected=False,
            rejection_reason=RejectionReason.ACCEPTED,
            edit_distance=Levenshtein.distance(T_A, T_cand),
            length_change_ratio=self._compute_length_change_ratio(T_A, T_cand),
        )

    def _check_global_rejection(
        self, T_A: str, T_cand: str
    ) -> Optional[BackfillResult]:
        """
        检查全局拒改红线

        Returns:
            BackfillResult if rejected, None otherwise
        """
        ed = Levenshtein.distance(T_A, T_cand)
        len_change_ratio = self._compute_length_change_ratio(T_A, T_cand)

        # 检查编辑距离
        if ed > self.config.max_edit_distance:
            return BackfillResult(
                T_final=T_A,
                is_rejected=True,
                rejection_reason=RejectionReason.GLOBAL_ED_EXCEEDED,
                edit_distance=ed,
                length_change_ratio=len_change_ratio,
            )

        # 检查长度变化
        if len_change_ratio > self.config.max_length_change_ratio:
            return BackfillResult(
                T_final=T_A,
                is_rejected=True,
                rejection_reason=RejectionReason.GLOBAL_LENGTH_CHANGE,
                edit_distance=ed,
                length_change_ratio=len_change_ratio,
            )

        return None

    def _check_boundary_constraint(
        self, T_A: str, T_cand: str
    ) -> Optional[BackfillResult]:
        """
        检查 BOUNDARY 路径约束：只允许首尾修改

        Returns:
            BackfillResult if rejected, None otherwise
        """
        K = self.config.boundary_K
        N = len(T_A)

        # 获取编辑操作
        ops = Levenshtein.editops(T_A, T_cand)

        # 检查是否所有修改都在边界区域
        for op_type, pos_A, pos_cand in ops:
            # 判断位置是否在边界
            if op_type in ["replace", "delete"]:
                # 对于 replace 和 delete，检查 T_A 的位置
                if not (pos_A < K or pos_A >= N - K):
                    # 中间区域被修改，违反约束
                    return BackfillResult(
                        T_final=T_A,
                        is_rejected=True,
                        rejection_reason=RejectionReason.BOUNDARY_VIOLATION,
                        edit_distance=Levenshtein.distance(T_A, T_cand),
                        length_change_ratio=self._compute_length_change_ratio(
                            T_A, T_cand
                        ),
                    )
            elif op_type == "insert":
                # 对于 insert，检查插入位置是否在边界
                if not (pos_A <= K or pos_A >= N - K):
                    return BackfillResult(
                        T_final=T_A,
                        is_rejected=True,
                        rejection_reason=RejectionReason.BOUNDARY_VIOLATION,
                        edit_distance=Levenshtein.distance(T_A, T_cand),
                        length_change_ratio=self._compute_length_change_ratio(
                            T_A, T_cand
                        ),
                    )

        return None

    def _check_ambiguity_constraint(
        self,
        T_A: str,
        T_cand: str,
        idx_susp: Optional[int],
        top2_chars: Optional[List[str]],
    ) -> Optional[BackfillResult]:
        """
        检查 AMBIGUITY 路径约束：只允许修改 idx_susp 处字符，且必须在 Top-2 内

        Returns:
            BackfillResult if rejected, None otherwise
        """
        if idx_susp is None or top2_chars is None:
            # 缺少必要参数，拒绝修改
            return BackfillResult(
                T_final=T_A,
                is_rejected=True,
                rejection_reason=RejectionReason.AMBIGUITY_VIOLATION,
                edit_distance=Levenshtein.distance(T_A, T_cand),
                length_change_ratio=self._compute_length_change_ratio(T_A, T_cand),
            )

        # 检查是否为单点替换
        if len(T_A) != len(T_cand):
            return BackfillResult(
                T_final=T_A,
                is_rejected=True,
                rejection_reason=RejectionReason.AMBIGUITY_VIOLATION,
                edit_distance=Levenshtein.distance(T_A, T_cand),
                length_change_ratio=self._compute_length_change_ratio(T_A, T_cand),
            )

        # 检查修改位置
        changed_positions = []
        for i, (c_a, c_cand) in enumerate(zip(T_A, T_cand)):
            if c_a != c_cand:
                changed_positions.append(i)

        # 必须恰好修改一个位置
        if len(changed_positions) != 1:
            reason = (
                RejectionReason.MULTIPLE_CHANGES
                if len(changed_positions) > 1
                else RejectionReason.ACCEPTED
            )
            if len(changed_positions) > 1:
                return BackfillResult(
                    T_final=T_A,
                    is_rejected=True,
                    rejection_reason=reason,
                    edit_distance=Levenshtein.distance(T_A, T_cand),
                    length_change_ratio=self._compute_length_change_ratio(T_A, T_cand),
                )

        # 检查修改位置是否为 idx_susp
        changed_pos = changed_positions[0]
        if changed_pos != idx_susp:
            return BackfillResult(
                T_final=T_A,
                is_rejected=True,
                rejection_reason=RejectionReason.AMBIGUITY_VIOLATION,
                edit_distance=Levenshtein.distance(T_A, T_cand),
                length_change_ratio=self._compute_length_change_ratio(T_A, T_cand),
            )

        # 检查新字符是否在 Top-2 内
        new_char = T_cand[changed_pos]
        if new_char not in top2_chars:
            return BackfillResult(
                T_final=T_A,
                is_rejected=True,
                rejection_reason=RejectionReason.TOP2_MISMATCH,
                edit_distance=Levenshtein.distance(T_A, T_cand),
                length_change_ratio=self._compute_length_change_ratio(T_A, T_cand),
            )

        return None

    def _compute_length_change_ratio(self, T_A: str, T_cand: str) -> float:
        """计算长度变化比例"""
        if len(T_A) == 0:
            return 0.0
        return abs(len(T_cand) - len(T_A)) / len(T_A)


# 便捷函数
def apply_strict_backfill(
    T_A: str,
    T_cand: str,
    route_type: RouteType,
    idx_susp: Optional[int] = None,
    top2_chars: Optional[List[str]] = None,
    config: Optional[BackfillConfig] = None,
) -> BackfillResult:
    """
    便捷函数：应用严格回填

    Args:
        T_A: Agent A 原始文本
        T_cand: VLM 候选修正文本
        route_type: 路由类型
        idx_susp: 存疑字符位置
        top2_chars: Top-2 候选字符
        config: 回填配置（可选）

    Returns:
        BackfillResult: 回填结果
    """
    if config is None:
        config = BackfillConfig()

    controller = StrictBackfillController(config)
    return controller.apply_backfill(T_A, T_cand, route_type, idx_susp, top2_chars)
