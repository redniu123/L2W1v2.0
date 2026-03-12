#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ Stage 2: Module Testing Script

测试 Stage 2 的三个核心模块：
1. 标签生成器 (prepare_calibration_data.py)
2. 校准评分器 (calibrated_scorer.py)
3. 严格回填控制器 (backfill.py)
"""

import sys
from pathlib import Path

import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.router.backfill import (
    BackfillConfig,
    RejectionReason,
    RouteType,
    StrictBackfillController,
)
from modules.router.calibrated_scorer import CalibratedScorer, CalibratedScorerConfig
from scripts.prepare_calibration_data import generate_deletion_label


def test_label_generator():
    """测试标签生成器"""
    print("=" * 60)
    print("测试 1: 标签生成器 (generate_deletion_label)")
    print("=" * 60)

    test_cases = [
        # (T_A, T_GT, expected_label, description)
        ("地球科学", "地球科学", 0, "完全匹配"),
        ("球科学", "地球科学", 1, "左边界漏字"),
        ("地球科", "地球科学", 1, "右边界漏字"),
        ("地科学", "地球科学", 0, "中间漏字（非边界）"),
        ("地球", "地球科学研究", 1, "右边界多字漏失"),
        ("", "地球", 1, "空文本"),
    ]

    passed = 0
    for T_A, T_GT, expected, desc in test_cases:
        result = generate_deletion_label(T_A, T_GT, K=2)
        status = "✓" if result == expected else "✗"
        if result == expected:
            passed += 1
        print(f"{status} {desc:20s} | T_A='{T_A}' T_GT='{T_GT}' | y={result} (期望={expected})")

    print(f"\n通过率: {passed}/{len(test_cases)}\n")


def test_calibrated_scorer():
    """测试校准评分器"""
    print("=" * 60)
    print("测试 2: 校准评分器 (CalibratedScorer)")
    print("=" * 60)

    # 创建配置
    config = CalibratedScorerConfig(
        enabled=True,
        weights={
            "v_edge": 0.5,
            "b_edge": 0.3,
            "v_edge_x_b_edge": 0.8,
            "drop": 0.2,
        },
        bias=-0.5,
    )

    scorer = CalibratedScorer(config)

    test_cases = [
        # (v_edge, b_edge, drop, description)
        (0.8, 0.9, 0.7, "高风险样本"),
        (0.1, 0.2, 0.1, "低风险样本"),
        (0.5, 0.5, 0.5, "中等风险样本"),
        (0.0, 0.0, 0.0, "零特征样本"),
    ]

    for v_edge, b_edge, drop, desc in test_cases:
        result = scorer.compute_score(v_edge, b_edge, drop)
        print(f"\n{desc}:")
        print(f"  输入: v_edge={v_edge:.2f}, b_edge={b_edge:.2f}, drop={drop:.2f}")
        print(f"  输出: s_b={result['s_b']:.4f}, logit={result['logit']:.4f}")

    print()


def test_backfill_controller():
    """测试严格回填控制器"""
    print("=" * 60)
    print("测试 3: 严格回填控制器 (StrictBackfillController)")
    print("=" * 60)

    config = BackfillConfig(
        strict_mode=True,
        max_edit_distance=2,
        max_length_change_ratio=0.2,
        boundary_K=2,
    )

    controller = StrictBackfillController(config)

    # 测试用例
    test_cases = [
        # (T_A, T_cand, route_type, idx_susp, top2_chars, expected_rejection, description)
        (
            "地球科学",
            "地球科学",
            RouteType.BOUNDARY,
            None,
            None,
            False,
            "无修改（应接受）",
        ),
        (
            "地球科学",
            "地球科学研究",
            RouteType.BOUNDARY,
            None,
            None,
            False,
            "BOUNDARY: 右边界插入（应接受）",
        ),
        (
            "地球科学",
            "地质科学",
            RouteType.BOUNDARY,
            None,
            None,
            True,
            "BOUNDARY: 中间修改（应拒绝）",
        ),
        (
            "地球科学",
            "地质科学",
            RouteType.AMBIGUITY,
            1,
            ["质", "球"],
            False,
            "AMBIGUITY: 单点替换且在Top-2内（应接受）",
        ),
        (
            "地球科学",
            "地质科学",
            RouteType.AMBIGUITY,
            1,
            ["壳", "核"],
            True,
            "AMBIGUITY: 单点替换但不在Top-2内（应拒绝）",
        ),
        (
            "地球科学",
            "地质科技",
            RouteType.AMBIGUITY,
            1,
            ["质", "球"],
            True,
            "AMBIGUITY: 多点修改（应拒绝）",
        ),
        (
            "地球科学",
            "地球科学研究院实验室",
            RouteType.BOUNDARY,
            None,
            None,
            True,
            "全局拒改: 长度变化过大（应拒绝）",
        ),
        (
            "地球科学",
            "完全不同的文本",
            RouteType.BOUNDARY,
            None,
            None,
            True,
            "全局拒改: ED过大（应拒绝）",
        ),
    ]

    passed = 0
    for T_A, T_cand, route_type, idx_susp, top2_chars, expected_rejection, desc in test_cases:
        result = controller.apply_backfill(T_A, T_cand, route_type, idx_susp, top2_chars)
        status = "✓" if result.is_rejected == expected_rejection else "✗"
        if result.is_rejected == expected_rejection:
            passed += 1

        print(f"\n{status} {desc}")
        print(f"  T_A: '{T_A}'")
        print(f"  T_cand: '{T_cand}'")
        print(f"  路由类型: {route_type.value}")
        print(f"  结果: {'拒绝' if result.is_rejected else '接受'} ({result.rejection_reason.value})")
        print(f"  ED: {result.edit_distance}, 长度变化: {result.length_change_ratio:.2%}")

    print(f"\n通过率: {passed}/{len(test_cases)}\n")


def test_integration():
    """集成测试：完整流程"""
    print("=" * 60)
    print("测试 4: 集成测试 (完整流程)")
    print("=" * 60)

    # 模拟完整流程
    print("\n场景: 边界漏字样本的完整处理流程\n")

    # 1. 标签生成
    T_A = "球科学"
    T_GT = "地球科学"
    y_deletion = generate_deletion_label(T_A, T_GT, K=2)
    print(f"[1] 标签生成:")
    print(f"    T_A: '{T_A}', T_GT: '{T_GT}'")
    print(f"    y_deletion: {y_deletion} (1=边界漏字)")

    # 2. 特征提取与评分（模拟）
    v_edge = 0.85  # 高边缘强度
    b_edge = 0.90  # 高 blank 强度
    drop = 0.60  # 中等置信度陡降

    config = CalibratedScorerConfig(
        enabled=True,
        weights={
            "v_edge": 0.5,
            "b_edge": 0.3,
            "v_edge_x_b_edge": 0.8,
            "drop": 0.2,
        },
        bias=-0.5,
    )
    scorer = CalibratedScorer(config)
    score_result = scorer.compute_score(v_edge, b_edge, drop)

    print(f"\n[2] 风险评分:")
    print(f"    v_edge={v_edge:.2f}, b_edge={b_edge:.2f}, drop={drop:.2f}")
    print(f"    s_b={score_result['s_b']:.4f} (校准后评分)")

    # 3. 路由决策（模拟）
    lambda_threshold = 0.5
    upgrade = score_result["s_b"] >= lambda_threshold
    route_type = RouteType.BOUNDARY if upgrade else RouteType.NONE

    print(f"\n[3] 路由决策:")
    print(f"    λ={lambda_threshold:.2f}")
    print(f"    upgrade={upgrade}, route_type={route_type.value}")

    # 4. VLM 修正（模拟）
    T_cand = "地球科学"  # VLM 修正结果

    print(f"\n[4] VLM 修正:")
    print(f"    T_cand: '{T_cand}'")

    # 5. 严格回填
    backfill_config = BackfillConfig(strict_mode=True)
    controller = StrictBackfillController(backfill_config)
    backfill_result = controller.apply_backfill(T_A, T_cand, route_type)

    print(f"\n[5] 严格回填:")
    print(f"    T_final: '{backfill_result.T_final}'")
    print(f"    状态: {'拒绝' if backfill_result.is_rejected else '接受'}")
    print(f"    原因: {backfill_result.rejection_reason.value}")

    # 6. 验证最终结果
    print(f"\n[6] 最终验证:")
    print(f"    T_GT: '{T_GT}'")
    print(f"    T_final: '{backfill_result.T_final}'")
    print(f"    匹配: {'✓' if backfill_result.T_final == T_GT else '✗'}")

    print()


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("SH-DA++ Stage 2: Module Testing")
    print("=" * 60 + "\n")

    test_label_generator()
    test_calibrated_scorer()
    test_backfill_controller()
    test_integration()

    print("=" * 60)
    print("所有测试完成！")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
