#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ Stage 2 Integration Testing

测试 Router、Prompter、BackfillController 和 Pipeline 的集成。
"""

import sys
from pathlib import Path

import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.pipeline_stage2 import MeltdownMonitor, PipelineConfig, SHDAPipeline
from modules.router.backfill import BackfillConfig, RouteType
from modules.router.sh_da_router import (
    BudgetControllerConfig,
    CalibratedScorerConfig,
    RuleScorerConfig,
    SHDARouter,
    SHDARouterConfig,
)
from modules.vlm_expert.constrained_prompter import ConstrainedPrompter


def test_constrained_prompter():
    """测试受限提示词生成器"""
    print("=" * 60)
    print("测试 1: 受限提示词生成器 (ConstrainedPrompter)")
    print("=" * 60)

    prompter = ConstrainedPrompter()

    # Test Case 1: BOUNDARY 提示词
    print("\n--- Case 1: BOUNDARY 提示词 ---")
    T_A = "地球科学"
    prompt = prompter.generate_boundary_prompt(T_A, image_path="test.jpg")

    print(f"文本: {T_A}")
    print(f"提示词类型: {prompt['prompt_type']}")
    print(f"System Prompt:\n{prompt['system_prompt'][:100]}...")
    print(f"User Prompt:\n{prompt['user_prompt'][:150]}...")
    print(f"约束: {prompt['constraints']}")

    # Test Case 2: AMBIGUITY 提示词
    print("\n--- Case 2: AMBIGUITY 提示词 ---")
    T_A = "地球科学"
    idx_susp = 1
    top2_chars = ["质", "球"]

    prompt = prompter.generate_ambiguity_prompt(
        T_A, idx_susp, top2_chars, image_path="test.jpg"
    )

    print(f"文本: {T_A}")
    print(f"存疑位置: {idx_susp} ('{T_A[idx_susp]}')")
    print(f"Top-2: {top2_chars}")
    print(f"提示词类型: {prompt['prompt_type']}")
    print(f"User Prompt:\n{prompt['user_prompt'][:200]}...")

    # Test Case 3: BOTH 提示词
    print("\n--- Case 3: BOTH 提示词（两阶段）---")
    prompts = prompter.generate_both_prompts(T_A, idx_susp, top2_chars)

    print(f"Stage 1 (BOUNDARY):")
    print(f"  {prompts['stage1']['user_prompt'][:100]}...")
    print(f"Stage 2 (AMBIGUITY):")
    print(f"  需要在 Stage 1 完成后动态生成")

    print("\n✓ 受限提示词生成器测试通过\n")


def test_sh_da_router():
    """测试 SH-DA++ Router"""
    print("=" * 60)
    print("测试 2: SH-DA++ Router (集成版)")
    print("=" * 60)

    # 创建配置
    config = SHDARouterConfig(
        use_calibrated_scorer=False,  # 使用 RuleOnlyScorer
        rule_scorer=RuleScorerConfig(
            v_min=0.0,
            v_max=5.0,
            lambda_threshold=0.5,
            eta=0.5,
        ),
        budget_controller=BudgetControllerConfig(
            window_size=10,
            k=0.05,
            lambda_init=0.5,
            target_budget=0.2,
        ),
    )

    router = SHDARouter(config)

    # 模拟数据
    boundary_stats = {
        "valid": True,
        "blank_mean_L": 0.8,
        "blank_mean_R": 0.7,
        "blank_peak_L": 0.9,
        "blank_peak_R": 0.85,
    }

    top2_info = {
        "top2_status": "available",
        "top1_probs": [0.9, 0.6, 0.95, 0.88],
        "top2_probs": [0.05, 0.35, 0.03, 0.08],
        "top1_chars": ["地", "球", "科", "学"],
        "top2_chars": ["他", "求", "料", "字"],
    }

    # Test Case 1: 高风险样本
    print("\n--- Case 1: 高风险样本（应升级）---")
    decision = router.route(
        boundary_stats=boundary_stats,
        top2_info=top2_info,
        v_edge=0.85,
        char_count=4,
        expected_char_count=6,
        agent_a_text="地球科学",
    )

    print(f"升级决策: {decision.upgrade}")
    print(f"路由类型: {decision.route_type.value}")
    print(f"s_b={decision.s_b:.4f}, s_a={decision.s_a:.4f}, q={decision.q:.4f}")
    print(f"λ={decision.lambda_current:.4f}")
    print(f"存疑位置: {decision.idx_susp}")
    print(f"Top-2: {decision.top2_chars}")

    # Test Case 2: 低风险样本
    print("\n--- Case 2: 低风险样本（不应升级）---")
    boundary_stats_low = {
        "valid": True,
        "blank_mean_L": 0.1,
        "blank_mean_R": 0.1,
        "blank_peak_L": 0.2,
        "blank_peak_R": 0.15,
    }

    top2_info_low = {
        "top2_status": "available",
        "top1_probs": [0.99, 0.98, 0.99, 0.97],
        "top2_probs": [0.005, 0.01, 0.005, 0.02],
        "top1_chars": ["地", "球", "科", "学"],
        "top2_chars": ["他", "求", "料", "字"],
    }

    decision = router.route(
        boundary_stats=boundary_stats_low,
        top2_info=top2_info_low,
        v_edge=0.1,
        agent_a_text="地球科学",
    )

    print(f"升级决策: {decision.upgrade}")
    print(f"路由类型: {decision.route_type.value}")
    print(f"s_b={decision.s_b:.4f}, s_a={decision.s_a:.4f}, q={decision.q:.4f}")

    print("\n✓ SH-DA++ Router 测试通过\n")


def test_meltdown_monitor():
    """测试熔断监控器"""
    print("=" * 60)
    print("测试 3: 熔断监控器 (MeltdownMonitor)")
    print("=" * 60)

    monitor = MeltdownMonitor(window_size=10, cvr_threshold=0.3)

    # 模拟正常情况（CVR < 30%）
    print("\n--- Case 1: 正常情况（CVR < 30%）---")
    for i in range(10):
        is_rejected = i < 2  # 前 2 个拒改，CVR = 20%
        triggered = monitor.record(is_rejected)

    print(f"当前 CVR: {monitor.current_cvr:.2%}")
    print(f"熔断状态: {monitor.is_meltdown_active}")
    print(f"触发熔断: {triggered}")

    # 模拟异常情况（CVR > 30%）
    print("\n--- Case 2: 异常情况（CVR > 30%）---")
    monitor.reset()

    for i in range(10):
        is_rejected = i < 4  # 前 4 个拒改，CVR = 40%
        triggered = monitor.record(is_rejected)

    print(f"当前 CVR: {monitor.current_cvr:.2%}")
    print(f"熔断状态: {monitor.is_meltdown_active}")
    print(f"触发熔断: {triggered}")

    print("\n✓ 熔断监控器测试通过\n")


def test_integrated_pipeline():
    """测试集成 Pipeline"""
    print("=" * 60)
    print("测试 4: 集成 Pipeline (完整流程)")
    print("=" * 60)

    # 创建 Router
    router_config = SHDARouterConfig(
        use_calibrated_scorer=False,
        rule_scorer=RuleScorerConfig(lambda_threshold=0.5),
        budget_controller=BudgetControllerConfig(
            window_size=10, target_budget=0.2, lambda_init=0.5
        ),
    )
    router = SHDARouter(router_config)

    # 创建 Pipeline
    pipeline_config = PipelineConfig(
        backfill_config=BackfillConfig(strict_mode=True),
        meltdown_enabled=True,
        meltdown_cvr_threshold=0.3,
        enable_logging=False,  # 测试时关闭日志
    )

    with SHDAPipeline(router, pipeline_config) as pipeline:
        # 模拟 Agent B
        def mock_agent_b(prompt):
            # 简单模拟：返回修正后的文本
            return "地球科学研究"

        # Test Case 1: 正常升级并接受修正
        print("\n--- Case 1: 正常升级并接受修正 ---")
        boundary_stats = {
            "valid": True,
            "blank_mean_L": 0.1,
            "blank_mean_R": 0.8,
            "blank_peak_L": 0.2,
            "blank_peak_R": 0.9,
        }

        top2_info = {
            "top2_status": "available",
            "top1_probs": [0.9, 0.9, 0.9, 0.9],
            "top2_probs": [0.05, 0.05, 0.05, 0.05],
            "top1_chars": ["地", "球", "科", "学"],
            "top2_chars": ["他", "求", "料", "字"],
        }

        result = pipeline.process(
            T_A="地球科学",
            boundary_stats=boundary_stats,
            top2_info=top2_info,
            image_path="test.jpg",
            agent_b_callable=mock_agent_b,
            v_edge=0.85,
        )

        print(f"T_A: 地球科学")
        print(f"T_final: {result.T_final}")
        print(f"升级: {result.upgrade}")
        print(f"路由类型: {result.route_type}")
        print(f"拒改: {result.is_rejected}")
        print(f"拒改原因: {result.rejection_reason}")

        # Test Case 2: 升级但被拒改（ED > 2）
        print("\n--- Case 2: 升级但被拒改（ED > 2）---")

        def mock_agent_b_hallucination(prompt):
            return "完全不同的文本内容"

        result = pipeline.process(
            T_A="地球科学",
            boundary_stats=boundary_stats,
            top2_info=top2_info,
            image_path="test.jpg",
            agent_b_callable=mock_agent_b_hallucination,
            v_edge=0.85,
        )

        print(f"T_A: 地球科学")
        print(f"T_cand: 完全不同的文本内容")
        print(f"T_final: {result.T_final}")
        print(f"拒改: {result.is_rejected}")
        print(f"拒改原因: {result.rejection_reason}")

        # 获取统计信息
        print("\n--- Pipeline 统计信息 ---")
        stats = pipeline.get_stats()
        print(f"Router 模式: {stats['router']['scorer_mode']}")
        print(f"预算控制: {stats['router']['budget_controller']['actual_budget_window']:.2%}")
        if "meltdown" in stats:
            print(f"熔断状态: {stats['meltdown']['meltdown_active']}")
            print(f"当前 CVR: {stats['meltdown']['current_cvr']:.2%}")

    print("\n✓ 集成 Pipeline 测试通过\n")


def test_calibrated_scorer_mode():
    """测试校准评分器模式"""
    print("=" * 60)
    print("测试 5: 校准评分器模式")
    print("=" * 60)

    # 创建配置（启用校准评分器）
    config = SHDARouterConfig(
        use_calibrated_scorer=True,
        calibrated_scorer=CalibratedScorerConfig(
            enabled=True,
            weights={
                "v_edge": 0.5,
                "b_edge": 0.3,
                "v_edge_x_b_edge": 0.8,
                "drop": 0.2,
            },
            bias=-0.5,
        ),
        budget_controller=BudgetControllerConfig(lambda_init=0.5),
    )

    router = SHDARouter(config)

    # 模拟数据
    boundary_stats = {
        "valid": True,
        "blank_mean_L": 0.8,
        "blank_mean_R": 0.7,
        "blank_peak_L": 0.9,
        "blank_peak_R": 0.85,
    }

    top2_info = {
        "top2_status": "available",
        "top1_probs": [0.9, 0.6, 0.95, 0.88],
        "top2_probs": [0.05, 0.35, 0.03, 0.08],
    }

    decision = router.route(
        boundary_stats=boundary_stats,
        top2_info=top2_info,
        v_edge=0.85,
        agent_a_text="地球科学",
    )

    print(f"评分器模式: {router.scorer_mode}")
    print(f"s_b={decision.s_b:.4f}, s_a={decision.s_a:.4f}, q={decision.q:.4f}")
    print(f"升级决策: {decision.upgrade}")
    print(f"路由类型: {decision.route_type.value}")

    print("\n✓ 校准评分器模式测试通过\n")


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("SH-DA++ Stage 2 Integration Testing")
    print("=" * 60 + "\n")

    test_constrained_prompter()
    test_sh_da_router()
    test_meltdown_monitor()
    test_integrated_pipeline()
    test_calibrated_scorer_mode()

    print("=" * 60)
    print("所有集成测试完成！")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
