#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
L2W1 全流程推理与评估脚本 (真实模式)

完整闭环测试流水线：
1. 读取 Baseline 结果 (Agent A 的识别输出)
2. UncertaintyRouter 进行路由决策
3. 对困难样本调用 Agent B (Qwen2.5-VL) 进行边缘扫描纠错
4. 计算 CER 改进和 VLM 调用成本

关键指标:
- CER_improvement = CER_baseline - CER_l2w1
- VLM Call Rate = VLM调用次数 / 总样本数
- 边界修复率 = 成功修复的边界错误数 / 总边界错误数

Usage:
    python scripts/run_l2w1_pipeline.py --limit 200
    python scripts/run_l2w1_pipeline.py --input results/baseline_results.jsonl
    python scripts/run_l2w1_pipeline.py --agent_b_model Qwen/Qwen2.5-VL-3B-Instruct
"""

import json
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from tqdm import tqdm

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ==================== 模块导入 ====================

# Router 模块
try:
    from modules.router.uncertainty_router import (
        UncertaintyRouter,
        RouterConfig,
        RoutingResult,
    )

    HAS_ROUTER = True
except ImportError as e:
    HAS_ROUTER = False
    print(f"[ERROR] Router 模块导入失败: {e}")
    print("[INFO] 请确保 modules/router/uncertainty_router.py 存在")

# Agent B 模块
try:
    from modules.vlm_expert import (
        AgentBExpert,
        AgentBConfig,
        EIPPromptTemplate,
    )

    HAS_AGENT_B = True
except ImportError as e:
    HAS_AGENT_B = False
    print(f"[ERROR] Agent B 模块导入失败: {e}")
    print("[INFO] 请确保已安装: pip install transformers accelerate bitsandbytes")

# V-CoT Prompter 模块
try:
    from modules.vlm_expert.v_cot_prompter import (
        VCoTPrompter,
        VCoTPromptConfig,
        BoundaryRiskType,
    )

    HAS_PROMPTER = True
except ImportError:
    HAS_PROMPTER = False

# 第三方库
try:
    import Levenshtein

    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False
    print("[WARNING] Levenshtein 未安装，将使用内置算法")


# ==================== 数据结构定义 ====================


@dataclass
class SampleResult:
    """单个样本的处理结果"""

    id: str
    image_path: str
    gt_text: str
    agent_a_text: str  # Baseline 预测
    final_text: str  # 最终输出

    # CER 指标
    original_cer: float
    final_cer: float
    cer_improvement: float

    # 路由信息
    is_vlm_used: bool
    routing_decision: str
    boundary_risk: bool
    boundary_reason: str

    # Agent B 信息
    agent_b_text: Optional[str] = None
    agent_b_latency_ms: float = 0.0

    # 置信度信息
    avg_confidence: float = 0.0
    left_boundary_confidence: float = 1.0
    right_boundary_confidence: float = 1.0


@dataclass
class PipelineStats:
    """流水线统计"""

    total_samples: int = 0
    processed_samples: int = 0
    vlm_called_samples: int = 0

    # CER 统计
    total_baseline_cer: float = 0.0
    total_final_cer: float = 0.0
    total_cer_improvement: float = 0.0

    # 边界修复统计
    boundary_risk_samples: int = 0
    boundary_fixed_samples: int = 0

    # 时间统计
    total_time_ms: float = 0.0
    vlm_total_time_ms: float = 0.0

    # 风险等级分布
    risk_distribution: Dict[str, int] = field(
        default_factory=lambda: {"low": 0, "medium": 0, "high": 0, "critical": 0}
    )

    @property
    def vlm_call_rate(self) -> float:
        if self.processed_samples == 0:
            return 0.0
        return self.vlm_called_samples / self.processed_samples

    @property
    def avg_baseline_cer(self) -> float:
        if self.processed_samples == 0:
            return 0.0
        return self.total_baseline_cer / self.processed_samples

    @property
    def avg_final_cer(self) -> float:
        if self.processed_samples == 0:
            return 0.0
        return self.total_final_cer / self.processed_samples

    @property
    def avg_cer_improvement(self) -> float:
        if self.processed_samples == 0:
            return 0.0
        return self.total_cer_improvement / self.processed_samples

    @property
    def boundary_fix_rate(self) -> float:
        if self.boundary_risk_samples == 0:
            return 0.0
        return self.boundary_fixed_samples / self.boundary_risk_samples


# ==================== 辅助函数 ====================


def calculate_cer(gt_text: str, pred_text: str) -> float:
    """计算字符错误率"""
    if len(gt_text) == 0:
        return 0.0 if len(pred_text) == 0 else 1.0

    if HAS_LEVENSHTEIN:
        edit_distance = Levenshtein.distance(gt_text, pred_text)
    else:
        m, n = len(gt_text), len(pred_text)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if gt_text[i - 1] == pred_text[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
        edit_distance = dp[m][n]

    return edit_distance / len(gt_text)


def load_baseline_results(input_path: Path, limit: int = None) -> List[Dict]:
    """加载 Baseline 结果"""
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    samples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            if limit and len(samples) >= limit:
                break

            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"[WARNING] 第 {line_num} 行 JSON 解析失败: {e}")

    print(
        f"[INFO] 加载了 {len(samples)} 个样本" + (f" (限制: {limit})" if limit else "")
    )
    return samples


# ==================== L2W1 流水线 ====================


class L2W1Pipeline:
    """
    L2W1 完整推理流水线 (真实模式)

    核心组件:
    - Router: 不确定性路由器 (边界敏感检测)
    - Agent B: Qwen2.5-VL (4-bit 量化)
    """

    def __init__(
        self,
        router_config: RouterConfig = None,
        agent_b_model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        use_4bit: bool = True,
        use_flash_attention: bool = True,
        image_root: str = "",
    ):
        self.image_root = Path(image_root) if image_root else None

        # 初始化 Router
        self._init_router(router_config)

        # 初始化 Agent B (真实 VLM)
        self._init_agent_b(agent_b_model_path, use_4bit, use_flash_attention)

        # 初始化 Prompter
        self._init_prompter()

        # 统计
        self.stats = PipelineStats()

    def _init_router(self, config: RouterConfig = None):
        """初始化路由器"""
        if HAS_ROUTER:
            self.router = UncertaintyRouter(config=config or RouterConfig())
            print("[INFO] Router 初始化成功")
        else:
            self.router = None
            print("[WARNING] Router 未初始化，将使用简化路由逻辑")

    def _init_agent_b(
        self,
        model_path: str,
        use_4bit: bool,
        use_flash_attention: bool,
    ):
        """初始化 Agent B (Qwen2.5-VL)"""
        if not HAS_AGENT_B:
            raise RuntimeError(
                "Agent B 模块不可用。请安装依赖:\n"
                "pip install transformers accelerate bitsandbytes\n"
                "pip install flash-attn --no-build-isolation"
            )

        print("=" * 60)
        print("初始化 Agent B (Qwen2.5-VL)")
        print("=" * 60)
        print(f"  模型: {model_path}")
        print(f"  4-bit 量化: {use_4bit}")
        print(f"  Flash Attention: {use_flash_attention}")

        config = AgentBConfig(
            model_path=model_path,
            use_4bit=use_4bit,
            use_flash_attention=use_flash_attention,
        )

        self.agent_b = AgentBExpert(config=config, lazy_init=False)
        print("[INFO] Agent B 初始化成功")

    def _init_prompter(self):
        """初始化 V-CoT 提示词生成器"""
        if HAS_PROMPTER:
            self.prompter = VCoTPrompter(VCoTPromptConfig())
            print("[INFO] V-CoT Prompter 初始化成功")
        else:
            self.prompter = None

    def _get_image_path(self, sample: Dict) -> str:
        """解析图像路径"""
        image_path = sample.get("image_path", "")
        if not image_path:
            return ""

        # 检查是否为绝对路径
        if Path(image_path).is_absolute():
            return image_path

        # 尝试相对于 image_root
        if self.image_root:
            full_path = self.image_root / image_path
            if full_path.exists():
                return str(full_path)

        # 尝试相对于项目根目录
        full_path = project_root / image_path
        if full_path.exists():
            return str(full_path)

        # 直接返回原始路径
        return image_path

    def _get_routing_decision(
        self, sample: Dict
    ) -> Tuple[bool, str, bool, str, float, float]:
        """获取路由决策"""
        pred_text = sample.get("pred_text", "")
        avg_conf = sample.get("avg_confidence", 1.0)
        char_confidences = sample.get("char_confidences", [])

        if self.router is None:
            # 简化路由逻辑
            left_conf = 1.0
            right_conf = 1.0
            if char_confidences:
                left_chars = char_confidences[:2]
                right_chars = char_confidences[-2:]
                if left_chars:
                    left_conf = sum(c.get("score", 1.0) for c in left_chars) / len(
                        left_chars
                    )
                if right_chars:
                    right_conf = sum(c.get("score", 1.0) for c in right_chars) / len(
                        right_chars
                    )

            boundary_risk = left_conf < 0.8 or right_conf < 0.8
            boundary_reason = ""
            if left_conf < 0.8:
                boundary_reason += f"左边界置信度低({left_conf:.2f}); "
            if right_conf < 0.8:
                boundary_reason += f"右边界置信度低({right_conf:.2f})"

            if avg_conf < 0.6 or (left_conf < 0.6 and right_conf < 0.6):
                is_hard = True
                risk_level = "high"
            elif avg_conf < 0.75 or boundary_risk:
                is_hard = True
                risk_level = "medium"
            else:
                is_hard = False
                risk_level = "low"

            return (
                is_hard,
                risk_level,
                boundary_risk,
                boundary_reason,
                left_conf,
                right_conf,
            )

        # 使用 Router 的边界敏感检测
        (
            boundary_risk,
            boundary_reason,
            left_conf,
            right_conf,
            aspect_ratio,
            char_density,
        ) = self.router.check_boundary_sensitivity(
            text=pred_text, char_confidences=char_confidences, image_size=None
        )

        if boundary_risk:
            is_hard = True
            risk_level = "high" if avg_conf < 0.7 else "medium"
        elif avg_conf < 0.7:
            is_hard = True
            risk_level = "medium"
        else:
            is_hard = False
            risk_level = "low"

        return (
            is_hard,
            risk_level,
            boundary_risk,
            boundary_reason,
            left_conf,
            right_conf,
        )

    def _build_agent_b_prompt(
        self,
        pred_text: str,
        left_conf: float,
        right_conf: float,
        boundary_risk: bool,
    ) -> str:
        """构建 Agent B 提示词"""
        if self.prompter and boundary_risk:
            # 使用 V-CoT 边界补全提示词
            risk_type = self.prompter.detect_boundary_risk_type(
                left_confidence=left_conf, right_confidence=right_conf
            )
            prompt = self.prompter.build_boundary_completion_prompt(
                pred_text=pred_text,
                risk_type=risk_type,
                left_confidence=left_conf,
                right_confidence=right_conf,
            )
            return prompt
        else:
            # 使用标准 EIP 提示词
            return EIPPromptTemplate.build_prompt(ocr_text=pred_text, risk_level="high")

    def _call_agent_b(
        self,
        image_path: str,
        pred_text: str,
        left_conf: float,
        right_conf: float,
        boundary_risk: bool,
    ) -> Tuple[str, float]:
        """调用 Agent B 进行纠错"""
        start_time = time.time()

        # 构建 Manifest
        manifest = {
            "ocr_text": pred_text,
            "suspicious_index": -1,  # 边界问题，不指定具体位置
            "suspicious_char": "",
            "risk_level": "high",
            "boundary_risk": boundary_risk,
            "left_boundary_confidence": left_conf,
            "right_boundary_confidence": right_conf,
        }

        # 调用 Agent B
        try:
            result = self.agent_b.process_hard_sample(
                image=image_path, manifest=manifest
            )
            corrected_text = result.get("corrected_text", pred_text)
        except Exception as e:
            print(f"[ERROR] Agent B 推理失败: {e}")
            corrected_text = pred_text

        latency_ms = (time.time() - start_time) * 1000
        return corrected_text, latency_ms

    def process_sample(self, sample: Dict) -> SampleResult:
        """处理单个样本"""
        sample_id = sample.get("id", "unknown")
        image_path = self._get_image_path(sample)
        gt_text = sample.get("gt_text", "")
        pred_text = sample.get("pred_text", "")
        avg_conf = sample.get("avg_confidence", 0.0)
        original_cer = sample.get("cer", calculate_cer(gt_text, pred_text))

        # 获取路由决策
        (
            is_hard,
            risk_level,
            boundary_risk,
            boundary_reason,
            left_conf,
            right_conf,
        ) = self._get_routing_decision(sample)

        # 初始化结果
        final_text = pred_text
        agent_b_text = None
        agent_b_latency = 0.0
        is_vlm_used = False

        # 如果是困难样本，调用 Agent B
        if is_hard:
            agent_b_text, agent_b_latency = self._call_agent_b(
                image_path=image_path,
                pred_text=pred_text,
                left_conf=left_conf,
                right_conf=right_conf,
                boundary_risk=boundary_risk,
            )
            final_text = agent_b_text
            is_vlm_used = True

        # 计算最终 CER
        final_cer = calculate_cer(gt_text, final_text)
        cer_improvement = original_cer - final_cer

        return SampleResult(
            id=sample_id,
            image_path=image_path,
            gt_text=gt_text,
            agent_a_text=pred_text,
            final_text=final_text,
            original_cer=original_cer,
            final_cer=final_cer,
            cer_improvement=cer_improvement,
            is_vlm_used=is_vlm_used,
            routing_decision=risk_level,
            boundary_risk=boundary_risk,
            boundary_reason=boundary_reason,
            agent_b_text=agent_b_text,
            agent_b_latency_ms=agent_b_latency,
            avg_confidence=avg_conf,
            left_boundary_confidence=left_conf,
            right_boundary_confidence=right_conf,
        )

    def run(
        self, samples: List[Dict], show_progress: bool = True
    ) -> List[SampleResult]:
        """运行完整流水线"""
        results = []
        self.stats = PipelineStats()
        self.stats.total_samples = len(samples)

        iterator = tqdm(samples, desc="L2W1 Pipeline", miniters=1000) if show_progress else samples

        for sample in iterator:
            try:
                result = self.process_sample(sample)
                results.append(result)

                # 更新统计
                self.stats.processed_samples += 1
                self.stats.total_baseline_cer += result.original_cer
                self.stats.total_final_cer += result.final_cer
                self.stats.total_cer_improvement += result.cer_improvement

                if result.is_vlm_used:
                    self.stats.vlm_called_samples += 1
                    self.stats.vlm_total_time_ms += result.agent_b_latency_ms

                if result.boundary_risk:
                    self.stats.boundary_risk_samples += 1
                    if result.cer_improvement > 0:
                        self.stats.boundary_fixed_samples += 1

                if result.routing_decision in self.stats.risk_distribution:
                    self.stats.risk_distribution[result.routing_decision] += 1

                # 实时更新进度条信息
                if show_progress and isinstance(iterator, tqdm):
                    iterator.set_postfix(
                        {
                            "VLM%": f"{self.stats.vlm_call_rate:.1%}",
                            "CER↓": f"{self.stats.avg_cer_improvement:.2%}",
                        }
                    )

            except Exception as e:
                print(f"\n[ERROR] 处理样本 {sample.get('id', '?')} 失败: {e}")
                continue

        return results


# ==================== 输出函数 ====================


def save_results(results: List[SampleResult], output_path: Path):
    """保存结果"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            result_dict = {
                "id": result.id,
                "image_path": result.image_path,
                "gt_text": result.gt_text,
                "agent_a_text": result.agent_a_text,
                "final_text": result.final_text,
                "original_cer": round(result.original_cer, 4),
                "final_cer": round(result.final_cer, 4),
                "cer_improvement": round(result.cer_improvement, 4),
                "is_vlm_used": result.is_vlm_used,
                "routing_decision": result.routing_decision,
                "boundary_risk": result.boundary_risk,
                "boundary_reason": result.boundary_reason,
                "agent_b_text": result.agent_b_text,
                "agent_b_latency_ms": round(result.agent_b_latency_ms, 2),
                "avg_confidence": round(result.avg_confidence, 4),
                "left_boundary_confidence": round(result.left_boundary_confidence, 4),
                "right_boundary_confidence": round(result.right_boundary_confidence, 4),
            }
            f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")

    print(f"[INFO] 结果已保存: {output_path}")


def save_stats_report(stats: PipelineStats, output_path: Path):
    """保存统计报告为 JSON"""
    report = {
        "summary": {
            "total_samples": stats.total_samples,
            "processed_samples": stats.processed_samples,
            "vlm_called_samples": stats.vlm_called_samples,
            "vlm_call_rate": round(stats.vlm_call_rate, 4),
        },
        "cer_metrics": {
            "baseline_avg_cer": round(stats.avg_baseline_cer, 4),
            "l2w1_final_avg_cer": round(stats.avg_final_cer, 4),
            "avg_cer_improvement": round(stats.avg_cer_improvement, 4),
            "relative_improvement_percent": round(
                stats.avg_cer_improvement / max(stats.avg_baseline_cer, 0.001) * 100, 2
            ),
        },
        "boundary_analysis": {
            "boundary_risk_samples": stats.boundary_risk_samples,
            "boundary_fixed_samples": stats.boundary_fixed_samples,
            "boundary_fix_rate": round(stats.boundary_fix_rate, 4),
        },
        "latency": {
            "total_time_ms": round(stats.total_time_ms, 2),
            "vlm_total_time_ms": round(stats.vlm_total_time_ms, 2),
            "avg_vlm_latency_ms": round(
                stats.vlm_total_time_ms / max(stats.vlm_called_samples, 1), 2
            ),
        },
        "risk_distribution": stats.risk_distribution,
    }

    report_path = output_path.parent / "l2w1_pipeline_stats.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 统计报告已保存: {report_path}")


def print_summary(stats: PipelineStats):
    """打印汇总报告"""
    print("\n" + "=" * 70)
    print("L2W1 流水线执行报告")
    print("=" * 70)

    print(f"\n📊 基础统计:")
    print(f"   总样本数: {stats.total_samples}")
    print(f"   处理样本: {stats.processed_samples}")

    print(f"\n🎯 CER 指标:")
    print(f"   Baseline 平均 CER: {stats.avg_baseline_cer:.2%}")
    print(f"   L2W1 最终平均 CER: {stats.avg_final_cer:.2%}")
    print(f"   平均 CER 改进: {stats.avg_cer_improvement:.2%}")
    rel_improve = stats.avg_cer_improvement / max(stats.avg_baseline_cer, 0.001) * 100
    print(f"   相对改进率: {rel_improve:.1f}%")

    print(f"\n🤖 VLM 调用统计:")
    print(f"   VLM 调用次数: {stats.vlm_called_samples}")
    print(f"   VLM 调用率: {stats.vlm_call_rate:.1%}")
    print(f"   VLM 总耗时: {stats.vlm_total_time_ms / 1000:.1f}s")
    if stats.vlm_called_samples > 0:
        avg_latency = stats.vlm_total_time_ms / stats.vlm_called_samples
        print(f"   VLM 平均延迟: {avg_latency:.0f}ms")

    print(f"\n🔧 边界修复统计:")
    print(f"   边界风险样本: {stats.boundary_risk_samples}")
    print(f"   边界修复成功: {stats.boundary_fixed_samples}")
    print(f"   边界修复率: {stats.boundary_fix_rate:.1%}")

    print(f"\n📈 风险等级分布:")
    for level, count in stats.risk_distribution.items():
        ratio = count / max(stats.processed_samples, 1)
        print(f"   {level.upper()}: {count} ({ratio:.1%})")

    # 论文可用公式
    print("\n" + "=" * 70)
    print("📝 论文可用数据:")
    print("=" * 70)
    print(
        f"""
公式验证:
  CER_baseline = {stats.avg_baseline_cer:.4f}
  CER_l2w1 = {stats.avg_final_cer:.4f}
  CER_improvement = CER_baseline - CER_l2w1 = {stats.avg_cer_improvement:.4f}

关键指标:
  • L2W1 将平均 CER 从 {stats.avg_baseline_cer:.2%} 降低到 {stats.avg_final_cer:.2%}
  • 相对改进 {rel_improve:.1f}%
  • VLM 调用率仅 {stats.vlm_call_rate:.1%}（节省 {(1 - stats.vlm_call_rate) * 100:.1f}% 的 VLM 成本）
  • 边界修复成功率 {stats.boundary_fix_rate:.1%}
"""
    )
    print("=" * 70)


# ==================== 主函数 ====================


def main():
    parser = argparse.ArgumentParser(
        description="L2W1 全流程推理与评估 (真实模式)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/baseline_results.jsonl",
        help="输入 Baseline 结果文件",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/l2w1_final_results.jsonl",
        help="输出结果文件",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="",
        help="图像根目录 (用于解析相对路径)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制处理样本数（用于快速测试）",
    )
    parser.add_argument(
        "--agent_b_model",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Agent B 模型路径或 HuggingFace 模型 ID",
    )
    parser.add_argument(
        "--no_4bit",
        action="store_true",
        help="禁用 4-bit 量化（需要更多显存）",
    )
    parser.add_argument(
        "--no_flash_attention",
        action="store_true",
        help="禁用 Flash Attention",
    )
    parser.add_argument(
        "--boundary_threshold",
        type=float,
        default=0.8,
        help="边界置信度阈值",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("L2W1 全流程推理与评估脚本 (真实模式)")
    print("=" * 70)
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"样本限制: {args.limit if args.limit else '无'}")
    print(f"Agent B 模型: {args.agent_b_model}")
    print(f"4-bit 量化: {not args.no_4bit}")
    print(f"Flash Attention: {not args.no_flash_attention}")
    print()

    # 检查必要模块
    if not HAS_AGENT_B:
        print("\n[ERROR] Agent B 模块不可用，无法运行真实模式")
        print("请安装以下依赖:")
        print("  pip install transformers>=4.37.0 accelerate bitsandbytes")
        print("  pip install flash-attn --no-build-isolation")
        sys.exit(1)

    # 加载数据
    input_path = Path(args.input)
    samples = load_baseline_results(input_path, args.limit)

    if not samples:
        print("[ERROR] 未加载到有效样本")
        sys.exit(1)

    # 配置路由器
    router_config = RouterConfig(boundary_confidence_threshold=args.boundary_threshold)

    # 初始化流水线
    print("\n[INFO] 初始化 L2W1 流水线...")
    pipeline = L2W1Pipeline(
        router_config=router_config,
        agent_b_model_path=args.agent_b_model,
        use_4bit=not args.no_4bit,
        use_flash_attention=not args.no_flash_attention,
        image_root=args.image_root,
    )

    # 运行流水线
    print("\n[INFO] 开始 L2W1 流水线处理...")
    start_time = time.time()

    results = pipeline.run(samples)

    total_time = time.time() - start_time
    pipeline.stats.total_time_ms = total_time * 1000

    # 保存结果
    output_path = Path(args.output)
    save_results(results, output_path)
    save_stats_report(pipeline.stats, output_path)

    # 打印汇总
    print_summary(pipeline.stats)

    print(f"\n[INFO] 总耗时: {total_time:.1f}s")
    print(f"[INFO] 完成! 请查看 {args.output} 获取详细结果")


if __name__ == "__main__":
    main()
