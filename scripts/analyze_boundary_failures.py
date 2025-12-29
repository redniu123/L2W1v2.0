#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
L2W1 边界失败深度分析脚本

对 PP-OCRv5 在 HWDB 数据集上的"边界感知坍塌"现象进行量化分析。

分析维度:
1. 错误位置归一化分布 (0=左边界, 1=右边界)
2. 错误类型分类 (Deletion/Substitution/Insertion)
3. 分段 CER 对比 (Boundary_Left / Boundary_Right / Mid_Section)
4. 论文级统计指标

输出:
- results/boundary_analysis_report.json: 详细分析报告
- results/error_heatmap_data.csv: 热图数据
- results/boundary_analysis.png: 可视化图表 (如果 matplotlib 可用)

Usage:
    python scripts/analyze_boundary_failures.py
    python scripts/analyze_boundary_failures.py --input results/baseline_results.jsonl
"""

import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import sys

# 第三方库
try:
    import Levenshtein
    from Levenshtein import editops

    HAS_LEVENSHTEIN = True
except ImportError:
    HAS_LEVENSHTEIN = False
    print("[WARNING] Levenshtein 库未安装，使用内置实现")

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("[WARNING] numpy 未安装，部分统计功能受限")

try:
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.use("Agg")  # 非交互式后端
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARNING] matplotlib 未安装，跳过可视化")


# ==================== 数据结构定义 ====================


@dataclass
class ErrorInstance:
    """单个错误实例"""

    sample_id: str
    error_type: str  # "delete", "insert", "replace"
    position: int  # 在 GT 文本中的位置
    normalized_position: float  # 归一化位置 [0, 1]
    gt_char: str  # GT 字符
    pred_char: str  # 预测字符 (替换/插入) 或空 (删除)
    gt_length: int  # GT 文本总长度


@dataclass
class SegmentStats:
    """分段统计"""

    total_chars: int = 0
    error_chars: int = 0
    delete_count: int = 0
    insert_count: int = 0
    replace_count: int = 0

    @property
    def cer(self) -> float:
        if self.total_chars == 0:
            return 0.0
        return self.error_chars / self.total_chars


@dataclass
class AnalysisReport:
    """分析报告"""

    # 基础统计
    total_samples: int = 0
    valid_samples: int = 0
    skipped_samples: int = 0

    # 整体指标
    overall_cer: float = 0.0
    overall_avg_confidence: float = 0.0

    # 分段 CER
    boundary_left_stats: SegmentStats = field(default_factory=SegmentStats)
    boundary_right_stats: SegmentStats = field(default_factory=SegmentStats)
    mid_section_stats: SegmentStats = field(default_factory=SegmentStats)

    # 边界分析
    boundary_cer: float = 0.0
    mid_cer: float = 0.0
    boundary_to_mid_ratio: float = 0.0

    # 位置分布
    edge_10_percent_error_ratio: float = 0.0  # 边缘 10% 区域的错误占比
    edge_20_percent_error_ratio: float = 0.0  # 边缘 20% 区域的错误占比

    # 错误类型分布
    total_errors: int = 0
    delete_errors: int = 0
    insert_errors: int = 0
    replace_errors: int = 0

    # 论文级结论
    hypothesis_confirmed: bool = False  # CER_boundary > 3 × CER_mid
    boundary_crisis_severity: str = ""  # 严重程度评估


# ==================== 核心分析逻辑 ====================


def simple_editops(s1: str, s2: str) -> List[Tuple[str, int, int]]:
    """
    简单的 edit operations 实现 (当 Levenshtein 不可用时)
    返回: [(operation, pos_s1, pos_s2), ...]
    """
    m, n = len(s1), len(s2)

    # 动态规划表
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # delete
                    dp[i][j - 1] + 1,  # insert
                    dp[i - 1][j - 1] + 1,  # replace
                )

    # 回溯获取操作序列
    ops = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i - 1] == s2[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            ops.append(("replace", i - 1, j - 1))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("delete", i - 1, j))
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(("insert", i, j - 1))
            j -= 1
        else:
            break

    return list(reversed(ops))


def get_editops(gt_text: str, pred_text: str) -> List[Tuple[str, int, int]]:
    """获取编辑操作列表"""
    if HAS_LEVENSHTEIN:
        return list(editops(gt_text, pred_text))
    else:
        return simple_editops(gt_text, pred_text)


def analyze_single_sample(
    sample: Dict, boundary_window: int = 2
) -> Tuple[List[ErrorInstance], Dict]:
    """
    分析单个样本的错误分布

    Args:
        sample: 样本数据 (包含 gt_text, pred_text, char_confidences 等)
        boundary_window: 边界窗口大小 (首尾各多少个字符)

    Returns:
        Tuple[错误实例列表, 分段统计字典]
    """
    sample_id = sample.get("id", "unknown")
    gt_text = sample.get("gt_text", "")
    pred_text = sample.get("pred_text", "")

    # 边界条件检查
    if len(gt_text) == 0:
        return [], {}

    # 获取编辑操作
    ops = get_editops(gt_text, pred_text)

    errors = []
    gt_len = len(gt_text)

    for op_type, gt_pos, pred_pos in ops:
        # 归一化位置计算
        if gt_len > 1:
            normalized_pos = gt_pos / (gt_len - 1) if gt_pos < gt_len else 1.0
        else:
            normalized_pos = 0.5  # 单字符情况

        # 获取相关字符
        gt_char = gt_text[gt_pos] if gt_pos < len(gt_text) else ""
        pred_char = pred_text[pred_pos] if pred_pos < len(pred_text) else ""

        error = ErrorInstance(
            sample_id=sample_id,
            error_type=op_type,
            position=gt_pos,
            normalized_position=normalized_pos,
            gt_char=gt_char,
            pred_char=pred_char,
            gt_length=gt_len,
        )
        errors.append(error)

    # 分段统计
    segment_stats = {
        "left": SegmentStats(total_chars=min(boundary_window, gt_len)),
        "right": SegmentStats(
            total_chars=min(boundary_window, max(0, gt_len - boundary_window))
        ),
        "mid": SegmentStats(total_chars=max(0, gt_len - 2 * boundary_window)),
    }

    # 根据位置分配错误
    for error in errors:
        pos = error.position

        # 确定所属分段
        if pos < boundary_window:
            segment = "left"
        elif pos >= gt_len - boundary_window:
            segment = "right"
        else:
            segment = "mid"

        # 更新统计
        segment_stats[segment].error_chars += 1

        if error.error_type == "delete":
            segment_stats[segment].delete_count += 1
        elif error.error_type == "insert":
            segment_stats[segment].insert_count += 1
        elif error.error_type == "replace":
            segment_stats[segment].replace_count += 1

    return errors, segment_stats


def load_baseline_results(input_path: Path) -> List[Dict]:
    """加载 baseline 结果"""
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    samples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"[WARNING] 第 {line_num} 行 JSON 解析失败: {e}")

    print(f"[INFO] 加载了 {len(samples)} 个样本")
    return samples


def run_analysis(
    samples: List[Dict], boundary_window: int = 2
) -> Tuple[AnalysisReport, List[ErrorInstance]]:
    """
    执行完整分析

    Args:
        samples: 样本列表
        boundary_window: 边界窗口大小

    Returns:
        Tuple[分析报告, 所有错误实例]
    """
    report = AnalysisReport()
    report.total_samples = len(samples)

    all_errors: List[ErrorInstance] = []

    # 聚合分段统计
    agg_left = SegmentStats()
    agg_right = SegmentStats()
    agg_mid = SegmentStats()

    total_cer = 0.0
    total_confidence = 0.0
    valid_count = 0

    for sample in samples:
        gt_text = sample.get("gt_text", "")
        pred_text = sample.get("pred_text", "")

        # 跳过无效样本
        if len(gt_text) < 3:  # 太短无法分段
            report.skipped_samples += 1
            continue

        # 分析单个样本
        errors, segment_stats = analyze_single_sample(sample, boundary_window)
        all_errors.extend(errors)

        # 聚合统计
        if "left" in segment_stats:
            agg_left.total_chars += segment_stats["left"].total_chars
            agg_left.error_chars += segment_stats["left"].error_chars
            agg_left.delete_count += segment_stats["left"].delete_count
            agg_left.insert_count += segment_stats["left"].insert_count
            agg_left.replace_count += segment_stats["left"].replace_count

        if "right" in segment_stats:
            agg_right.total_chars += segment_stats["right"].total_chars
            agg_right.error_chars += segment_stats["right"].error_chars
            agg_right.delete_count += segment_stats["right"].delete_count
            agg_right.insert_count += segment_stats["right"].insert_count
            agg_right.replace_count += segment_stats["right"].replace_count

        if "mid" in segment_stats:
            agg_mid.total_chars += segment_stats["mid"].total_chars
            agg_mid.error_chars += segment_stats["mid"].error_chars
            agg_mid.delete_count += segment_stats["mid"].delete_count
            agg_mid.insert_count += segment_stats["mid"].insert_count
            agg_mid.replace_count += segment_stats["mid"].replace_count

        # 累计指标
        total_cer += sample.get("cer", 0.0)
        total_confidence += sample.get("avg_confidence", 0.0)
        valid_count += 1

    report.valid_samples = valid_count

    # 计算整体指标
    if valid_count > 0:
        report.overall_cer = total_cer / valid_count
        report.overall_avg_confidence = total_confidence / valid_count

    # 分段统计
    report.boundary_left_stats = agg_left
    report.boundary_right_stats = agg_right
    report.mid_section_stats = agg_mid

    # 计算分段 CER
    boundary_chars = agg_left.total_chars + agg_right.total_chars
    boundary_errors = agg_left.error_chars + agg_right.error_chars

    if boundary_chars > 0:
        report.boundary_cer = boundary_errors / boundary_chars
    if agg_mid.total_chars > 0:
        report.mid_cer = agg_mid.error_chars / agg_mid.total_chars

    # 边界与中间的比值
    if report.mid_cer > 0:
        report.boundary_to_mid_ratio = report.boundary_cer / report.mid_cer
    else:
        report.boundary_to_mid_ratio = float("inf") if report.boundary_cer > 0 else 0.0

    # 错误类型统计
    report.total_errors = len(all_errors)
    report.delete_errors = sum(1 for e in all_errors if e.error_type == "delete")
    report.insert_errors = sum(1 for e in all_errors if e.error_type == "insert")
    report.replace_errors = sum(1 for e in all_errors if e.error_type == "replace")

    # 位置分布分析
    if all_errors:
        edge_10_count = sum(
            1
            for e in all_errors
            if e.normalized_position <= 0.1 or e.normalized_position >= 0.9
        )
        edge_20_count = sum(
            1
            for e in all_errors
            if e.normalized_position <= 0.2 or e.normalized_position >= 0.8
        )

        report.edge_10_percent_error_ratio = edge_10_count / len(all_errors)
        report.edge_20_percent_error_ratio = edge_20_count / len(all_errors)

    # 验证假设: CER_boundary > 3 × CER_mid
    report.hypothesis_confirmed = report.boundary_cer > 3 * report.mid_cer

    # 评估严重程度
    if report.boundary_to_mid_ratio >= 5:
        report.boundary_crisis_severity = "CRITICAL"
    elif report.boundary_to_mid_ratio >= 3:
        report.boundary_crisis_severity = "SEVERE"
    elif report.boundary_to_mid_ratio >= 2:
        report.boundary_crisis_severity = "MODERATE"
    elif report.boundary_to_mid_ratio >= 1.5:
        report.boundary_crisis_severity = "MILD"
    else:
        report.boundary_crisis_severity = "NORMAL"

    return report, all_errors


def generate_heatmap_data(errors: List[ErrorInstance], bins: int = 20) -> List[Dict]:
    """
    生成热图数据

    Args:
        errors: 错误实例列表
        bins: 位置区间数量

    Returns:
        热图数据列表
    """
    if not errors:
        return []

    # 按位置区间统计
    bin_counts = defaultdict(lambda: {"total": 0, "delete": 0, "insert": 0, "replace": 0})

    for error in errors:
        bin_idx = int(error.normalized_position * bins)
        bin_idx = min(bin_idx, bins - 1)  # 确保不越界

        bin_counts[bin_idx]["total"] += 1
        bin_counts[bin_idx][error.error_type] += 1

    # 转换为列表
    heatmap_data = []
    for bin_idx in range(bins):
        bin_start = bin_idx / bins
        bin_end = (bin_idx + 1) / bins

        counts = bin_counts[bin_idx]
        heatmap_data.append(
            {
                "bin_index": bin_idx,
                "position_start": round(bin_start, 3),
                "position_end": round(bin_end, 3),
                "position_center": round((bin_start + bin_end) / 2, 3),
                "total_errors": counts["total"],
                "delete_errors": counts["delete"],
                "insert_errors": counts["insert"],
                "replace_errors": counts["replace"],
            }
        )

    return heatmap_data


def save_heatmap_csv(heatmap_data: List[Dict], output_path: Path):
    """保存热图数据为 CSV"""
    if not heatmap_data:
        print("[WARNING] 无热图数据可保存")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=heatmap_data[0].keys())
        writer.writeheader()
        writer.writerows(heatmap_data)

    print(f"[INFO] 热图数据已保存: {output_path}")


def save_report_json(report: AnalysisReport, output_path: Path):
    """保存分析报告为 JSON"""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 转换为字典
    report_dict = {
        "summary": {
            "total_samples": report.total_samples,
            "valid_samples": report.valid_samples,
            "skipped_samples": report.skipped_samples,
            "overall_cer": round(report.overall_cer, 4),
            "overall_avg_confidence": round(report.overall_avg_confidence, 4),
        },
        "segment_analysis": {
            "boundary_left": {
                "total_chars": report.boundary_left_stats.total_chars,
                "error_chars": report.boundary_left_stats.error_chars,
                "cer": round(report.boundary_left_stats.cer, 4),
                "delete_count": report.boundary_left_stats.delete_count,
                "insert_count": report.boundary_left_stats.insert_count,
                "replace_count": report.boundary_left_stats.replace_count,
            },
            "boundary_right": {
                "total_chars": report.boundary_right_stats.total_chars,
                "error_chars": report.boundary_right_stats.error_chars,
                "cer": round(report.boundary_right_stats.cer, 4),
                "delete_count": report.boundary_right_stats.delete_count,
                "insert_count": report.boundary_right_stats.insert_count,
                "replace_count": report.boundary_right_stats.replace_count,
            },
            "mid_section": {
                "total_chars": report.mid_section_stats.total_chars,
                "error_chars": report.mid_section_stats.error_chars,
                "cer": round(report.mid_section_stats.cer, 4),
                "delete_count": report.mid_section_stats.delete_count,
                "insert_count": report.mid_section_stats.insert_count,
                "replace_count": report.mid_section_stats.replace_count,
            },
        },
        "boundary_crisis_metrics": {
            "boundary_cer": round(report.boundary_cer, 4),
            "mid_cer": round(report.mid_cer, 4),
            "boundary_to_mid_ratio": round(report.boundary_to_mid_ratio, 2),
            "edge_10_percent_error_ratio": round(report.edge_10_percent_error_ratio, 4),
            "edge_20_percent_error_ratio": round(report.edge_20_percent_error_ratio, 4),
        },
        "error_type_distribution": {
            "total_errors": report.total_errors,
            "delete_errors": report.delete_errors,
            "delete_ratio": round(report.delete_errors / max(1, report.total_errors), 4),
            "insert_errors": report.insert_errors,
            "insert_ratio": round(report.insert_errors / max(1, report.total_errors), 4),
            "replace_errors": report.replace_errors,
            "replace_ratio": round(report.replace_errors / max(1, report.total_errors), 4),
        },
        "hypothesis_test": {
            "hypothesis": "CER_boundary > 3 × CER_mid",
            "confirmed": report.hypothesis_confirmed,
            "boundary_crisis_severity": report.boundary_crisis_severity,
        },
        "paper_ready_stats": {
            "边界区域CER": f"{report.boundary_cer * 100:.2f}%",
            "中间区域CER": f"{report.mid_cer * 100:.2f}%",
            "边界/中间比值": f"{report.boundary_to_mid_ratio:.2f}x",
            "边缘10%错误占比": f"{report.edge_10_percent_error_ratio * 100:.1f}%",
            "删除错误占比": f"{report.delete_errors / max(1, report.total_errors) * 100:.1f}%",
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)

    print(f"[INFO] 分析报告已保存: {output_path}")


def generate_visualization(
    report: AnalysisReport,
    heatmap_data: List[Dict],
    errors: List[ErrorInstance],
    output_path: Path,
):
    """生成可视化图表"""
    if not HAS_MATPLOTLIB:
        print("[INFO] matplotlib 未安装，跳过可视化")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 分段 CER 对比
    ax1 = axes[0, 0]
    segments = ["Boundary\nLeft", "Mid\nSection", "Boundary\nRight"]
    cers = [
        report.boundary_left_stats.cer,
        report.mid_section_stats.cer,
        report.boundary_right_stats.cer,
    ]
    colors = ["#e74c3c", "#27ae60", "#e74c3c"]
    bars = ax1.bar(segments, cers, color=colors, alpha=0.8, edgecolor="black")
    ax1.set_ylabel("CER", fontsize=12)
    ax1.set_title("Segment-wise CER Comparison", fontsize=14, fontweight="bold")
    ax1.set_ylim([0, max(cers) * 1.3 if cers else 1])

    # 添加数值标注
    for bar, cer in zip(bars, cers):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{cer:.2%}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # 2. 错误位置分布热图
    ax2 = axes[0, 1]
    if heatmap_data:
        positions = [d["position_center"] for d in heatmap_data]
        counts = [d["total_errors"] for d in heatmap_data]
        ax2.bar(
            positions,
            counts,
            width=1 / len(heatmap_data),
            color="#3498db",
            alpha=0.7,
            edgecolor="black",
        )
        ax2.set_xlabel("Normalized Position (0=Left, 1=Right)", fontsize=12)
        ax2.set_ylabel("Error Count", fontsize=12)
        ax2.set_title(
            "Error Distribution by Position\n(Boundary Crisis Visualization)",
            fontsize=14,
            fontweight="bold",
        )
        ax2.axvspan(0, 0.1, alpha=0.3, color="red", label="Edge 10%")
        ax2.axvspan(0.9, 1.0, alpha=0.3, color="red")
        ax2.legend(loc="upper right")

    # 3. 错误类型分布
    ax3 = axes[1, 0]
    error_types = ["Deletion", "Substitution", "Insertion"]
    error_counts = [report.delete_errors, report.replace_errors, report.insert_errors]
    colors = ["#e74c3c", "#f39c12", "#9b59b6"]
    wedges, texts, autotexts = ax3.pie(
        error_counts,
        labels=error_types,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        explode=(0.05, 0, 0),
    )
    ax3.set_title("Error Type Distribution", fontsize=14, fontweight="bold")

    # 4. 论文级统计摘要
    ax4 = axes[1, 1]
    ax4.axis("off")

    summary_text = f"""
    ╔══════════════════════════════════════════════════╗
    ║         BOUNDARY CRISIS ANALYSIS REPORT          ║
    ╠══════════════════════════════════════════════════╣
    ║                                                  ║
    ║  Total Samples: {report.valid_samples:,}                            
    ║  Overall CER: {report.overall_cer:.2%}                           
    ║                                                  ║
    ║  ─────────── SEGMENT CER ───────────            ║
    ║  Boundary (Left + Right): {report.boundary_cer:.2%}              
    ║  Middle Section: {report.mid_cer:.2%}                       
    ║  Boundary/Mid Ratio: {report.boundary_to_mid_ratio:.2f}x                     
    ║                                                  ║
    ║  ─────────── EDGE ANALYSIS ───────────          ║
    ║  Edge 10% Error Ratio: {report.edge_10_percent_error_ratio:.1%}                  
    ║  Edge 20% Error Ratio: {report.edge_20_percent_error_ratio:.1%}                  
    ║                                                  ║
    ║  ─────────── HYPOTHESIS TEST ───────────        ║
    ║  H₀: CER_boundary > 3 × CER_mid                 ║
    ║  Result: {'✓ CONFIRMED' if report.hypothesis_confirmed else '✗ NOT CONFIRMED'}                           
    ║  Severity: {report.boundary_crisis_severity}                             
    ║                                                  ║
    ╚══════════════════════════════════════════════════╝
    """

    ax4.text(
        0.5,
        0.5,
        summary_text,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="center",
        horizontalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"[INFO] 可视化图表已保存: {output_path}")


def print_summary(report: AnalysisReport):
    """打印分析摘要"""
    print("\n" + "=" * 70)
    print("L2W1 边界失败分析报告")
    print("=" * 70)

    print(f"\n📊 基础统计:")
    print(f"   总样本数: {report.total_samples}")
    print(f"   有效样本: {report.valid_samples}")
    print(f"   跳过样本: {report.skipped_samples}")
    print(f"   整体 CER: {report.overall_cer:.2%}")
    print(f"   平均置信度: {report.overall_avg_confidence:.4f}")

    print(f"\n📍 分段 CER 分析:")
    print(f"   边界左侧 (首2字符): {report.boundary_left_stats.cer:.2%}")
    print(f"   中间区域: {report.mid_section_stats.cer:.2%}")
    print(f"   边界右侧 (尾2字符): {report.boundary_right_stats.cer:.2%}")

    print(f"\n🔥 边界危机指标:")
    print(f"   边界区域 CER: {report.boundary_cer:.2%}")
    print(f"   中间区域 CER: {report.mid_cer:.2%}")
    print(f"   边界/中间比值: {report.boundary_to_mid_ratio:.2f}x")
    print(f"   边缘 10% 错误占比: {report.edge_10_percent_error_ratio:.1%}")
    print(f"   边缘 20% 错误占比: {report.edge_20_percent_error_ratio:.1%}")

    print(f"\n❌ 错误类型分布:")
    print(f"   总错误数: {report.total_errors}")
    print(
        f"   删除错误: {report.delete_errors} ({report.delete_errors/max(1,report.total_errors):.1%})"
    )
    print(
        f"   替换错误: {report.replace_errors} ({report.replace_errors/max(1,report.total_errors):.1%})"
    )
    print(
        f"   插入错误: {report.insert_errors} ({report.insert_errors/max(1,report.total_errors):.1%})"
    )

    print(f"\n📋 假设验证:")
    print(f"   假设: CER_boundary > 3 × CER_mid")
    print(f"   结果: {'✓ 假设成立' if report.hypothesis_confirmed else '✗ 假设不成立'}")
    print(f"   危机严重程度: {report.boundary_crisis_severity}")

    # 论文结论
    print(f"\n" + "=" * 70)
    print("📝 论文可引用结论:")
    print("=" * 70)
    if report.hypothesis_confirmed:
        print(
            f"""
我们的实验表明，PP-OCRv5 在 HWDB 数据集上存在显著的"边界感知坍塌"现象：
• 边界区域（首尾各 2 字符）的 CER 为 {report.boundary_cer:.2%}，
  是中间区域 ({report.mid_cer:.2%}) 的 {report.boundary_to_mid_ratio:.1f} 倍。
• 边缘 10% 区域贡献了 {report.edge_10_percent_error_ratio:.0%} 的总错误。
• 删除错误（字符丢失）占比 {report.delete_errors/max(1,report.total_errors):.0%}，
  表明模型倾向于"忽略"边界字符而非"误识别"。
这一发现验证了我们提出的边界敏感路由策略的必要性。
"""
        )
    else:
        print(
            f"""
边界分析结果显示，当前数据集的边界效应不显著：
• 边界区域 CER: {report.boundary_cer:.2%}
• 中间区域 CER: {report.mid_cer:.2%}
• 比值: {report.boundary_to_mid_ratio:.2f}x (未达到 3x 阈值)
建议进一步检查数据质量或调整分析参数。
"""
        )

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="L2W1 边界失败深度分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="results/baseline_results.jsonl",
        help="输入 JSONL 文件路径",
    )
    parser.add_argument(
        "--output_report",
        type=str,
        default="results/boundary_analysis_report.json",
        help="输出报告路径",
    )
    parser.add_argument(
        "--output_heatmap",
        type=str,
        default="results/error_heatmap_data.csv",
        help="输出热图数据路径",
    )
    parser.add_argument(
        "--output_chart",
        type=str,
        default="results/boundary_analysis.png",
        help="输出可视化图表路径",
    )
    parser.add_argument(
        "--boundary_window",
        type=int,
        default=2,
        help="边界窗口大小 (首尾各多少字符)",
    )
    parser.add_argument(
        "--heatmap_bins",
        type=int,
        default=20,
        help="热图位置区间数量",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("L2W1 边界失败深度分析脚本")
    print("=" * 70)
    print(f"输入文件: {args.input}")
    print(f"边界窗口: 首尾各 {args.boundary_window} 字符")
    print()

    # 加载数据
    input_path = Path(args.input)
    samples = load_baseline_results(input_path)

    if not samples:
        print("[ERROR] 未加载到有效样本")
        sys.exit(1)

    # 执行分析
    print("[INFO] 正在分析错误分布...")
    report, errors = run_analysis(samples, args.boundary_window)

    # 生成热图数据
    print("[INFO] 生成热图数据...")
    heatmap_data = generate_heatmap_data(errors, args.heatmap_bins)

    # 保存结果
    save_report_json(report, Path(args.output_report))
    save_heatmap_csv(heatmap_data, Path(args.output_heatmap))

    # 生成可视化
    generate_visualization(report, heatmap_data, errors, Path(args.output_chart))

    # 打印摘要
    print_summary(report)


if __name__ == "__main__":
    main()

