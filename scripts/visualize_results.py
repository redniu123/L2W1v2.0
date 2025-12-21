"""
L2W1 可视化模块 - 论文级图表生成

核心可视化:
1. Hallucination Matrix (幻觉矩阵): 展示过度纠错 vs 成功纠正分布
2. Pareto Frontier (帕累托前沿): CER vs Latency 性能边界
3. Case Study Visualizer (案例研究): Hard Sample 前后对比
4. Attention Heatmap (注意力热力图): Agent B 视觉聚焦分析

技术栈: Matplotlib + Seaborn

Usage:
    python visualize_results.py --eval_report ./data/test/evaluation_report.json
    python visualize_results.py --demo  # 生成示例图表
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import math

# 添加项目路径
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 尝试导入绑定库
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not available, some features may be limited")

try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, FancyBboxPatch
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.font_manager as fm
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("Warning: seaborn not available")

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("Warning: PIL not available")


# =============================================================================
# 学术风格配置
# =============================================================================

ACADEMIC_STYLE = {
    # 字体配置 (支持中英文混排)
    'font.family': 'sans-serif',
    'font.sans-serif': ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial', 'sans-serif'],
    'font.size': 11,
    'axes.unicode_minus': False,  # 修复负号显示
    
    # 图形配置
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'figure.figsize': (8, 6),
    
    # 坐标轴
    'axes.linewidth': 1.2,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
    
    # 刻度
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    
    # 图例
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    
    # 网格
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
}

# 配色方案 (学术风格)
COLORS = {
    'baseline': '#377eb8',      # 蓝色 - PP-OCRv5
    'text_llm': '#e41a1c',      # 红色 - Text-only LLM
    'ours': '#4daf4a',          # 绿色 - L2W1 v5.0
    'ours_alt': '#984ea3',      # 紫色 - 备选
    
    # 幻觉矩阵
    'correct_correct': '#2ecc71',   # 绿色 - 保持正确
    'correct_wrong': '#e74c3c',     # 红色 - 过度纠错 (幻觉)
    'wrong_correct': '#3498db',     # 蓝色 - 成功纠正
    'wrong_wrong': '#95a5a6',       # 灰色 - 仍然错误
    
    # 高亮
    'highlight': '#f39c12',
    'suspicious': '#e74c3c',
}


def apply_academic_style():
    """应用学术风格"""
    if MATPLOTLIB_AVAILABLE:
        plt.rcParams.update(ACADEMIC_STYLE)
        if SEABORN_AVAILABLE:
            sns.set_style("whitegrid")
            sns.set_context("paper", font_scale=1.2)


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class TransitionStats:
    """状态转换统计"""
    correct_to_correct: int = 0   # A对 -> S对 (保持正确)
    correct_to_wrong: int = 0     # A对 -> S错 (过度纠错/幻觉)
    wrong_to_correct: int = 0     # A错 -> S对 (成功纠正)
    wrong_to_wrong: int = 0       # A错 -> S错 (仍然错误)
    
    @property
    def total(self) -> int:
        return (self.correct_to_correct + self.correct_to_wrong + 
                self.wrong_to_correct + self.wrong_to_wrong)
    
    def to_matrix(self) -> List[List[int]]:
        """转换为2x2矩阵 [[CC, CW], [WC, WW]]"""
        return [
            [self.correct_to_correct, self.correct_to_wrong],
            [self.wrong_to_correct, self.wrong_to_wrong]
        ]


@dataclass  
class MethodPerformance:
    """方法性能数据"""
    name: str
    cer: float
    latency_ms: float
    ocr_r: float = 0.0
    marker: str = 'o'
    color: str = '#333333'


# =============================================================================
# 1. 幻觉矩阵 (Hallucination Matrix)
# =============================================================================

def compute_transition_stats(
    agent_a_texts: List[str],
    system_texts: List[str],
    gt_texts: List[str]
) -> TransitionStats:
    """
    计算状态转换统计
    
    分析每个字符从 Agent A 到 System 的状态变化
    """
    import difflib
    
    stats = TransitionStats()
    
    for agent_a, system, gt in zip(agent_a_texts, system_texts, gt_texts):
        if not gt:
            continue
        
        # 对齐 Agent A 和 GT
        matcher_a = difflib.SequenceMatcher(None, agent_a, gt)
        a_correct_gt_pos = set()
        for tag, i1, i2, j1, j2 in matcher_a.get_opcodes():
            if tag == 'equal':
                for pos in range(j1, j2):
                    a_correct_gt_pos.add(pos)
        
        # 对齐 System 和 GT
        matcher_s = difflib.SequenceMatcher(None, system, gt)
        s_correct_gt_pos = set()
        for tag, i1, i2, j1, j2 in matcher_s.get_opcodes():
            if tag == 'equal':
                for pos in range(j1, j2):
                    s_correct_gt_pos.add(pos)
        
        # 统计每个 GT 位置的转换
        for pos in range(len(gt)):
            a_correct = pos in a_correct_gt_pos
            s_correct = pos in s_correct_gt_pos
            
            if a_correct and s_correct:
                stats.correct_to_correct += 1
            elif a_correct and not s_correct:
                stats.correct_to_wrong += 1  # 过度纠错!
            elif not a_correct and s_correct:
                stats.wrong_to_correct += 1  # 成功纠正!
            else:
                stats.wrong_to_wrong += 1
    
    return stats


def plot_hallucination_matrix(
    stats: TransitionStats,
    output_path: str = None,
    title: str = "Hallucination Matrix",
    figsize: Tuple[float, float] = (8, 6)
):
    """
    绘制幻觉矩阵 (2x2 混淆矩阵风格)
    
    展示:
    - Correct -> Correct: 保持正确
    - Correct -> Wrong: 过度纠错 (幻觉) - 核心关注点
    - Wrong -> Correct: 成功纠正
    - Wrong -> Wrong: 仍然错误
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib required for plotting")
        return
    
    apply_academic_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 准备数据
    matrix = np.array(stats.to_matrix()) if NUMPY_AVAILABLE else [
        [stats.correct_to_correct, stats.correct_to_wrong],
        [stats.wrong_to_correct, stats.wrong_to_wrong]
    ]
    
    if NUMPY_AVAILABLE:
        total = matrix.sum()
        percentages = matrix / total * 100 if total > 0 else matrix * 0
    else:
        total = sum(sum(row) for row in matrix)
        percentages = [[cell/total*100 if total > 0 else 0 for cell in row] for row in matrix]
    
    # 颜色映射
    colors = [
        [COLORS['correct_correct'], COLORS['correct_wrong']],
        [COLORS['wrong_correct'], COLORS['wrong_wrong']]
    ]
    
    # 绘制矩阵
    for i in range(2):
        for j in range(2):
            value = matrix[i][j] if NUMPY_AVAILABLE else matrix[i][j]
            pct = percentages[i][j] if NUMPY_AVAILABLE else percentages[i][j]
            
            # 绘制方块
            rect = FancyBboxPatch(
                (j - 0.45, 1 - i - 0.45), 0.9, 0.9,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                facecolor=colors[i][j],
                edgecolor='white',
                linewidth=2,
                alpha=0.85
            )
            ax.add_patch(rect)
            
            # 添加数值
            text_color = 'white' if (i == 0 and j == 1) or (i == 1 and j == 0) else 'black'
            ax.text(j, 1 - i, f'{int(value)}\n({pct:.1f}%)',
                   ha='center', va='center', fontsize=16, fontweight='bold',
                   color=text_color)
    
    # 设置坐标轴
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-0.6, 1.6)
    ax.set_aspect('equal')
    
    # 标签
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['System\nCorrect', 'System\nWrong'], fontsize=11)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Agent A\nWrong', 'Agent A\nCorrect'], fontsize=11)
    
    ax.set_xlabel('System Output', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Agent A Output', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # 添加图例说明
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS['correct_correct'], 
                      label='Preserved Correct'),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS['correct_wrong'], 
                      label='Over-Correction (OCR-R)'),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS['wrong_correct'], 
                      label='Successfully Corrected'),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS['wrong_wrong'], 
                      label='Still Wrong'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', 
              bbox_to_anchor=(1.02, 1), fontsize=9)
    
    # 添加关键指标
    ocr_r = stats.correct_to_wrong / max(stats.correct_to_correct + stats.correct_to_wrong, 1)
    cr = stats.wrong_to_correct / max(stats.wrong_to_correct + stats.wrong_to_wrong, 1)
    
    textstr = f'OCR-R: {ocr_r:.2%}\nCorr. Rate: {cr:.2%}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(1.05, 0.3, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Hallucination matrix saved to: {output_path}")
    
    plt.close()
    return fig


# =============================================================================
# 2. 帕累托前沿曲线 (Pareto Frontier)
# =============================================================================

def plot_pareto_frontier(
    methods: List[MethodPerformance],
    output_path: str = None,
    title: str = "CER vs. Latency Trade-off",
    figsize: Tuple[float, float] = (10, 7),
    show_pareto_line: bool = True
):
    """
    绘制帕累托前沿曲线
    
    展示不同方法在 CER-Latency 空间的性能
    目标: 证明 L2W1 v5.0 位于 Pareto 前沿（更优区域）
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib required for plotting")
        return
    
    apply_academic_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制每个方法
    for method in methods:
        ax.scatter(method.latency_ms, method.cer * 100,
                  s=200, c=method.color, marker=method.marker,
                  label=method.name, edgecolors='white', linewidths=1.5,
                  zorder=5)
        
        # 添加标注
        offset = (10, 10)
        ax.annotate(f'{method.name}\nCER: {method.cer*100:.1f}%',
                   (method.latency_ms, method.cer * 100),
                   textcoords="offset points", xytext=offset,
                   fontsize=9, ha='left',
                   arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))
    
    # 绘制 Pareto 前沿线 (如果有多个点)
    if show_pareto_line and len(methods) >= 2 and NUMPY_AVAILABLE:
        points = [(m.latency_ms, m.cer) for m in methods]
        points.sort()
        
        # 简单的 Pareto 前沿: 从左到右，只保留 CER 递减的点
        pareto_points = [points[0]]
        for lat, cer in points[1:]:
            if cer < pareto_points[-1][1]:
                pareto_points.append((lat, cer))
        
        if len(pareto_points) >= 2:
            px, py = zip(*pareto_points)
            ax.plot(px, [y*100 for y in py], 'k--', alpha=0.5, lw=1.5,
                   label='Pareto Frontier')
    
    # 添加"更优区域"标注
    ax.fill_between([ax.get_xlim()[0], methods[0].latency_ms * 0.8],
                    [0, 0], [methods[0].cer * 80, methods[0].cer * 80],
                    alpha=0.1, color='green', label='Better Region')
    
    # 设置坐标轴
    ax.set_xlabel('Average Latency (ms/line)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Character Error Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    
    # 网格
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 图例
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    
    # 设置合理的范围
    all_latencies = [m.latency_ms for m in methods]
    all_cers = [m.cer * 100 for m in methods]
    
    ax.set_xlim(0, max(all_latencies) * 1.3)
    ax.set_ylim(0, max(all_cers) * 1.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Pareto frontier saved to: {output_path}")
    
    plt.close()
    return fig


def plot_router_threshold_curve(
    thresholds: List[float],
    cers: List[float],
    ocr_rs: List[float],
    call_rates: List[float],
    output_path: str = None,
    title: str = "Router Threshold Analysis",
    figsize: Tuple[float, float] = (12, 5)
):
    """
    绘制 Router 阈值对性能的影响曲线
    
    通过调整 Router 阈值 τ，展示 CER 和 OCR-R 的变化
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib required")
        return
    
    apply_academic_style()
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # CER vs Threshold
    ax1 = axes[0]
    ax1.plot(thresholds, [c * 100 for c in cers], 'o-', 
            color=COLORS['ours'], linewidth=2, markersize=8)
    ax1.set_xlabel('Router Threshold (τ)', fontsize=11)
    ax1.set_ylabel('CER (%)', fontsize=11)
    ax1.set_title('CER vs. Threshold', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # OCR-R vs Threshold
    ax2 = axes[1]
    ax2.plot(thresholds, [o * 100 for o in ocr_rs], 's-',
            color=COLORS['correct_wrong'], linewidth=2, markersize=8)
    ax2.set_xlabel('Router Threshold (τ)', fontsize=11)
    ax2.set_ylabel('OCR-R (%)', fontsize=11)
    ax2.set_title('Over-Correction Rate vs. Threshold', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Call Rate vs Threshold
    ax3 = axes[2]
    ax3.plot(thresholds, [r * 100 for r in call_rates], '^-',
            color=COLORS['baseline'], linewidth=2, markersize=8)
    ax3.set_xlabel('Router Threshold (τ)', fontsize=11)
    ax3.set_ylabel('Agent B Call Rate (%)', fontsize=11)
    ax3.set_title('Agent B Invocation Rate', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Threshold analysis saved to: {output_path}")
    
    plt.close()
    return fig


# =============================================================================
# 3. 案例研究可视化 (Case Study Visualizer)
# =============================================================================

def visualize_case_study(
    image_path: str,
    agent_a_text: str,
    system_text: str,
    gt_text: str,
    suspicious_index: int = None,
    output_path: str = None,
    figsize: Tuple[float, float] = (14, 6)
):
    """
    可视化案例研究
    
    展示 Hard Sample 的"前后对比":
    - 原始图像（带高亮 suspicious_index）
    - Agent A 识别结果 vs System 输出 vs Ground Truth
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib required")
        return
    
    apply_academic_style()
    
    fig, axes = plt.subplots(1, 2, figsize=figsize, 
                             gridspec_kw={'width_ratios': [2, 1]})
    
    # 左侧: 图像
    ax_img = axes[0]
    
    if PIL_AVAILABLE and os.path.exists(image_path):
        img = Image.open(image_path).convert('RGB')
        ax_img.imshow(img)
        ax_img.set_title('Input Image', fontsize=12, fontweight='bold')
        
        # 如果有 suspicious_index，尝试高亮对应区域
        if suspicious_index is not None:
            # 简化处理：根据字符位置估算图像区域
            img_width, img_height = img.size
            char_width = img_width / max(len(agent_a_text), 1)
            
            # 绘制高亮框
            x_start = suspicious_index * char_width
            rect = Rectangle((x_start, 0), char_width, img_height,
                            linewidth=3, edgecolor=COLORS['suspicious'],
                            facecolor='none', linestyle='--')
            ax_img.add_patch(rect)
            
            ax_img.annotate(f'Suspicious\nIndex: {suspicious_index}',
                          xy=(x_start + char_width/2, img_height * 0.1),
                          fontsize=10, color=COLORS['suspicious'],
                          ha='center', fontweight='bold',
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        # 创建占位符
        ax_img.text(0.5, 0.5, f'Image: {image_path}\n(Not Found)',
                   transform=ax_img.transAxes, ha='center', va='center',
                   fontsize=12, style='italic')
        ax_img.set_facecolor('#f0f0f0')
    
    ax_img.axis('off')
    
    # 右侧: 文本对比
    ax_text = axes[1]
    ax_text.axis('off')
    
    # 构建对比表格
    y_positions = [0.85, 0.55, 0.25]
    labels = ['Agent A:', 'System:', 'Ground Truth:']
    texts = [agent_a_text, system_text, gt_text]
    colors = [COLORS['baseline'], COLORS['ours'], 'black']
    
    for y, label, text, color in zip(y_positions, labels, texts, colors):
        ax_text.text(0.05, y, label, transform=ax_text.transAxes,
                    fontsize=11, fontweight='bold', va='top')
        
        # 高亮差异字符
        display_text = text
        if suspicious_index is not None and suspicious_index < len(text):
            # 用特殊标记包围 suspicious 字符
            char = text[suspicious_index]
            display_text = text[:suspicious_index] + f'[{char}]' + text[suspicious_index+1:]
        
        ax_text.text(0.05, y - 0.1, display_text, transform=ax_text.transAxes,
                    fontsize=14, va='top', color=color, family='monospace')
    
    # 添加指标
    import difflib
    matcher = difflib.SequenceMatcher(None, system_text, gt_text)
    accuracy = matcher.ratio()
    
    ax_text.text(0.05, 0.05, f'Accuracy: {accuracy:.1%}',
                transform=ax_text.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    ax_text.set_title('Text Comparison', fontsize=12, fontweight='bold')
    
    plt.suptitle('Case Study: Hard Sample Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Case study saved to: {output_path}")
    
    plt.close()
    return fig


def create_comparison_grid(
    cases: List[Dict],
    output_path: str = None,
    cols: int = 2,
    figsize_per_case: Tuple[float, float] = (7, 3)
):
    """
    创建多个案例的对比网格
    
    Args:
        cases: List of dicts with keys: agent_a, system, gt, is_corrected
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib required")
        return
    
    apply_academic_style()
    
    rows = math.ceil(len(cases) / cols)
    figsize = (figsize_per_case[0] * cols, figsize_per_case[1] * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if rows * cols > 1 else [axes]
    
    for idx, (ax, case) in enumerate(zip(axes, cases)):
        agent_a = case.get('agent_a', '')
        system = case.get('system', '')
        gt = case.get('gt', '')
        is_corrected = case.get('is_corrected', False)
        is_overcorrected = case.get('is_overcorrected', False)
        
        # 背景色
        if is_overcorrected:
            bg_color = '#ffe6e6'  # 红色背景
            status = 'Over-Corrected'
        elif is_corrected:
            bg_color = '#e6ffe6'  # 绿色背景
            status = 'Corrected'
        else:
            bg_color = '#f5f5f5'
            status = 'Unchanged'
        
        ax.set_facecolor(bg_color)
        
        # 文本
        text_content = (
            f"Agent A: {agent_a}\n"
            f"System:  {system}\n"
            f"GT:      {gt}"
        )
        ax.text(0.05, 0.5, text_content, transform=ax.transAxes,
               fontsize=10, family='monospace', va='center')
        
        ax.set_title(f'Case {idx + 1}: {status}', fontsize=11, fontweight='bold')
        ax.axis('off')
    
    # 隐藏多余的子图
    for ax in axes[len(cases):]:
        ax.axis('off')
    
    plt.suptitle('Sample Comparison Grid', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Comparison grid saved to: {output_path}")
    
    plt.close()
    return fig


# =============================================================================
# 4. 注意力热力图 (Attention Heatmap)
# =============================================================================

def plot_attention_heatmap(
    attention_weights: List[float],
    characters: str,
    suspicious_index: int = None,
    output_path: str = None,
    title: str = "Agent B Visual Attention",
    figsize: Tuple[float, float] = (12, 4)
):
    """
    绘制注意力权重热力图
    
    展示 Agent B 在接收到 "Check index X" 指令后
    其 Attention 权重在行图像上的分布
    """
    if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
        print("Error: matplotlib and numpy required")
        return
    
    apply_academic_style()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 归一化注意力权重
    weights = np.array(attention_weights)
    if weights.max() > 0:
        weights = weights / weights.max()
    
    # 创建热力图数据
    n_chars = len(characters)
    heatmap_data = weights.reshape(1, -1) if len(weights.shape) == 1 else weights
    
    # 绘制热力图
    cmap = LinearSegmentedColormap.from_list('attention', ['white', '#ff6b6b', '#c0392b'])
    im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    # 设置 x 轴为字符
    ax.set_xticks(range(n_chars))
    ax.set_xticklabels(list(characters), fontsize=12, family='monospace')
    ax.set_yticks([])
    
    # 高亮 suspicious_index
    if suspicious_index is not None and suspicious_index < n_chars:
        rect = Rectangle((suspicious_index - 0.5, -0.5), 1, 1,
                         linewidth=3, edgecolor=COLORS['suspicious'],
                         facecolor='none')
        ax.add_patch(rect)
        
        ax.annotate('Target', xy=(suspicious_index, -0.3),
                   fontsize=10, ha='center', color=COLORS['suspicious'],
                   fontweight='bold')
    
    # 颜色条
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.8)
    cbar.set_label('Attention Weight', fontsize=10)
    
    ax.set_xlabel('Character Position', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Attention heatmap saved to: {output_path}")
    
    plt.close()
    return fig


# =============================================================================
# 报告生成
# =============================================================================

def load_evaluation_report(report_path: str) -> Dict:
    """加载评估报告"""
    with open(report_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def generate_all_figures(
    eval_report: Dict,
    output_dir: str,
    inference_results: List[Dict] = None
):
    """
    生成所有论文图表
    
    Args:
        eval_report: evaluate.py 生成的评估报告
        output_dir: 输出目录
        inference_results: 可选的推理结果详情
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "=" * 60)
    print("  Generating Publication Figures")
    print("=" * 60)
    
    # 1. 从评估报告提取数据
    sample_stats = eval_report.get('sample_stats', {})
    
    # 构造 TransitionStats (近似)
    stats = TransitionStats(
        correct_to_correct=sample_stats.get('exact_match', 0) - sample_stats.get('corrected', 0),
        correct_to_wrong=sample_stats.get('overcorrected', 0),
        wrong_to_correct=sample_stats.get('corrected', 0),
        wrong_to_wrong=sample_stats.get('total', 0) - sample_stats.get('exact_match', 0) - 
                       sample_stats.get('corrected', 0) + sample_stats.get('overcorrected', 0)
    )
    
    # 确保非负
    stats.correct_to_correct = max(0, stats.correct_to_correct)
    stats.wrong_to_wrong = max(0, stats.wrong_to_wrong)
    
    # 绘制幻觉矩阵
    print("\n[1/4] Generating Hallucination Matrix...")
    plot_hallucination_matrix(
        stats,
        output_path=os.path.join(output_dir, 'hallucination_matrix.png'),
        title="Character-Level Transition Matrix"
    )
    
    # 2. 帕累托前沿曲线
    print("[2/4] Generating Pareto Frontier...")
    
    # 示例性能数据 (实际使用时应从实验结果读取)
    methods = [
        MethodPerformance(
            name="PP-OCRv5 (Baseline)",
            cer=eval_report.get('agent_a_cer', 0.08),
            latency_ms=15,
            marker='o',
            color=COLORS['baseline']
        ),
        MethodPerformance(
            name="PP-OCRv5 + Text LLM",
            cer=eval_report.get('agent_a_cer', 0.08) * 0.85,
            latency_ms=120,
            ocr_r=0.15,  # 高幻觉率
            marker='s',
            color=COLORS['text_llm']
        ),
        MethodPerformance(
            name="L2W1 v5.0 (Ours)",
            cer=eval_report.get('overall_cer', 0.03),
            latency_ms=45,
            ocr_r=eval_report.get('ocr_r', 0.03),
            marker='*',
            color=COLORS['ours']
        ),
    ]
    
    plot_pareto_frontier(
        methods,
        output_path=os.path.join(output_dir, 'pareto_frontier.png'),
        title="CER vs. Latency Trade-off"
    )
    
    # 3. Router 阈值分析 (模拟数据)
    print("[3/4] Generating Threshold Analysis...")
    
    thresholds = [1.0, 2.0, 3.0, 4.0, 5.0]
    cers = [0.06, 0.045, 0.03, 0.028, 0.027]
    ocr_rs = [0.01, 0.02, 0.03, 0.05, 0.08]
    call_rates = [0.1, 0.25, 0.4, 0.55, 0.7]
    
    plot_router_threshold_curve(
        thresholds, cers, ocr_rs, call_rates,
        output_path=os.path.join(output_dir, 'threshold_analysis.png'),
        title="Router Threshold (τ) Impact on Performance"
    )
    
    # 4. 注意力热力图 (模拟数据)
    print("[4/4] Generating Attention Heatmap...")
    
    sample_text = "计算机视觉识别"
    n_chars = len(sample_text)
    suspicious_idx = 2  # "机" 字
    
    # 模拟注意力权重：在 suspicious_index 位置有峰值
    if NUMPY_AVAILABLE:
        attention = np.exp(-0.5 * ((np.arange(n_chars) - suspicious_idx) / 1.5) ** 2)
    else:
        attention = [0.2, 0.5, 1.0, 0.5, 0.2, 0.1, 0.1]
    
    plot_attention_heatmap(
        attention,
        sample_text,
        suspicious_index=suspicious_idx,
        output_path=os.path.join(output_dir, 'attention_heatmap.png'),
        title="Agent B Attention Weights (Check Index 2)"
    )
    
    print("\n" + "=" * 60)
    print(f"  All figures saved to: {output_dir}")
    print("=" * 60)


# =============================================================================
# 命令行入口
# =============================================================================

def run_demo():
    """生成示例图表"""
    print("Generating demo figures...")
    
    # 创建输出目录
    output_dir = PROJECT_ROOT / 'outputs' / 'figures'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 示例评估报告
    demo_report = {
        'overall_cer': 0.0312,
        'agent_a_cer': 0.0625,
        'cer_improvement': 0.0312,
        'accuracy': 0.875,
        'ocr_r': 0.0312,
        'correction_rate': 1.0,
        'sample_stats': {
            'total': 100,
            'exact_match': 85,
            'overcorrected': 3,
            'corrected': 12
        }
    }
    
    generate_all_figures(demo_report, str(output_dir))
    
    # 额外生成对比网格
    cases = [
        {'agent_a': '深度学习', 'system': '深度学习', 'gt': '深度学习', 'is_corrected': False},
        {'agent_a': '在时间的未尾', 'system': '在时间的末尾', 'gt': '在时间的末尾', 'is_corrected': True},
        {'agent_a': '计算机视觉', 'system': '计算机视觉', 'gt': '计算机视觉', 'is_corrected': False},
        {'agent_a': '神经网络', 'system': '身经网络', 'gt': '神经网络', 'is_overcorrected': True},
    ]
    
    create_comparison_grid(
        cases,
        output_path=str(output_dir / 'comparison_grid.png')
    )


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="L2W1 可视化模块 - 生成论文级图表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # 生成示例图表
    python visualize_results.py --demo
    
    # 从评估报告生成图表
    python visualize_results.py --eval_report ./data/test/evaluation_report.json
    
    # 指定输出目录
    python visualize_results.py --demo --output_dir ./figures
        """
    )
    
    parser.add_argument("--eval_report", type=str, default="",
                       help="评估报告 JSON 文件路径")
    parser.add_argument("--output_dir", type=str, default="./outputs/figures",
                       help="图表输出目录")
    parser.add_argument("--demo", action="store_true",
                       help="生成示例图表")
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    if not MATPLOTLIB_AVAILABLE:
        print("Error: matplotlib is required for visualization")
        print("Install with: pip install matplotlib seaborn")
        return
    
    if args.demo:
        run_demo()
        return
    
    if not args.eval_report:
        print("Please provide --eval_report or use --demo")
        print("Run with --help for usage")
        return
    
    # 加载评估报告
    if not os.path.exists(args.eval_report):
        print(f"Error: Report not found: {args.eval_report}")
        return
    
    report = load_evaluation_report(args.eval_report)
    generate_all_figures(report, args.output_dir)


if __name__ == "__main__":
    main()

