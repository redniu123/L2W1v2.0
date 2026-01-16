#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v4.0 预算控制器稳定性测试脚本

功能：
1. 加载配置并初始化 OnlineBudgetController
2. 模拟流式推理，按顺序处理 test_metadata.jsonl
3. 记录每一步的决策、阈值和调用率
4. 验证最终调用率是否满足 |Actual - B| ≤ 0.5%
5. 检查震荡是否超过 B ± 3%
6. 生成可视化图表

Usage:
    python scripts/test_budget_stability.py \
        --config configs/router_config.yaml \
        --metadata ./data/raw/HWDB_Benchmark/test_metadata.jsonl \
        --image_root ./data/raw/HWDB_Benchmark/
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
import cv2

# ========== 中文字体配置 (Linux/Windows 兼容) ==========
def setup_chinese_font():
    """
    配置中文字体，支持 Linux 和 Windows 环境
    优先级：SimHei > WenQuanYi Micro Hei > WenQuanYi Zen Hei > DejaVu Sans
    """
    import matplotlib.font_manager as fm
    
    # 候选中文字体列表（按优先级排序）
    chinese_fonts = [
        'SimHei',                    # Windows 黑体
        'WenQuanYi Micro Hei',       # Linux 常用
        'WenQuanYi Zen Hei',         # Linux 常用
        'Noto Sans CJK SC',          # Google Noto 字体
        'Source Han Sans SC',        # Adobe 思源黑体
        'Microsoft YaHei',           # Windows 微软雅黑
        'AR PL UMing CN',            # Linux 明体
        'DejaVu Sans',               # 回退字体
    ]
    
    # 获取系统可用字体
    available_fonts = set(f.name for f in fm.fontManager.ttflist)
    
    # 查找第一个可用的中文字体
    selected_font = 'DejaVu Sans'
    for font in chinese_fonts:
        if font in available_fonts:
            selected_font = font
            break
    
    # 设置 matplotlib 字体
    plt.rcParams['font.sans-serif'] = [selected_font] + chinese_fonts
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    return selected_font

# 初始化中文字体
_selected_font = setup_chinese_font()

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))


@dataclass
class StepRecord:
    """单步决策记录"""
    sample_idx: int
    q: float
    upgrade: bool
    lambda_before: float
    lambda_after: float
    actual_budget_window: float  # 滑动窗口内的实际调用率
    actual_budget_total: float   # 总体实际调用率
    target_budget: float
    is_warmup: bool
    lambda_updated: bool


class ANSIColors:
    """ANSI 颜色代码"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_color(text: str, color: str = ANSIColors.RESET):
    """打印带颜色的文本"""
    print(f"{color}{text}{ANSIColors.RESET}")


def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    import yaml
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}
    
    return config


def infer_image_root(metadata_path: str) -> str:
    """从 metadata 路径自动推断 image_root"""
    metadata_path_obj = Path(metadata_path)
    return str(metadata_path_obj.parent)


def resolve_image_path(image_path_str: str, image_root: str = None, metadata_dir: str = None) -> Optional[str]:
    """
    智能解析图像路径（复用 calibrate_router.py 的逻辑）
    """
    image_path_str = os.path.normpath(image_path_str)
    candidates = []
    
    if os.path.isabs(image_path_str):
        candidates.append(image_path_str)
    
    if image_root:
        image_root_path = Path(image_root)
        image_path_normalized = image_path_str.replace("\\", "/").strip("./")
        image_root_normalized = str(image_root_path).replace("\\", "/").strip("./")
        
        if not image_path_normalized.startswith(image_root_normalized):
            candidates.append(str(image_root_path / image_path_str))
        else:
            candidates.append(image_path_str)
        
        filename_only = Path(image_path_str).name
        candidates.append(str(image_root_path / filename_only))
        candidates.append(str(image_root_path / "images" / filename_only))
    
    if metadata_dir:
        metadata_dir_path = Path(metadata_dir)
        candidates.append(str(metadata_dir_path / image_path_str))
        candidates.append(str(metadata_dir_path / Path(image_path_str).name))
        candidates.append(str(metadata_dir_path / "images" / Path(image_path_str).name))
    
    candidates.append(image_path_str)
    candidates.append(str(Path(image_path_str).name))
    
    for candidate in candidates:
        candidate_path = Path(candidate)
        if candidate_path.exists() and candidate_path.is_file():
            return str(candidate_path.resolve())
    
    return None


def load_metadata(metadata_path: str, limit: int = None) -> List[Dict]:
    """加载 metadata JSONL 文件"""
    samples = []
    
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"[Warning] 解析失败 (行 {i+1}): {e}")
    
    return samples


def compute_sobel_edge_strength(image: np.ndarray) -> float:
    """计算图像的 Sobel 边缘响应强度"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    
    h, w = gray.shape
    boundary_width = max(1, int(w * 0.1))
    
    left_region = magnitude[:, :boundary_width]
    right_region = magnitude[:, -boundary_width:]
    v_edge = (np.mean(left_region) + np.mean(right_region)) / 2.0
    
    return float(v_edge)


def process_samples_stream(
    samples: List[Dict],
    image_root: str,
    metadata_dir: str,
    config: Dict,
    model_dir: str,
    target_budget: float = None,
    limit: int = None,
    verbose: bool = False,
) -> Tuple[List[StepRecord], Dict]:
    """
    流式处理样本，记录每一步的决策和状态
    
    Args:
        samples: 样本列表
        image_root: 图像根目录
        metadata_dir: metadata 目录
        config: 配置字典
        model_dir: Agent A 模型目录
        limit: 最大样本数
        verbose: 是否打印详细日志
    
    Returns:
        Tuple[records, stats]: 记录列表和统计信息
    """
    from modules.router.uncertainty_router import (
        OnlineBudgetController,
        BudgetControllerConfig,
        RuleOnlyScorer,
        RuleScorerConfig,
    )
    from modules.paddle_engine.predict_rec_modified import TextRecognizerWithLogits
    
    # 1. 初始化配置
    sh_da_config = config.get('sh_da_v4', {})
    
    if not sh_da_config:
        print("[Warning] 配置文件中未找到 'sh_da_v4' 部分，使用默认配置")
    
    # RuleOnlyScorer 配置
    rule_scorer_cfg = sh_da_config.get('rule_scorer', {})
    scorer_config = RuleScorerConfig(
        v_min=rule_scorer_cfg.get('v_min', 0.0),
        v_max=rule_scorer_cfg.get('v_max', 100.0),
        lambda_threshold=rule_scorer_cfg.get('lambda_threshold', 0.5),
        eta=rule_scorer_cfg.get('eta', 0.5),
    )
    scorer = RuleOnlyScorer(config=scorer_config)
    
    # OnlineBudgetController 配置 (SH-DA++ v4.0 稳定性优化)
    # k: 减小至 0.01 以提高稳定性
    # window_size: 增大至 500 以减少噪声
    budget_cfg = sh_da_config.get('budget_controller', {})
    if target_budget is not None:
        budget_cfg = dict(budget_cfg)
        budget_cfg['target_budget'] = target_budget
    budget_config = BudgetControllerConfig(
        window_size=budget_cfg.get('window_size', 500),   # 增大窗口大小
        k=budget_cfg.get('k', 0.01),                      # 减小比例系数
        lambda_min=budget_cfg.get('lambda_min', 0.0),
        lambda_max=budget_cfg.get('lambda_max', 2.0),
        lambda_init=budget_cfg.get('lambda_init', 0.5),
        target_budget=budget_cfg.get('target_budget', 0.2),
    )
    controller = OnlineBudgetController(config=budget_config)
    
    # 添加微小随机扰动以制造决策梯度 (当 q 分布单一时)
    epsilon_range = (1e-6, 1e-5)
    
    # 初始化 Agent A（严格模式，无模拟）
    # 检查模型目录
    model_path = Path(model_dir)
    if not model_path.exists():
        print(f"[FATAL] 模型目录不存在: {model_dir}")
        sys.exit(1)
    
    # 检查模型文件（兼容 PP-OCRv5 新格式）
    has_params = (model_path / "inference.pdiparams").exists() or (model_path / "model.pdiparams").exists()
    has_model = (
        (model_path / "inference.pdmodel").exists() or 
        (model_path / "inference.json").exists() or
        (model_path / "model.pdmodel").exists() or
        (model_path / "model.json").exists()
    )
    
    if not has_params or not has_model:
        print(f"[FATAL] 模型文件缺失: {model_dir}")
        print("  需要 inference.pdiparams 和 (inference.pdmodel 或 inference.json)")
        sys.exit(1)
    
    try:
        import tools.infer.utility as utility
        parser = utility.init_args()
        args = parser.parse_args([])
        
        # 设置模型路径
        args.rec_model_dir = model_dir
        args.rec_batch_num = 1
        args.rec_image_shape = "3, 48, 320"
        args.use_gpu = True
        args.warmup = False
        args.benchmark = False
        args.use_onnx = False
        args.return_word_box = False
        
        # 从配置读取字典路径
        agent_a_cfg = config.get('agent_a', {})
        args.rec_char_dict_path = agent_a_cfg.get('rec_char_dict_path', './ppocr/utils/ppocr_keys_v1.txt')
        
        engine = TextRecognizerWithLogits(args)
        print(f"[✓] Agent A 初始化成功: {model_dir}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[FATAL] Agent A 初始化失败: {e}")
        print("\n请检查:")
        print("  1. 模型文件是否完整")
        print("  2. PaddlePaddle 是否正确安装")
        print("  3. GPU/CUDA 是否可用")
        sys.exit(1)
    
    # 2. 流式处理
    records = []
    failed_count = 0
    success_count = 0
    
    process_count = len(samples) if limit is None else min(limit, len(samples))
    
    print(f"\n开始流式处理 {process_count} 个样本...")
    print(f"  目标调用率 B = {budget_config.target_budget:.1%}")
    print(f"  窗口大小 W = {budget_config.window_size}")
    print(f"  初始阈值 λ₀ = {budget_config.lambda_init:.4f}")
    print(f"  比例系数 k = {budget_config.k}")
    
    start_time = time.time()
    
    for i, sample in enumerate(samples):
        if limit and i >= limit:
            break
        
        try:
            # 读取图像
            image_path_str = sample.get('image_path', sample.get('image', ''))
            if not image_path_str:
                failed_count += 1
                continue
            
            resolved_path = resolve_image_path(
                image_path_str=image_path_str,
                image_root=image_root,
                metadata_dir=metadata_dir,
            )
            
            if resolved_path is None:
                failed_count += 1
                if verbose and i < 5:
                    print(f"[Warning] 无法解析图像路径 (样本 {i+1}): {image_path_str}")
                continue
            
            image = cv2.imread(resolved_path)
            if image is None:
                failed_count += 1
                continue
            
            # 计算 q 分数
            try:
                output = engine([image])
                boundary_stats = output.get('boundary_stats', [{}])[0] if output.get('boundary_stats') else {}
                top2_info = output.get('top2_info', [{}])[0] if output.get('top2_info') else {}
                
                v_edge = compute_sobel_edge_strength(image)
                
                result = scorer.score(
                    boundary_stats=boundary_stats,
                    top2_info=top2_info,
                    r_d=0.0,
                    v_edge=v_edge,
                )
                q = result.q
                
                # 添加微小随机扰动以强行制造决策梯度 (当 q 分布单一时)
                # ε ∈ [10^-6, 10^-5]，足够小不影响主要决策，但能防止完全相同的 q 值
                epsilon = np.random.uniform(epsilon_range[0], epsilon_range[1])
                q = q + epsilon
                
            except Exception as e:
                if verbose and i < 5:
                    print(f"[Warning] 推理失败 (样本 {i+1}): {e}")
                failed_count += 1
                continue
            
            # 使用 OnlineBudgetController 进行决策
            upgrade, details = controller.step(q)
            
            # 记录
            record = StepRecord(
                sample_idx=i,
                q=q,
                upgrade=upgrade,
                lambda_before=details.get('lambda_before', controller.current_lambda),
                lambda_after=details.get('lambda_after', controller.current_lambda),
                actual_budget_window=details.get('actual_budget', 0.0),
                actual_budget_total=controller.total_budget,
                target_budget=budget_config.target_budget,
                is_warmup=details.get('is_warmup', False),
                lambda_updated=details.get('updated', False),
            )
            records.append(record)
            success_count += 1
            
            # 进度输出
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                stats = controller.get_stats()
                print(f"  进度: {i+1}/{process_count} | "
                      f"λ = {controller.current_lambda:.4f} | "
                      f"B̄_window = {stats['actual_budget_window']:.2%} | "
                      f"B̄_total = {stats['total_budget']:.2%} | "
                      f"速度: {rate:.1f} samples/s")
        
        except Exception as e:
            failed_count += 1
            if verbose:
                print(f"[Error] 处理样本 {i+1} 失败: {e}")
    
    elapsed = time.time() - start_time
    
    # 最终统计
    final_stats = controller.get_stats()
    
    # 获取 lambda_history（通过私有属性或 get_stats）
    lambda_history = controller._lambda_history if hasattr(controller, '_lambda_history') else []
    
    stats = {
        'total_samples': process_count,
        'success_samples': success_count,
        'failed_samples': failed_count,
        'final_lambda': controller.current_lambda,
        'lambda_init': budget_config.lambda_init,
        'final_budget_window': final_stats['actual_budget_window'],
        'final_budget_total': final_stats['total_budget'],
        'target_budget': budget_config.target_budget,
        'window_size': budget_config.window_size,
        'total_upgrades': final_stats['total_upgrades'],
        'processing_time': elapsed,
        'lambda_history': lambda_history,
    }
    
    print(f"\n处理完成: 成功 {success_count}, 失败 {failed_count}, 耗时 {elapsed:.2f}s")
    
    return records, stats


def evaluate_stability(records: List[StepRecord], stats: Dict) -> Dict:
    """
    评估预算控制器的稳定性
    
    Returns:
        Dict: 评估结果，包含是否通过各项检查
    """
    target_budget = stats['target_budget']
    final_budget_total = stats['final_budget_total']
    final_budget_window = stats['final_budget_window']
    
    # 1. 硬约束检查: |Actual - B| ≤ 0.5%
    hard_constraint_threshold = 0.005  # 0.5%
    error_total = abs(final_budget_total - target_budget)
    error_window = abs(final_budget_window - target_budget)
    
    hard_constraint_pass = error_total <= hard_constraint_threshold
    
    # 2. 震荡检查: 是否超过 B ± 3%
    oscillation_threshold = 0.03  # 3%
    window_budgets = [r.actual_budget_window for r in records if not r.is_warmup]
    
    if window_budgets:
        max_oscillation = max([abs(b - target_budget) for b in window_budgets])
        oscillation_pass = max_oscillation <= oscillation_threshold
    else:
        max_oscillation = 0.0
        oscillation_pass = True
    
    # 3. 统计信息
    if len(window_budgets) > 0:
        budget_mean = np.mean(window_budgets)
        budget_std = np.std(window_budgets)
        budget_min = np.min(window_budgets)
        budget_max = np.max(window_budgets)
    else:
        budget_mean = final_budget_total
        budget_std = 0.0
        budget_min = final_budget_total
        budget_max = final_budget_total
    
    return {
        'hard_constraint_pass': hard_constraint_pass,
        'error_total': error_total,
        'error_window': error_window,
        'oscillation_pass': oscillation_pass,
        'max_oscillation': max_oscillation,
        'budget_mean': budget_mean,
        'budget_std': budget_std,
        'budget_min': budget_min,
        'budget_max': budget_max,
        'target_budget': target_budget,
    }


def visualize_results(records: List[StepRecord], stats: Dict, eval_result: Dict, output_path: str):
    """生成可视化图表"""
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    sample_indices = [r.sample_idx for r in records]
    lambda_values = [r.lambda_after for r in records]
    budget_window = [r.actual_budget_window for r in records]
    budget_total = [r.actual_budget_total for r in records]
    
    target_budget = stats['target_budget']
    
    # 子图 1: λ 阈值调整曲线
    ax1 = axes[0]
    ax1.plot(sample_indices, lambda_values, 'b-', linewidth=1.5, label='λ (阈值)')
    ax1.axhline(y=stats.get('lambda_init', 0.5), color='gray', linestyle='--', alpha=0.5, label='λ₀ (初始值)')
    ax1.set_xlabel('样本索引', fontsize=12)
    ax1.set_ylabel('阈值 λ', fontsize=12)
    ax1.set_title('阈值 λ 随样本流动的调整曲线', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # 标记 warmup 结束
    warmup_end = stats.get('window_size', 200)
    if len(sample_indices) > warmup_end:
        ax1.axvline(x=warmup_end, color='orange', linestyle='--', alpha=0.5, label='Warmup 结束')
        ax1.legend(loc='best')
    
    # 子图 2: 实际调用率曲线
    ax2 = axes[1]
    ax2.plot(sample_indices, budget_window, 'g-', linewidth=1.5, alpha=0.7, label='B̄_window (滑动窗口)')
    ax2.plot(sample_indices, budget_total, 'r-', linewidth=1.5, alpha=0.7, label='B̄_total (总体)')
    ax2.axhline(y=target_budget, color='blue', linestyle='-', linewidth=2, label=f'目标 B = {target_budget:.1%}')
    
    # 绘制约束区域
    hard_threshold = 0.005  # 0.5%
    oscillation_threshold = 0.03  # 3%
    ax2.axhspan(
        target_budget - hard_threshold,
        target_budget + hard_threshold,
        alpha=0.2, color='green', label='硬约束区域 (±0.5%)'
    )
    ax2.axhspan(
        target_budget - oscillation_threshold,
        target_budget + oscillation_threshold,
        alpha=0.1, color='yellow', label='震荡检查区域 (±3%)'
    )
    
    ax2.set_xlabel('样本索引', fontsize=12)
    ax2.set_ylabel('调用率', fontsize=12)
    ax2.set_title('实际调用率随样本流动的变化', fontsize=14, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([max(0, target_budget - 0.1), min(1, target_budget + 0.1)])
    
    # 标记 warmup 结束
    if len(sample_indices) > warmup_end:
        ax2.axvline(x=warmup_end, color='orange', linestyle='--', alpha=0.5)
    
    # 添加评估结果文本
    textstr = (
        f"最终统计:\n"
        f"  B̄_total = {stats['final_budget_total']:.4f} ({stats['final_budget_total']:.2%})\n"
        f"  B̄_window = {stats['final_budget_window']:.4f} ({stats['final_budget_window']:.2%})\n"
        f"  误差 |B̄ - B| = {eval_result['error_total']:.4f} ({eval_result['error_total']*100:.2f}%)\n"
        f"  最大震荡 = {eval_result['max_oscillation']:.4f} ({eval_result['max_oscillation']*100:.2f}%)\n"
        f"  硬约束: {'✓ 通过' if eval_result['hard_constraint_pass'] else '✗ 失败'}\n"
        f"  震荡检查: {'✓ 通过' if eval_result['oscillation_pass'] else '✗ 失败'}"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    # 论文级质量：dpi=300
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n[✓] 可视化图表已保存: {output_path} (dpi=300, 论文级质量)")


def main():
    parser = argparse.ArgumentParser(
        description='SH-DA++ v4.0 预算控制器稳定性测试',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='配置文件路径 (configs/router_config.yaml)'
    )
    parser.add_argument(
        '--metadata', '-m',
        type=str,
        default='./data/raw/HWDB_Benchmark/test_metadata.jsonl',
        help='测试集 metadata JSONL 文件路径'
    )
    parser.add_argument(
        '--image_root', '-i',
        type=str,
        default=None,
        help='图像根目录（未指定则从 metadata 路径推断）'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='最大样本数（用于快速测试）'
    )
    parser.add_argument(
        '--budget',
        type=float,
        default=None,
        help='目标调用率 B（覆盖配置文件）'
    )
    parser.add_argument(
        '--target_b',
        type=float,
        dest='budget',
        help='目标调用率 B 的别名 (等同于 --budget)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='./results/call_rate_over_time.png',
        help='可视化图表输出路径'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='打印详细日志'
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./models/agent_a_ppocr/PP-OCRv5_server_rec_infer/',
        help='Agent A 模型目录路径'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  SH-DA++ v4.0 预算控制器稳定性测试")
    print("=" * 70)
    
    # 1. 加载配置
    print("\n[Step 1] 加载配置...")
    config = load_config(args.config)
    print(f"  配置文件: {args.config}")
    
    # 2. 解析路径
    if args.image_root is None:
        args.image_root = infer_image_root(args.metadata)
        print(f"[Info] 自动推断 Image Root: {args.image_root}")
    
    metadata_dir = str(Path(args.metadata).parent)
    
    print(f"  Metadata: {args.metadata}")
    print(f"  Image Root: {args.image_root}")
    
    # 3. 加载 metadata
    print("\n[Step 2] 加载 metadata...")
    samples = load_metadata(args.metadata, limit=args.limit)
    print(f"  已加载 {len(samples)} 个样本")
    
    if not samples:
        print("[Error] 未加载任何样本")
        sys.exit(1)
    
    # 4. 流式处理
    print("\n[Step 3] 流式处理样本...")
    records, stats = process_samples_stream(
        samples=samples,
        image_root=args.image_root,
        metadata_dir=metadata_dir,
        config=config,
        model_dir=args.model_dir,
        target_budget=args.budget,
        limit=args.limit,
        verbose=args.verbose,
    )
    
    if not records:
        print("[Error] 未处理任何有效样本")
        sys.exit(1)
    
    # 5. 评估稳定性
    print("\n[Step 4] 评估稳定性...")
    eval_result = evaluate_stability(records, stats)
    
    # 6. 输出结果
    print("\n" + "=" * 70)
    print("  评估结果")
    print("=" * 70)
    
    target_budget = stats['target_budget']
    final_budget_total = stats['final_budget_total']
    final_budget_window = stats['final_budget_window']
    
    print(f"\n目标调用率 B: {target_budget:.4f} ({target_budget:.1%})")
    print(f"最终总体调用率 B̄_total: {final_budget_total:.4f} ({final_budget_total:.2%})")
    print(f"最终窗口调用率 B̄_window: {final_budget_window:.4f} ({final_budget_window:.2%})")
    print(f"误差 |B̄_total - B|: {eval_result['error_total']:.4f} ({eval_result['error_total']*100:.2f}%)")
    
    # 硬约束检查
    print(f"\n【硬约束检查】|Actual - B| ≤ 0.5%:")
    if eval_result['hard_constraint_pass']:
        print_color(f"  ✓ 通过 (误差: {eval_result['error_total']*100:.2f}%)", ANSIColors.GREEN)
    else:
        print_color(f"  ✗ 失败 (误差: {eval_result['error_total']*100:.2f}% > 0.5%)", ANSIColors.RED)
    
    # 震荡检查
    print(f"\n【震荡检查】是否超过 B ± 3%:")
    if eval_result['oscillation_pass']:
        print_color(f"  ✓ 通过 (最大震荡: {eval_result['max_oscillation']*100:.2f}%)", ANSIColors.GREEN)
    else:
        print_color(
            f"  ⚠️  警告: 震荡超过 3% (最大震荡: {eval_result['max_oscillation']*100:.2f}%)",
            ANSIColors.RED + ANSIColors.BOLD
        )
        print_color(
            "  建议: 减小 k 或增大 W 以提高稳定性",
            ANSIColors.YELLOW
        )
    
    # 统计信息
    print(f"\n统计信息:")
    print(f"  窗口调用率均值: {eval_result['budget_mean']:.4f} ({eval_result['budget_mean']:.2%})")
    print(f"  窗口调用率标准差: {eval_result['budget_std']:.4f} ({eval_result['budget_std']*100:.2f}%)")
    print(f"  窗口调用率范围: [{eval_result['budget_min']:.4f}, {eval_result['budget_max']:.4f}]")
    print(f"  最终阈值 λ: {stats['final_lambda']:.4f}")
    print(f"  总升级次数: {stats['total_upgrades']}/{stats['success_samples']}")
    
    # 7. 生成可视化
    print("\n[Step 5] 生成可视化图表...")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    visualize_results(records, stats, eval_result, str(output_path))
    
    print("\n" + "=" * 70)
    print("  测试完成!")
    print("=" * 70)
    
    # 返回退出码
    if not eval_result['hard_constraint_pass']:
        sys.exit(1)
    if not eval_result['oscillation_pass']:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()

