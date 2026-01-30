#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v4.0 路由器参数校准脚本

功能：
1. 读取验证集 metadata，处理图像获取 Stage 0 信号
2. 提取 v_edge（Sobel 边缘响应）和计算 q 分数
3. 使用分位数估计 v_min, v_max, λ_0
4. 自动更新 configs/router_config.yaml
5. 输出分布直方图信息

公式：
- v_min = percentile(v_edge, 1%)
- v_max = percentile(v_edge, 99%)
- λ_0 = percentile(q, (1-B)×100)  # B=0.2 → 80% 分位数

Usage:
    python scripts/calibrate_router.py \
        --metadata ./data/raw/HWDB_Benchmark/train_metadata.jsonl \
        --image_root ./data/raw/HWDB_Benchmark/ \
        --budget 0.2 \
        --limit 1000 \
        --output_config ./configs/router_config.yaml
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import time

import numpy as np
import cv2

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))


@dataclass
class CalibrationStats:
    """校准统计结果"""

    # 样本统计
    total_samples: int = 0
    valid_samples: int = 0
    failed_samples: int = 0

    # v_edge 分布
    v_edge_min: float = 0.0
    v_edge_max: float = 0.0
    v_edge_mean: float = 0.0
    v_edge_std: float = 0.0
    v_edge_p1: float = 0.0  # 1% 分位数 → v_min
    v_edge_p99: float = 0.0  # 99% 分位数 → v_max

    # q 分布
    q_min: float = 0.0
    q_max: float = 0.0
    q_mean: float = 0.0
    q_std: float = 0.0
    q_p_threshold: float = 0.0  # (1-B) 分位数 → λ_0

    # 分诊类型分布
    route_type_counts: Dict[str, int] = field(default_factory=dict)

    # 校准参数
    calibrated_v_min: float = 0.0
    calibrated_v_max: float = 0.0
    calibrated_lambda_0: float = 0.0
    target_budget: float = 0.2


def compute_sobel_edge_strength(image: np.ndarray) -> float:
    """
    计算图像的 Sobel 边缘响应强度

    用于量化边界区域的视觉清晰度 v_edge

    Args:
        image: BGR 或灰度图像

    Returns:
        v_edge: Sobel 梯度响应的均值
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Sobel 梯度
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # 梯度幅值
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # 提取左右边界区域 (各 10% 宽度)
    h, w = gray.shape
    boundary_width = max(1, int(w * 0.1))

    left_region = magnitude[:, :boundary_width]
    right_region = magnitude[:, -boundary_width:]

    # 边界区域均值
    v_edge = (np.mean(left_region) + np.mean(right_region)) / 2.0

    return float(v_edge)


def load_metadata(metadata_path: str, limit: int = None) -> List[Dict]:
    """
    加载 metadata JSONL 文件

    Args:
        metadata_path: metadata 文件路径
        limit: 最大样本数（用于快速测试）

    Returns:
        samples: 样本列表
    """
    samples = []

    with open(metadata_path, "r", encoding="utf-8") as f:
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
                print(f"[Warning] 解析失败 (行 {i + 1}): {e}")

    return samples


def infer_image_root(metadata_path: str) -> str:
    """
    从 metadata 路径自动推断 image_root

    例如：metadata_path = "./data/raw/HWDB_Benchmark/train_metadata.jsonl"
         → image_root = "./data/raw/HWDB_Benchmark/"

    Args:
        metadata_path: metadata 文件路径

    Returns:
        image_root: 推断的图像根目录
    """
    metadata_path_obj = Path(metadata_path)
    # 返回 metadata 文件所在目录
    return str(metadata_path_obj.parent)


def resolve_image_path(
    image_path_str: str, image_root: str = None, metadata_dir: str = None
) -> Optional[str]:
    """
    智能解析图像路径，支持多种格式

    解析顺序：
    1. 绝对路径（直接检查）
    2. 相对于 image_root
    3. 相对于 metadata 文件所在目录
    4. 相对于当前工作目录
    5. 仅文件名（在 image_root 中搜索）

    Args:
        image_path_str: 图像路径字符串（来自 metadata）
        image_root: 图像根目录
        metadata_dir: metadata 文件所在目录

    Returns:
        解析后的完整路径，如果不存在则返回 None
    """
    # 标准化路径
    image_path_str = os.path.normpath(image_path_str)

    # 候选路径列表
    candidates = []

    # 优先级 1: 绝对路径
    if os.path.isabs(image_path_str):
        candidates.append(image_path_str)

    # 优先级 2: 相对于 image_root
    if image_root:
        image_root_path = Path(image_root)
        # 检查是否已经包含 image_root（避免重复拼接）
        image_path_normalized = image_path_str.replace("\\", "/").strip("./")
        image_root_normalized = str(image_root_path).replace("\\", "/").strip("./")

        if not image_path_normalized.startswith(image_root_normalized):
            # 未包含，尝试拼接
            candidates.append(str(image_root_path / image_path_str))
        else:
            # 已包含，直接使用
            candidates.append(image_path_str)

        # 仅文件名情况
        filename_only = Path(image_path_str).name
        candidates.append(str(image_root_path / filename_only))
        candidates.append(str(image_root_path / "images" / filename_only))

    # 优先级 3: 相对于 metadata 文件所在目录
    if metadata_dir:
        metadata_dir_path = Path(metadata_dir)
        candidates.append(str(metadata_dir_path / image_path_str))
        candidates.append(str(metadata_dir_path / Path(image_path_str).name))
        candidates.append(str(metadata_dir_path / "images" / Path(image_path_str).name))

    # 优先级 4: 相对于当前工作目录
    candidates.append(image_path_str)
    candidates.append(str(Path(image_path_str).name))

    # 尝试每个候选路径
    for candidate in candidates:
        candidate_path = Path(candidate)
        if candidate_path.exists() and candidate_path.is_file():
            return str(candidate_path.resolve())

    # 所有候选路径都不存在
    return None


def process_samples_with_engine(
    samples: List[Dict],
    image_root: str,
    metadata_dir: str = None,
    eta: float = 0.5,
    model_dir: str = None,
    verbose: bool = False,
) -> Tuple[List[float], List[float], List[str], List[Dict]]:
    """
    使用 TextRecognizerWithLogits 处理样本，提取信号

    Args:
        samples: 样本列表
        image_root: 图像根目录
        metadata_dir: metadata 文件所在目录（用于路径解析）
        eta: q 计算中的 η 参数
        verbose: 是否打印详细日志

    Returns:
        Tuple[v_edges, q_scores, route_types, details]:
            - v_edges: Sobel 边缘响应列表
            - q_scores: 综合优先级列表
            - route_types: 分诊类型列表
            - details: 详细信息列表
    """
    v_edges = []
    q_scores = []
    route_types = []
    details = []

    # 导入必要模块（严格模式：失败则退出）
    try:
        from modules.paddle_engine.predict_rec_modified import TextRecognizerWithLogits
        from modules.router.uncertainty_router import RuleOnlyScorer, RuleScorerConfig
    except ImportError as e:
        print(f"[FATAL] 无法导入必要模块: {e}")
        print("\n请确保以下模块可用:")
        print("  - modules.paddle_engine.predict_rec_modified.TextRecognizerWithLogits")
        print("  - modules.router.uncertainty_router.RuleOnlyScorer")
        sys.exit(1)

    # 初始化 RuleOnlyScorer（使用默认配置，后续校准）
    scorer_config = RuleScorerConfig(
        v_min=0.0,
        v_max=100.0,  # 初始宽范围
        lambda_threshold=0.5,
        eta=eta,
    )
    scorer = RuleOnlyScorer(config=scorer_config)

    # 初始化 Engine（严格模式：失败则退出）
    engine = None
    try:
        import tools.infer.utility as utility

        # 正确初始化 args：先创建 parser，再 parse_args
        parser = utility.init_args()
        args = parser.parse_args([])  # 使用默认参数

        # 设置模型路径
        if model_dir:
            args.rec_model_dir = model_dir
        else:
            # 尝试从默认路径读取
            default_model_dir = "./models/ppocrv5_rec"
            if os.path.exists(default_model_dir):
                args.rec_model_dir = default_model_dir
            else:
                print(f"[FATAL] 模型目录不存在: {default_model_dir}")
                print("\n请下载 PP-OCRv5 识别模型:")
                print(
                    "  wget https://paddle-model-ecology.bj.bcebos.com/model/ocr/PP-OCRv5/ch_PP-OCRv5_rec_infer.tar"
                )
                print("  tar -xf ch_PP-OCRv5_rec_infer.tar")
                print("  mv ch_PP-OCRv5_rec_infer ./models/ppocrv5_rec")
                print("\n或使用 --model_dir 指定模型路径")
                sys.exit(1)

        # 检查模型文件是否存在（兼容 PP-OCRv5 新格式）
        model_path = Path(args.rec_model_dir)

        # 检查 params 文件（inference.pdiparams 或 model.pdiparams）
        has_params = (model_path / "inference.pdiparams").exists() or (
            model_path / "model.pdiparams"
        ).exists()

        # 检查 model 文件（.pdmodel 或 .json，PP-OCRv5 使用 .json）
        has_model = (
            (model_path / "inference.pdmodel").exists()
            or (model_path / "inference.json").exists()
            or (model_path / "model.pdmodel").exists()
            or (model_path / "model.json").exists()
        )

        if not has_params:
            print(f"[FATAL] 模型参数文件缺失: {args.rec_model_dir}")
            print("  需要 inference.pdiparams 或 model.pdiparams")
            sys.exit(1)

        if not has_model:
            print(f"[FATAL] 模型定义文件缺失: {args.rec_model_dir}")
            print(
                "  需要以下之一: inference.pdmodel, inference.json, model.pdmodel, model.json"
            )
            sys.exit(1)

        print(f"[✓] 模型文件检查通过: {args.rec_model_dir}")

        # 设置其他必要参数
        args.rec_batch_num = 1
        args.rec_image_shape = "3, 48, 320"
        args.use_gpu = True
        args.warmup = False
        args.benchmark = False
        args.use_onnx = False
        args.return_word_box = False

        engine = TextRecognizerWithLogits(args)
        print(f"[✓] Agent A 初始化成功: {args.rec_model_dir}")

    except Exception as e:
        print(f"[FATAL] Agent A 初始化失败: {e}")
        import traceback

        traceback.print_exc()
        print("\n请检查:")
        print("  1. 模型文件是否完整")
        print("  2. PaddlePaddle 是否正确安装")
        print("  3. GPU/CUDA 是否可用")
        sys.exit(1)

    # 统计信息
    failed_count = 0
    success_count = 0

    # 处理样本
    for i, sample in enumerate(samples):
        if (i + 1) % 1000 == 0:
            print(
                f"[Progress] 处理中: {i + 1}/{len(samples)} (成功: {success_count}, 失败: {failed_count})"
            )

        try:
            # 获取图像路径（支持 'image' 和 'image_path' 字段）
            image_path_str = sample.get("image_path", sample.get("image", ""))

            if not image_path_str:
                failed_count += 1
                if verbose:
                    print(f"[Warning] 样本 {i + 1} 缺少图像路径字段")
                continue

            # 使用智能路径解析
            resolved_path = resolve_image_path(
                image_path_str=image_path_str,
                image_root=image_root,
                metadata_dir=metadata_dir,
            )

            if resolved_path is None:
                failed_count += 1
                if verbose or i < 5:  # 前5个失败时总是打印
                    print(
                        f"[Warning] 无法解析图像路径 (样本 {i + 1}): {image_path_str}"
                    )
                    if image_root:
                        print(f"  尝试的根目录: {image_root}")
                    if metadata_dir:
                        print(f"  Metadata 目录: {metadata_dir}")
                continue

            # 读取图像
            image = cv2.imread(resolved_path)
            if image is None:
                failed_count += 1
                if verbose or i < 5:
                    print(f"[Warning] 无法读取图像 (样本 {i + 1}): {resolved_path}")
                continue

            # 1. 计算 v_edge (Sobel 边缘响应)
            v_edge = compute_sobel_edge_strength(image)

            # 2. 获取 Stage 0 信号并计算 q（严格模式：真实推理）
            output = engine([image])

            # 提取识别结果
            results_list = output.get("results", [])
            rec_text = ""
            rec_conf = 0.0
            if results_list and len(results_list) > 0:
                if (
                    isinstance(results_list[0], (list, tuple))
                    and len(results_list[0]) >= 2
                ):
                    rec_text, rec_conf = results_list[0][0], results_list[0][1]
                elif isinstance(results_list[0], dict):
                    rec_text = results_list[0].get("text", "")
                    rec_conf = results_list[0].get("confidence", 0.0)

            boundary_stats = (
                output.get("boundary_stats", [{}])[0]
                if output.get("boundary_stats")
                else {}
            )
            top2_info = (
                output.get("top2_info", [{}])[0] if output.get("top2_info") else {}
            )

            # 计算 q 分数
            result = scorer.score(
                boundary_stats=boundary_stats,
                top2_info=top2_info,
                r_d=0.0,
                v_edge=v_edge,
            )

            q = result.q
            route_type = result.route_type.value

            # ========== 实时打印识别结果（每 100 个样本 + 前 5 个） ==========
            if (i + 1) <= 5 or (i + 1) % 100 == 0:
                display_text = rec_text[:25] + "..." if len(rec_text) > 25 else rec_text
                sample_id = sample.get("id", f"sample_{i}")
                print(
                    f"  [{sample_id}] '{display_text}' | conf={rec_conf:.2%} | q={q:.4f} | v_edge={v_edge:.2f}"
                )

            v_edges.append(v_edge)
            q_scores.append(q)
            route_types.append(route_type)
            details.append(
                {
                    "id": sample.get("id", f"sample_{i}"),
                    "text": rec_text,
                    "confidence": rec_conf,
                    "v_edge": v_edge,
                    "q": q,
                    "route_type": route_type,
                }
            )
            success_count += 1

        except Exception as e:
            failed_count += 1
            if verbose:
                print(f"[Error] 处理样本 {i + 1} 失败: {e}")
            import traceback

            if verbose:
                traceback.print_exc()

    if verbose or failed_count > 0:
        print(
            f"\n[统计] 处理完成: 成功 {success_count}, 失败 {failed_count}, 总计 {len(samples)}"
        )

    return v_edges, q_scores, route_types, details


def compute_calibration_stats(
    v_edges: List[float],
    q_scores: List[float],
    route_types: List[str],
    target_budget: float = 0.2,
) -> CalibrationStats:
    """
    计算校准统计量

    Args:
        v_edges: v_edge 列表
        q_scores: q 分数列表
        route_types: 分诊类型列表
        target_budget: 目标调用率 B

    Returns:
        CalibrationStats: 校准统计结果
    """
    stats = CalibrationStats()

    if not v_edges or not q_scores:
        return stats

    v_arr = np.array(v_edges)
    q_arr = np.array(q_scores)

    stats.total_samples = len(v_edges)
    stats.valid_samples = len(v_edges)
    stats.target_budget = target_budget

    # v_edge 分布统计
    stats.v_edge_min = float(np.min(v_arr))
    stats.v_edge_max = float(np.max(v_arr))
    stats.v_edge_mean = float(np.mean(v_arr))
    stats.v_edge_std = float(np.std(v_arr))
    stats.v_edge_p1 = float(np.percentile(v_arr, 1))
    stats.v_edge_p99 = float(np.percentile(v_arr, 99))

    # q 分布统计
    stats.q_min = float(np.min(q_arr))
    stats.q_max = float(np.max(q_arr))
    stats.q_mean = float(np.mean(q_arr))
    stats.q_std = float(np.std(q_arr))

    # λ_0 = (1-B) 分位数
    # B=0.2 → 取 80% 分位数，使得 top 20% 的样本被升级
    lambda_percentile = (1 - target_budget) * 100
    stats.q_p_threshold = float(np.percentile(q_arr, lambda_percentile))

    # 分诊类型分布
    for rt in route_types:
        stats.route_type_counts[rt] = stats.route_type_counts.get(rt, 0) + 1

    # 校准参数
    stats.calibrated_v_min = stats.v_edge_p1
    stats.calibrated_v_max = stats.v_edge_p99
    stats.calibrated_lambda_0 = stats.q_p_threshold

    return stats


def print_histogram(data: np.ndarray, name: str, bins: int = 20):
    """
    打印 ASCII 直方图

    Args:
        data: 数据数组
        name: 数据名称
        bins: 分箱数
    """
    hist, bin_edges = np.histogram(data, bins=bins)
    max_count = max(hist)
    bar_width = 40

    print(f"\n{'=' * 60}")
    print(f"  {name} 分布直方图")
    print(f"{'=' * 60}")
    print(f"  范围: [{np.min(data):.4f}, {np.max(data):.4f}]")
    print(f"  均值: {np.mean(data):.4f}, 标准差: {np.std(data):.4f}")
    print(f"  1%: {np.percentile(data, 1):.4f}, 99%: {np.percentile(data, 99):.4f}")
    print(f"{'=' * 60}")

    for i in range(len(hist)):
        left = bin_edges[i]
        right = bin_edges[i + 1]
        count = hist[i]
        bar_len = int(count / max_count * bar_width) if max_count > 0 else 0
        bar = "█" * bar_len
        print(f"  [{left:8.3f}, {right:8.3f}): {count:5d} | {bar}")

    print(f"{'=' * 60}\n")


def update_config_yaml(
    config_path: str,
    v_min: float,
    v_max: float,
    lambda_0: float,
    target_budget: float,
):
    """
    更新 router_config.yaml 配置文件

    Args:
        config_path: 配置文件路径
        v_min: 校准的 v_min
        v_max: 校准的 v_max
        lambda_0: 校准的 λ_0
        target_budget: 目标调用率
    """
    import yaml

    # 读取现有配置
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    else:
        config = {}

    # 添加/更新 SH-DA++ v4.0 配置节
    if "sh_da_v4" not in config:
        config["sh_da_v4"] = {}

    existing_rule_scorer = config["sh_da_v4"].get("rule_scorer", {}) or {}
    existing_rule_scorer.update(
        {
            "v_min": round(v_min, 4),
            "v_max": round(v_max, 4),
            "lambda_threshold": round(lambda_0, 4),
            "eta": 0.5,
            "_calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "_target_budget": target_budget,
        }
    )
    config["sh_da_v4"]["rule_scorer"] = existing_rule_scorer

    config["sh_da_v4"]["budget_controller"] = {
        "window_size": 200,
        "k": 0.05,
        "lambda_min": 0.0,
        "lambda_max": 2.0,
        "lambda_init": round(lambda_0, 4),
        "target_budget": target_budget,
    }

    # 写入配置
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(
            config, f, allow_unicode=True, default_flow_style=False, sort_keys=False
        )

    print(f"\n[✓] 配置已更新: {config_path}")


def load_config_defaults(config_path: str) -> Dict:
    """
    从配置文件加载默认值

    Args:
        config_path: 配置文件路径

    Returns:
        defaults: 默认参数字典
    """
    defaults = {}

    if not os.path.exists(config_path):
        return defaults

    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

        # 从配置中提取可能的默认值
        # 注意：router_config.yaml 可能不包含这些字段，这里只是预留接口
        if "calibration" in config:
            calib = config["calibration"]
            defaults["metadata"] = calib.get("metadata_path", "")
            defaults["image_root"] = calib.get("image_root", "")
            defaults["eta"] = calib.get("eta", 0.5)

        # 如果存在 sh_da_v4 配置，读取其中的参数
        if "sh_da_v4" in config:
            sh_da = config["sh_da_v4"]
            if "rule_scorer" in sh_da:
                defaults["eta"] = sh_da["rule_scorer"].get("eta", 0.5)
            if "budget_controller" in sh_da:
                defaults["budget"] = sh_da["budget_controller"].get(
                    "target_budget", 0.2
                )

        # 从 agent_a 配置读取模型目录
        if "agent_a" in config:
            defaults["model_dir"] = config["agent_a"].get(
                "model_dir", "./models/ppocrv5_rec"
            )

    except Exception as e:
        print(f"[Warning] 读取配置文件失败: {e}")

    return defaults


def main():
    parser = argparse.ArgumentParser(
        description="SH-DA++ v4.0 路由器参数校准脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 使用配置文件
    python scripts/calibrate_router.py \\
        --config configs/router_config.yaml \\
        --target_b 0.2

    # 使用 HWDB 训练集校准
    python scripts/calibrate_router.py \\
        --metadata ./data/raw/HWDB_Benchmark/train_metadata.jsonl \\
        --image_root ./data/raw/HWDB_Benchmark/ \\
        --budget 0.2 \\
        --limit 1000

    # 仅计算统计，不更新配置
    python scripts/calibrate_router.py \\
        --metadata ./data/test_metadata.jsonl \\
        --dry_run
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="配置文件路径（用于读取默认值和更新配置）",
    )
    parser.add_argument(
        "--metadata", "-m", type=str, default=None, help="Metadata JSONL 文件路径"
    )
    parser.add_argument(
        "--image_root",
        "-i",
        type=str,
        default=None,
        help="图像根目录（用于拼接相对路径）",
    )
    parser.add_argument(
        "--budget",
        "-b",
        type=float,
        default=None,
        help="目标调用率 B (default: 0.2 = 20%%)",
    )
    parser.add_argument(
        "--target_b",
        type=float,
        dest="budget",
        help="目标调用率 B 的别名 (等同于 --budget)",
    )
    parser.add_argument(
        "--eta", type=float, default=None, help="q 计算中的 η 参数 (default: 0.5)"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Agent A 模型目录路径 (默认: ./models/ppocrv5_rec)",
    )
    parser.add_argument(
        "--limit", "-l", type=int, default=None, help="最大样本数（用于快速测试）"
    )
    parser.add_argument(
        "--output_config",
        "-o",
        type=str,
        default=None,
        help="输出配置文件路径（默认使用 --config 指定的路径）",
    )
    parser.add_argument(
        "--output_stats",
        type=str,
        default="./results/calibration_stats.json",
        help="输出统计结果 JSON 路径",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="仅计算统计，不更新配置文件"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="打印详细日志")

    args = parser.parse_args()

    # 从配置文件加载默认值（如果提供了 --config）
    config_defaults = {}
    if args.config:
        config_defaults = load_config_defaults(args.config)
        # 如果未指定 output_config，使用 --config 指定的路径
        if args.output_config is None:
            args.output_config = args.config

    # 设置默认值（优先级：命令行参数 > 配置文件 > 硬编码默认值）
    if args.metadata is None:
        args.metadata = config_defaults.get(
            "metadata", "./data/raw/HWDB_Benchmark/train_metadata.jsonl"
        )

    if args.image_root is None:
        args.image_root = config_defaults.get("image_root", "")
        # 如果仍为空，从 metadata 路径自动推断
        if not args.image_root:
            args.image_root = infer_image_root(args.metadata)
            if args.image_root:
                print(f"[Info] 自动推断 Image Root: {args.image_root}")

    if args.budget is None:
        args.budget = config_defaults.get("budget", 0.2)

    if args.eta is None:
        args.eta = config_defaults.get("eta", 0.5)

    if args.model_dir is None:
        args.model_dir = config_defaults.get("model_dir", "./models/ppocrv5_rec")

    if args.output_config is None:
        args.output_config = "./configs/router_config.yaml"

    print("=" * 70)
    print("  SH-DA++ v4.0 路由器参数校准")
    print("=" * 70)
    if args.config:
        print(f"  配置文件: {args.config}")
    print(f"  Metadata: {args.metadata}")
    print(f"  Image Root: {args.image_root or '(自动推断或使用绝对路径)'}")
    print(f"  Metadata 目录: {Path(args.metadata).parent}")
    print(f"  目标调用率 B: {args.budget:.1%}")
    print(f"  η 参数: {args.eta}")
    print(f"  样本限制: {args.limit or '无限制'}")
    print(f"  输出配置: {args.output_config}")
    print("=" * 70)

    # 1. 加载 metadata
    print("\n[Step 1] 加载 metadata...")

    if not os.path.exists(args.metadata):
        print(f"[Error] Metadata 文件不存在: {args.metadata}")
        print("\n请提供有效的 metadata 文件路径，格式为 JSONL，每行包含:")
        print('  {"id": "...", "image_path": "...", "gt_text": "..."}')
        sys.exit(1)

    samples = load_metadata(args.metadata, limit=args.limit)
    print(f"  已加载 {len(samples)} 个样本")

    if not samples:
        print("[Error] 未加载任何样本")
        sys.exit(1)

    # 2. 处理样本，提取信号
    print("\n[Step 2] 提取 v_edge 和计算 q 分数...")

    # 获取 metadata 文件所在目录
    metadata_dir = str(Path(args.metadata).parent)

    start_time = time.time()
    v_edges, q_scores, route_types, details = process_samples_with_engine(
        samples=samples,
        image_root=args.image_root,
        metadata_dir=metadata_dir,
        eta=args.eta,
        model_dir=args.model_dir,
        verbose=args.verbose,
    )
    elapsed = time.time() - start_time

    print(f"  处理完成: {len(v_edges)} 个有效样本")
    print(f"  耗时: {elapsed:.2f}s ({len(v_edges) / elapsed:.1f} samples/s)")

    if not v_edges:
        print("[Error] 未提取到有效数据")
        sys.exit(1)

    # 3. 计算校准统计量
    print("\n[Step 3] 计算分位数...")

    stats = compute_calibration_stats(
        v_edges=v_edges,
        q_scores=q_scores,
        route_types=route_types,
        target_budget=args.budget,
    )

    print(f"\n  校准结果:")
    print(f"  {'─' * 40}")
    print(f"  v_min (1% 分位数):   {stats.calibrated_v_min:.4f}")
    print(f"  v_max (99% 分位数):  {stats.calibrated_v_max:.4f}")
    print(
        f"  λ_0 ({int((1 - args.budget) * 100)}% 分位数): {stats.calibrated_lambda_0:.4f}"
    )
    print(f"  {'─' * 40}")

    # 4. 打印直方图
    print("\n[Step 4] 分布直方图...")
    print_histogram(np.array(v_edges), "v_edge (Sobel 边缘响应)")
    print_histogram(np.array(q_scores), "q (综合优先级)")

    # 分诊类型分布
    print(f"\n分诊类型分布:")
    print(f"  {'─' * 30}")
    for rt, count in sorted(stats.route_type_counts.items()):
        pct = count / stats.valid_samples * 100
        print(f"  {rt:12s}: {count:5d} ({pct:5.1f}%)")
    print(f"  {'─' * 30}")

    # 5. 保存统计结果
    stats_dict = {
        "total_samples": stats.total_samples,
        "valid_samples": stats.valid_samples,
        "v_edge": {
            "min": stats.v_edge_min,
            "max": stats.v_edge_max,
            "mean": stats.v_edge_mean,
            "std": stats.v_edge_std,
            "p1": stats.v_edge_p1,
            "p99": stats.v_edge_p99,
        },
        "q": {
            "min": stats.q_min,
            "max": stats.q_max,
            "mean": stats.q_mean,
            "std": stats.q_std,
            "p_threshold": stats.q_p_threshold,
            "threshold_percentile": (1 - args.budget) * 100,
        },
        "route_type_counts": stats.route_type_counts,
        "calibrated_params": {
            "v_min": stats.calibrated_v_min,
            "v_max": stats.calibrated_v_max,
            "lambda_0": stats.calibrated_lambda_0,
            "eta": args.eta,
            "target_budget": args.budget,
        },
        "metadata": {
            "source_file": args.metadata,
            "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }

    stats_path = Path(args.output_stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_dict, f, ensure_ascii=False, indent=2)

    print(f"\n[✓] 统计结果已保存: {args.output_stats}")

    # 5.1. 保存 PP-OCRv5 识别结果到 JSONL
    ppocrv5_text_path = stats_path.parent / "ppocrv5_text.jsonl"
    with open(ppocrv5_text_path, "w", encoding="utf-8") as f:
        for detail in details:
            record = {
                "id": detail.get("id", ""),
                "text": detail.get("text", ""),
                "confidence": detail.get("confidence", 0.0),
                "q": detail.get("q", 0.0),
                "v_edge": detail.get("v_edge", 0.0),
                "route_type": detail.get("route_type", ""),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    print(f"[✓] PP-OCRv5 识别结果已保存: {ppocrv5_text_path} ({len(details)} 条)")

    # 6. 更新配置文件
    if not args.dry_run:
        print("\n[Step 5] 更新配置文件...")

        config_path = Path(args.output_config)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            update_config_yaml(
                config_path=str(config_path),
                v_min=stats.calibrated_v_min,
                v_max=stats.calibrated_v_max,
                lambda_0=stats.calibrated_lambda_0,
                target_budget=args.budget,
            )
        except ImportError:
            print("[Warning] 需要安装 pyyaml: pip install pyyaml")
            print(f"\n手动更新配置:")
            print(f"  v_min: {stats.calibrated_v_min:.4f}")
            print(f"  v_max: {stats.calibrated_v_max:.4f}")
            print(f"  lambda_init: {stats.calibrated_lambda_0:.4f}")
    else:
        print("\n[Info] Dry run 模式，跳过配置更新")
        print(f"\n建议手动更新 {args.output_config}:")
        print(f"  v_min: {stats.calibrated_v_min:.4f}")
        print(f"  v_max: {stats.calibrated_v_max:.4f}")
        print(f"  lambda_init: {stats.calibrated_lambda_0:.4f}")

    # 总结
    print("\n" + "=" * 70)
    print("  校准完成!")
    print("=" * 70)
    print(f"\n  使用以下参数初始化 RuleOnlyScorer 和 OnlineBudgetController:")
    print(f"""
    from modules.router.uncertainty_router import (
        RuleOnlyScorer, RuleScorerConfig,
        OnlineBudgetController, BudgetControllerConfig
    )
    
    scorer = RuleOnlyScorer(RuleScorerConfig(
        v_min={stats.calibrated_v_min:.4f},
        v_max={stats.calibrated_v_max:.4f},
        lambda_threshold={stats.calibrated_lambda_0:.4f},
        eta={args.eta},
    ))
    
    controller = OnlineBudgetController(BudgetControllerConfig(
        lambda_init={stats.calibrated_lambda_0:.4f},
        target_budget={args.budget},
    ))
    """)


if __name__ == "__main__":
    main()
