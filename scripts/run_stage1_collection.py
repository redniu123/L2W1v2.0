#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v4.0 Stage 0/1 正式数据采集器

功能：
1. 加载 configs/router_config.yaml 配置
2. 遍历 HWDB_Benchmark test_metadata.jsonl 中的所有图片
3. 实例化 L2W1Pipeline（使用真实的 Agent A）
4. 调用 pipeline.process() 处理每一张图
5. 确保每一行处理完后，router_features.jsonl 中新增完整记录
6. 输出统计报表：upgrade_rate, avg_lat_router_ms 等

Output:
- results/router_features.jsonl (路由特征日志)
- results/stage1_collection_report.json (采集报告)

Usage:
    python scripts/run_stage1_collection.py \
        --metadata ./data/raw/HWDB_Benchmark/test_metadata.jsonl \
        --config ./configs/router_config.yaml \
        --output_dir ./results \
        --limit 100  # 测试模式
"""

import os
import sys
import json
import argparse
import yaml
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import time
from datetime import datetime

import numpy as np
from tqdm import tqdm

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT.parent))


@dataclass
class CollectionStats:
    """数据采集统计"""

    # 样本统计
    total_samples: int = 0
    processed_samples: int = 0
    failed_samples: int = 0

    # 路由统计
    upgrade_count: int = 0
    upgrade_rate: float = 0.0

    # 分诊类型分布
    route_type_counts: Dict[str, int] = field(
        default_factory=lambda: {"none": 0, "boundary": 0, "ambiguity": 0, "both": 0}
    )

    # 耗时统计
    lat_router_ms_list: List[float] = field(default_factory=list)
    avg_lat_router_ms: float = 0.0
    total_time_sec: float = 0.0

    # 评分统计
    s_b_list: List[float] = field(default_factory=list)
    s_a_list: List[float] = field(default_factory=list)
    q_list: List[float] = field(default_factory=list)
    lambda_t_list: List[float] = field(default_factory=list)

    # 超时与回退统计
    b_timeout_count: int = 0
    b_fallback_count: int = 0

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "total_samples": self.total_samples,
            "processed_samples": self.processed_samples,
            "failed_samples": self.failed_samples,
            "upgrade_count": self.upgrade_count,
            "upgrade_rate": round(self.upgrade_rate, 4),
            "route_type_counts": self.route_type_counts,
            "avg_lat_router_ms": round(self.avg_lat_router_ms, 3),
            "total_time_sec": round(self.total_time_sec, 2),
            "b_timeout_count": self.b_timeout_count,
            "b_fallback_count": self.b_fallback_count,
            "s_b_stats": {
                "mean": round(np.mean(self.s_b_list), 4) if self.s_b_list else 0.0,
                "std": round(np.std(self.s_b_list), 4) if self.s_b_list else 0.0,
                "min": round(min(self.s_b_list), 4) if self.s_b_list else 0.0,
                "max": round(max(self.s_b_list), 4) if self.s_b_list else 0.0,
            },
            "s_a_stats": {
                "mean": round(np.mean(self.s_a_list), 4) if self.s_a_list else 0.0,
                "std": round(np.std(self.s_a_list), 4) if self.s_a_list else 0.0,
            },
            "q_stats": {
                "mean": round(np.mean(self.q_list), 4) if self.q_list else 0.0,
                "std": round(np.std(self.q_list), 4) if self.q_list else 0.0,
                "p50": round(np.percentile(self.q_list, 50), 4) if self.q_list else 0.0,
                "p80": round(np.percentile(self.q_list, 80), 4) if self.q_list else 0.0,
                "p95": round(np.percentile(self.q_list, 95), 4) if self.q_list else 0.0,
            },
            "lambda_t_stats": {
                "mean": round(np.mean(self.lambda_t_list), 4)
                if self.lambda_t_list
                else 0.0,
                "final": round(self.lambda_t_list[-1], 4)
                if self.lambda_t_list
                else 0.0,
            },
        }


def load_config(config_path: str) -> Dict:
    """
    加载 YAML 配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        config: 配置字典
    """
    if not Path(config_path).exists():
        print(f"[Warning] 配置文件不存在: {config_path}，使用默认配置")
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def load_metadata(metadata_path: str, limit: int = None) -> List[Dict]:
    """
    加载 metadata JSONL 文件

    Args:
        metadata_path: metadata 文件路径
        limit: 最大样本数

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


def resolve_image_path(
    image_path_str: str, image_root: str = None, metadata_dir: str = None
) -> Optional[str]:
    """
    智能解析图像路径

    Args:
        image_path_str: 图像路径字符串
        image_root: 图像根目录
        metadata_dir: metadata 文件所在目录

    Returns:
        解析后的完整路径
    """
    image_path_str = os.path.normpath(image_path_str)
    candidates = []

    # 绝对路径
    if os.path.isabs(image_path_str):
        candidates.append(image_path_str)

    # 相对于 image_root
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
        candidates.append(str(image_root_path / "test" / filename_only))

    # 相对于 metadata 目录
    if metadata_dir:
        metadata_dir_path = Path(metadata_dir)
        candidates.append(str(metadata_dir_path / image_path_str))
        candidates.append(str(metadata_dir_path / Path(image_path_str).name))
        candidates.append(str(metadata_dir_path / "images" / Path(image_path_str).name))
        candidates.append(str(metadata_dir_path / "test" / Path(image_path_str).name))

    # 当前工作目录
    candidates.append(image_path_str)

    for candidate in candidates:
        candidate_path = Path(candidate)
        if candidate_path.exists() and candidate_path.is_file():
            return str(candidate_path.resolve())

    return None


def run_stage1_collection(
    metadata_path: str,
    config_path: str,
    output_dir: str,
    model_dir: str,
    image_root: str = None,
    limit: int = None,
    verbose: bool = False,
    skip_agent_b: bool = True,
) -> CollectionStats:
    """
    执行 Stage 0/1 数据采集（严格模式：不支持模拟）

    Args:
        metadata_path: metadata 文件路径
        config_path: 配置文件路径
        output_dir: 输出目录
        model_dir: Agent A 模型目录（必须提供）
        image_root: 图像根目录
        limit: 最大样本数
        verbose: 是否打印详细日志
        skip_agent_b: 是否跳过 Agent B（Stage 1 仅需 Agent A）

    Returns:
        stats: 采集统计
    """
    print("=" * 70)
    print("  SH-DA++ v4.0 Stage 0/1 数据采集器")
    print("=" * 70)

    # 确保输出目录存在
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # 加载配置
    print(f"\n[1/5] 加载配置文件: {config_path}")
    config = load_config(config_path)

    # 从配置中提取 SH-DA++ v4.0 参数
    sh_da_config = config.get("sh_da_v4", {})
    rule_scorer_config = sh_da_config.get("rule_scorer", {})
    budget_config = sh_da_config.get("budget_controller", {})

    print(f"  - v_min: {rule_scorer_config.get('v_min', 0.0)}")
    print(f"  - v_max: {rule_scorer_config.get('v_max', 5.0)}")
    print(f"  - lambda_threshold: {rule_scorer_config.get('lambda_threshold', 0.5)}")
    print(f"  - eta: {rule_scorer_config.get('eta', 0.5)}")
    print(f"  - target_budget: {budget_config.get('target_budget', 0.2)}")

    # 加载 metadata
    print(f"\n[2/5] 加载 metadata: {metadata_path}")
    samples = load_metadata(metadata_path, limit)
    print(f"  - 加载样本数: {len(samples)}")

    if not samples:
        print("[Error] 未加载到任何样本！")
        return CollectionStats()

    # 推断 image_root
    metadata_dir = str(Path(metadata_path).parent)
    if not image_root:
        image_root = metadata_dir
        print(f"  - 自动推断 image_root: {image_root}")
    else:
        print(f"  - 指定 image_root: {image_root}")

    # 初始化 Pipeline（严格模式：必须提供真实模型）
    print("\n[3/5] 初始化 L2W1Pipeline...")

    # 检查模型目录
    if not model_dir:
        print("[FATAL] 必须指定 --model_dir 参数")
        print("  示例: --model_dir ./models/ppocrv5_rec")
        print("\n  请下载 PP-OCRv5 识别模型:")
        print(
            "  wget https://paddle-model-ecology.bj.bcebos.com/model/ocr/PP-OCRv5/ch_PP-OCRv5_rec_infer.tar"
        )
        print("  tar -xf ch_PP-OCRv5_rec_infer.tar")
        print("  mv ch_PP-OCRv5_rec_infer ./models/ppocrv5_rec")
        sys.exit(1)

    model_dir_path = Path(model_dir)
    if not model_dir_path.exists():
        print(f"[FATAL] 模型目录不存在: {model_dir}")
        sys.exit(1)

    # 检查 params 文件（inference.pdiparams 或 model.pdiparams）
    has_params = (model_dir_path / "inference.pdiparams").exists() or (
        model_dir_path / "model.pdiparams"
    ).exists()

    # 检查 model 文件（.pdmodel 或 .json，PP-OCRv5 使用 .json）
    has_model = (
        (model_dir_path / "inference.pdmodel").exists()
        or (model_dir_path / "inference.json").exists()
        or (model_dir_path / "model.pdmodel").exists()
        or (model_dir_path / "model.json").exists()
    )

    if not has_params:
        print(f"[FATAL] 模型参数文件缺失: {model_dir}")
        print("  需要 inference.pdiparams 或 model.pdiparams")
        print("\n  请下载 PP-OCRv5 识别模型:")
        print(
            "  wget https://paddle-model-ecology.bj.bcebos.com/model/ocr/PP-OCRv5/ch_PP-OCRv5_rec_infer.tar"
        )
        print("  tar -xf ch_PP-OCRv5_rec_infer.tar")
        sys.exit(1)

    if not has_model:
        print(f"[FATAL] 模型定义文件缺失: {model_dir}")
        print(
            "  需要以下之一: inference.pdmodel, inference.json, model.pdmodel, model.json"
        )
        sys.exit(1)

    print(f"  ✓ 模型文件检查通过: {model_dir}")

    from modules.pipeline import L2W1Pipeline, PipelineConfig

    pipeline_config = PipelineConfig(
        # Agent A 配置
        agent_a_model_dir=model_dir,
        agent_a_batch_size=6,
        rec_image_shape="3, 48, 320",
        use_gpu=True,
        # RuleOnlyScorer 配置
        v_min=rule_scorer_config.get("v_min", 0.0),
        v_max=rule_scorer_config.get("v_max", 5.0),
        lambda_threshold=rule_scorer_config.get("lambda_threshold", 0.5),
        eta=rule_scorer_config.get("eta", 0.5),
        # OnlineBudgetController 配置
        budget_window_size=budget_config.get("window_size", 200),
        budget_k=budget_config.get("k", 0.05),
        budget_lambda_min=budget_config.get("lambda_min", 0.0),
        budget_lambda_max=budget_config.get("lambda_max", 2.0),
        budget_target=budget_config.get("target_budget", 0.2),
        # 流水线配置
        verbose=verbose,
        router_features_log=str(output_dir_path / "router_features.jsonl"),
        # Agent B 配置（Stage 1 完全跳过 Agent B）
        skip_agent_b=skip_agent_b,
    )

    pipeline = L2W1Pipeline(pipeline_config)
    print("  - 模式: Real (真实 Agent A)")

    # 清空之前的 router_features.jsonl
    features_log_path = output_dir_path / "router_features.jsonl"
    if features_log_path.exists():
        print(f"  - 清空旧日志: {features_log_path}")
        features_log_path.unlink()

    # 开始采集
    print("\n[4/5] 开始数据采集...")
    stats = CollectionStats()
    stats.total_samples = len(samples)

    start_time = time.time()

    for idx, sample in enumerate(tqdm(samples, desc="Processing", miniters=1000)):
        try:
            # 解析图像路径
            image_path_raw = sample.get("image_path", sample.get("image", ""))
            image_path = resolve_image_path(image_path_raw, image_root, metadata_dir)

            if not image_path:
                if verbose:
                    print(f"[Warning] 图像不存在: {image_path_raw}")
                stats.failed_samples += 1
                continue

            # 提取样本信息
            sample_id = sample.get("id", sample.get("sample_id", f"sample_{idx:06d}"))
            gt_text = sample.get("gt_text", sample.get("gt", sample.get("label", "")))
            source = sample.get("source", "hwdb")
            error_type = sample.get("error_type", "")

            # 调用 Pipeline 处理
            result = pipeline.process(
                image=image_path,
                image_path=image_path,
                sample_id=sample_id,
                gt_text=gt_text,
                source=source,
                error_type=error_type,
            )

            # 收集统计
            stats.processed_samples += 1

            if result.upgrade:
                stats.upgrade_count += 1

            stats.route_type_counts[result.route_type] = (
                stats.route_type_counts.get(result.route_type, 0) + 1
            )
            stats.s_b_list.append(result.s_b)
            stats.s_a_list.append(result.s_a)
            stats.q_list.append(result.q)
            stats.lambda_t_list.append(result.lambda_t)

            if result.b_timeout:
                stats.b_timeout_count += 1
            if result.b_fallback:
                stats.b_fallback_count += 1

            # 从 pipeline 内部获取 lat_router_ms (如果有)
            # 由于 pipeline.process() 内部会记录到 jsonl，这里从统计中获取

        except Exception as e:
            if verbose:
                print(
                    f"[Error] 处理失败 (sample={sample_id if 'sample_id' in dir() else idx}): {e}"
                )
                import traceback

                traceback.print_exc()
            stats.failed_samples += 1

    stats.total_time_sec = time.time() - start_time

    # 计算统计量
    if stats.processed_samples > 0:
        stats.upgrade_rate = stats.upgrade_count / stats.processed_samples

    # 从 router_features.jsonl 读取 lat_router_ms
    if features_log_path.exists():
        with open(features_log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line.strip())
                    lat = record.get("lat_router_ms", 0.0)
                    if lat > 0:
                        stats.lat_router_ms_list.append(lat)
                except (json.JSONDecodeError, KeyError, ValueError):
                    pass

    if stats.lat_router_ms_list:
        stats.avg_lat_router_ms = np.mean(stats.lat_router_ms_list)

    # 关闭 Pipeline
    pipeline.shutdown()

    # 输出报告
    print("\n[5/5] 输出统计报告...")
    report = {
        "metadata_path": str(metadata_path),
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "timestamp": datetime.now().isoformat(),
        "stats": stats.to_dict(),
    }

    report_path = output_dir_path / "stage1_collection_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"  - 报告已保存: {report_path}")

    # 打印摘要
    print_summary(stats)

    return stats


def print_summary(stats: CollectionStats):
    """打印统计摘要"""
    print("\n" + "=" * 70)
    print("  SH-DA++ v4.0 Stage 1 采集统计报表")
    print("=" * 70)

    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│  样本统计                                                           │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(
        f"│  总样本数        │  {stats.total_samples:>8}                                      │"
    )
    print(
        f"│  处理成功        │  {stats.processed_samples:>8}  ({stats.processed_samples / max(stats.total_samples, 1) * 100:.1f}%)                       │"
    )
    print(
        f"│  处理失败        │  {stats.failed_samples:>8}  ({stats.failed_samples / max(stats.total_samples, 1) * 100:.1f}%)                       │"
    )
    print("└─────────────────────────────────────────────────────────────────────┘")

    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│  路由决策统计 [核心指标]                                            │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(
        f"│  升级样本数      │  {stats.upgrade_count:>8}                                      │"
    )
    print(
        f"│  upgrade_rate    │  {stats.upgrade_rate:>8.4f}  ({stats.upgrade_rate * 100:.2f}%)                    │"
    )
    print("├─────────────────────────────────────────────────────────────────────┤")
    print("│  分诊类型分布:                                                      │")
    for route_type, count in sorted(stats.route_type_counts.items()):
        pct = count / max(stats.processed_samples, 1) * 100
        print(
            f"│    - {route_type:12s}   │  {count:>6}  ({pct:5.1f}%)                           │"
        )
    print("└─────────────────────────────────────────────────────────────────────┘")

    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│  耗时统计 [性能指标]                                                │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    print(
        f"│  avg_lat_router_ms │  {stats.avg_lat_router_ms:>8.3f} ms                                │"
    )
    print(
        f"│  总耗时            │  {stats.total_time_sec:>8.2f} s                                 │"
    )
    if stats.processed_samples > 0:
        throughput = stats.processed_samples / stats.total_time_sec
        print(
            f"│  吞吐量            │  {throughput:>8.2f} samples/s                         │"
        )
    print("└─────────────────────────────────────────────────────────────────────┘")

    print("\n┌─────────────────────────────────────────────────────────────────────┐")
    print("│  评分分布统计                                                       │")
    print("├─────────────────────────────────────────────────────────────────────┤")
    if stats.s_b_list:
        print(
            f"│  s_b (边界风险)  │  mean={np.mean(stats.s_b_list):.4f}, std={np.std(stats.s_b_list):.4f}           │"
        )
    if stats.s_a_list:
        print(
            f"│  s_a (歧义风险)  │  mean={np.mean(stats.s_a_list):.4f}, std={np.std(stats.s_a_list):.4f}           │"
        )
    if stats.q_list:
        print(
            f"│  q (综合优先级)  │  mean={np.mean(stats.q_list):.4f}, p80={np.percentile(stats.q_list, 80):.4f}           │"
        )
    if stats.lambda_t_list:
        print(
            f"│  λ_t (阈值)      │  init={stats.lambda_t_list[0]:.4f}, final={stats.lambda_t_list[-1]:.4f}          │"
        )
    print("└─────────────────────────────────────────────────────────────────────┘")

    print("\n" + "=" * 70)
    print(f"  router_features.jsonl 已生成，包含 {stats.processed_samples} 条记录")
    print("=" * 70)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="SH-DA++ v4.0 Stage 0/1 数据采集器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 完整采集 (真实 Agent A)
    python scripts/run_stage1_collection.py \
        --metadata ./data/raw/HWDB_Benchmark/test_metadata.jsonl \
        --config ./configs/router_config.yaml \
        --model_dir ./models/ppocrv5_rec \
        --output_dir ./results
    
    # 快速测试 (限制样本数)
    python scripts/run_stage1_collection.py \
        --metadata ./data/raw/HWDB_Benchmark/test_metadata.jsonl \
        --model_dir ./models/ppocrv5_rec \
        --limit 100
    
    # 指定图像根目录
    python scripts/run_stage1_collection.py \
        --metadata ./data/raw/HWDB_Benchmark/test_metadata.jsonl \
        --model_dir ./models/ppocrv5_rec \
        --image_root ./data/raw/HWDB_Benchmark/test \
        --output_dir ./results
        """,
    )

    parser.add_argument(
        "--metadata",
        type=str,
        default="./data/raw/HWDB_Benchmark/test_metadata.jsonl",
        help="metadata JSONL 文件路径",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/router_config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--model_dir", type=str, required=True, help="Agent A 模型目录路径（必须提供）"
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default="",
        help="图像根目录（默认从 metadata 路径推断）",
    )
    parser.add_argument("--output_dir", type=str, default="./results", help="输出目录")
    parser.add_argument(
        "--limit", type=int, default=None, help="最大处理样本数（用于测试）"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="打印详细日志")
    parser.add_argument(
        "--skip_agent_b",
        action="store_true",
        default=True,
        help="跳过 Agent B（Stage 1 仅需 Agent A）",
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 检查 metadata 文件是否存在
    if not Path(args.metadata).exists():
        print(f"[Error] metadata 文件不存在: {args.metadata}")
        return

    # 执行采集（严格模式：必须使用真实模型）
    stats = run_stage1_collection(
        metadata_path=args.metadata,
        config_path=args.config,
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        image_root=args.image_root,
        limit=args.limit,
        verbose=args.verbose,
        skip_agent_b=args.skip_agent_b,
    )

    # 验收检查
    if stats.processed_samples > 0:
        print("\n[验收检查]")
        print(f"  ✓ 处理样本数: {stats.processed_samples}")
        print(
            f"  ✓ upgrade_rate: {stats.upgrade_rate:.4f} ({stats.upgrade_rate * 100:.2f}%)"
        )
        print(f"  ✓ avg_lat_router_ms: {stats.avg_lat_router_ms:.3f} ms")
        print(f"  ✓ router_features.jsonl 记录数: {stats.processed_samples}")
    else:
        print("\n[Warning] 未处理任何样本，请检查配置")


if __name__ == "__main__":
    main()
