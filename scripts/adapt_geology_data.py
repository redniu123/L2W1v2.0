#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ Stage 2: Data Adapter for Geology Dataset

将原始地质数据转换为 L2W1 Master Data Protocol v2.0 格式。

V2.0 协议格式：
{
    "id": "unique_id",
    "image": "path/to/image.jpg",
    "gt_text": "ground_truth_text",
    "source": "geology",
    "metadata": {
        "confidence": 0.95,
        "domain": "geology"
    }
}
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm


def adapt_geology_data(
    input_jsonl: str,
    output_jsonl: str,
    image_dir: str = None,
) -> Dict:
    """
    将地质数据适配为 V2.0 协议格式

    Args:
        input_jsonl: 输入 JSONL 文件路径
        output_jsonl: 输出 JSONL 文件路径
        image_dir: 图像目录（可选，用于验证路径）

    Returns:
        Dict: 适配统计信息
    """
    input_path = Path(input_jsonl)
    output_path = Path(output_jsonl)

    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_jsonl}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 统计信息
    stats = {
        "total_records": 0,
        "valid_records": 0,
        "invalid_records": 0,
        "errors": [],
    }

    with open(input_path, "r", encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:
        for line_idx, line in enumerate(tqdm(fin, desc="适配数据")):
            try:
                stats["total_records"] += 1
                record = json.loads(line.strip())

                # 提取字段（兼容多种格式）
                record_id = record.get("id") or record.get("image_id") or f"geo_{line_idx}"
                image_path = record.get("image") or record.get("image_path")
                gt_text = record.get("gt_text") or record.get("text") or record.get("label")
                confidence = record.get("confidence", 0.95)

                # 验证必要字段
                if not image_path or not gt_text:
                    stats["invalid_records"] += 1
                    stats["errors"].append(
                        f"Line {line_idx}: 缺少 image 或 gt_text 字段"
                    )
                    continue

                # 构造 V2.0 格式
                v2_record = {
                    "id": str(record_id),
                    "image": str(image_path),
                    "gt_text": str(gt_text),
                    "source": "geology",
                    "metadata": {
                        "confidence": float(confidence),
                        "domain": "geology",
                    },
                }

                # 写入输出文件
                fout.write(json.dumps(v2_record, ensure_ascii=False) + "\n")
                stats["valid_records"] += 1

            except json.JSONDecodeError as e:
                stats["invalid_records"] += 1
                stats["errors"].append(f"Line {line_idx}: JSON 解析错误 - {e}")
            except Exception as e:
                stats["invalid_records"] += 1
                stats["errors"].append(f"Line {line_idx}: {type(e).__name__} - {e}")

    return stats


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="地质数据适配器")
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default="data/geo/geotext.jsonl",
        help="输入 JSONL 文件路径",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="data/geo/geotext_v2.jsonl",
        help="输出 JSONL 文件路径",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="data/geo/geotext/",
        help="图像目录",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SH-DA++ Stage 2: 地质数据适配器")
    print("=" * 60)

    stats = adapt_geology_data(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        image_dir=args.image_dir,
    )

    print("\n适配完成！")
    print(f"总记录数: {stats['total_records']}")
    print(f"有效记录: {stats['valid_records']}")
    print(f"无效记录: {stats['invalid_records']}")

    if stats["errors"]:
        print(f"\n前 5 个错误:")
        for error in stats["errors"][:5]:
            print(f"  - {error}")

    print(f"\n输出文件: {args.output_jsonl}")


if __name__ == "__main__":
    main()
