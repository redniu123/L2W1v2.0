#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成地质专项元数据 (jsonl)

功能:
- 遍历指定目录下所有图像文件
- 输出 jsonl: {"id": "geo_00x", "image_path": "...", "gt_text": ""}
"""

import argparse
import json
from pathlib import Path


def iter_images(root_dir: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    for path in root_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def main():
    parser = argparse.ArgumentParser(
        description="Generate geo metadata jsonl for SH-DA++ v4.0"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/geo/geotext",
        help="地质专项图像目录 (默认: data/geo/geotext)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/geo/geotext.jsonl",
        help="输出 jsonl 路径 (默认: data/geo/geotext.jsonl)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir 不存在: {input_dir}")

    images = sorted(iter_images(input_dir))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for idx, img_path in enumerate(images, start=1):
            record = {
                "id": f"geo_{idx:03d}",
                "image_path": img_path.as_posix(),
                "gt_text": "",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[OK] 生成 {len(images)} 条记录 -> {output_path}")


if __name__ == "__main__":
    main()
