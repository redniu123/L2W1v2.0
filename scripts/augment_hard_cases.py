#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ Stage 2: Hard Case Synthesis via Boundary Cropping

策略：
  随机挑选 20% 的图像，在左右边缘向内强行裁剪 2%-5% 的像素，
  但不改变对应的 Ground Truth，人工制造边界截断样本（正样本）。

输出：
  - data/geo/geotext_augmented.jsonl  (原始 + 增强样本)
  - data/geo/geotext_aug/             (裁剪后的图像)
"""

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def crop_boundary(
    img: np.ndarray,
    left_ratio: float,
    right_ratio: float,
) -> np.ndarray:
    """
    向内裁剪图像左右边缘

    Args:
        img: 原始图像 (H, W, C)
        left_ratio:  左侧裁剪比例 (e.g. 0.03 = 裁掉左边 3%)
        right_ratio: 右侧裁剪比例 (e.g. 0.04 = 裁掉右边 4%)

    Returns:
        裁剪后的图像
    """
    H, W = img.shape[:2]
    left_px = max(1, int(W * left_ratio))
    right_px = max(1, int(W * right_ratio))
    right_end = W - right_px
    if right_end <= left_px:
        return img  # 防止过度裁剪
    return img[:, left_px:right_end]


def augment_hard_cases(
    input_jsonl: str,
    output_jsonl: str,
    aug_image_dir: str,
    augment_ratio: float = 0.20,
    crop_min: float = 0.02,
    crop_max: float = 0.05,
    seed: int = 42,
) -> None:
    """
    生成边界裁剪增强样本

    Args:
        input_jsonl:   原始 JSONL 文件路径
        output_jsonl:  输出 JSONL 文件路径（原始 + 增强）
        aug_image_dir: 增强图像保存目录
        augment_ratio: 随机挑选比例（默认 20%）
        crop_min:      最小裁剪比例（默认 2%）
        crop_max:      最大裁剪比例（默认 5%）
        seed:          随机种子
    """
    random.seed(seed)
    np.random.seed(seed)

    input_path = Path(input_jsonl)
    output_path = Path(output_jsonl)
    aug_dir = Path(aug_image_dir)
    aug_dir.mkdir(parents=True, exist_ok=True)
    data_root = input_path.resolve().parent

    # 读取原始样本
    samples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"原始样本数: {len(samples)}")

    # 随机挑选需要增强的样本索引
    n_aug = int(len(samples) * augment_ratio)
    aug_indices = set(random.sample(range(len(samples)), n_aug))
    print(f"将对 {n_aug} 张图像（{augment_ratio:.0%}）进行边界裁剪增强")

    augmented_samples = []
    skip_count = 0

    for i, sample in enumerate(tqdm(samples, desc="Augmenting")):
        if i not in aug_indices:
            continue

        # 兼容多种字段名
        image_path = sample.get("image") or sample.get("image_path")
        T_GT = sample.get("gt_text") or sample.get("text") or sample.get("label", "")

        if not image_path or not T_GT:
            skip_count += 1
            continue

        # 解析绝对路径
        img_path = Path(image_path)
        if not img_path.is_absolute():
            img_path = data_root / img_path

        img = cv2.imread(str(img_path))
        if img is None:
            skip_count += 1
            continue

        # 随机决定裁剪侧（左、右、或双侧）
        side = random.choice(["left", "right", "both"])
        left_ratio = 0.0
        right_ratio = 0.0

        if side in ("left", "both"):
            left_ratio = random.uniform(crop_min, crop_max)
        if side in ("right", "both"):
            right_ratio = random.uniform(crop_min, crop_max)

        cropped = crop_boundary(img, left_ratio, right_ratio)

        # 保存增强图像
        stem = img_path.stem
        aug_filename = f"{stem}_aug_L{left_ratio:.3f}_R{right_ratio:.3f}.jpg"
        # aug_dir 使用绝对路径，避免相对路径混乱
        aug_dir_abs = Path(aug_image_dir).resolve()
        aug_dir_abs.mkdir(parents=True, exist_ok=True)
        aug_save_path = aug_dir_abs / aug_filename
        cv2.imwrite(str(aug_save_path), cropped)

        # 构造新样本（GT 不变，图像路径更新为裁剪版本）
        new_sample = dict(sample)
        # 使用相对于 data_root 的路径，确保 prepare_calibration_data.py 能正确解析
        try:
            rel_path = aug_save_path.relative_to(data_root)
        except ValueError:
            # 若无法相对化，使用绝对路径
            rel_path = aug_save_path
        new_sample["image"] = str(rel_path)
        new_sample["augmented"] = True
        new_sample["crop_left"] = round(left_ratio, 4)
        new_sample["crop_right"] = round(right_ratio, 4)
        new_sample["crop_side"] = side
        # 强制标记为正样本：裁剪图像必然存在边界截断，GT 不变即构成漏字
        new_sample["force_positive"] = True

        augmented_samples.append(new_sample)

    print(f"\n成功生成增强样本: {len(augmented_samples)} 条 (跳过: {skip_count})")

    # 写出合并 JSONL（原始 + 增强）
    all_samples = samples + augmented_samples
    with open(output_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n✓ 合并数据集已保存: {output_path}")
    print(f"  原始样本: {len(samples)}")
    print(f"  增强样本: {len(augmented_samples)}")
    print(f"  合计:     {len(all_samples)}")
    print(f"\n预期正样本增量: +{len(augmented_samples)} 条")
    new_pos_ratio = (len(augmented_samples) + int(len(samples) * 0.0293)) / len(all_samples)
    print(f"预期新正样本比例: ~{new_pos_ratio:.2%}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SH-DA++ Stage 2: Hard Case Synthesis")
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default="data/geo/geotext.jsonl",
        help="原始 JSONL 文件路径",
    )
    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="data/geo/geotext_augmented.jsonl",
        help="输出 JSONL 文件路径（原始 + 增强）",
    )
    parser.add_argument(
        "--aug_image_dir",
        type=str,
        default="data/geo/geotext_aug",
        help="增强图像保存目录",
    )
    parser.add_argument(
        "--augment_ratio",
        type=float,
        default=0.20,
        help="随机挑选比例（默认 0.20 = 20%%）",
    )
    parser.add_argument(
        "--crop_min",
        type=float,
        default=0.02,
        help="最小裁剪比例（默认 2%%）",
    )
    parser.add_argument(
        "--crop_max",
        type=float,
        default=0.05,
        help="最大裁剪比例（默认 5%%）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    augment_hard_cases(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        aug_image_dir=args.aug_image_dir,
        augment_ratio=args.augment_ratio,
        crop_min=args.crop_min,
        crop_max=args.crop_max,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
