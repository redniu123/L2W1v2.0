#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ Stage 2: Dynamic Adversarial Cropping

学术级 Hard Case 合成方案：
  步进式裁剪 + Agent A 真实验证。
  只保留 Agent A 真实发生边界漏字的图像，标签由 Levenshtein 对齐算法得出。
  绝对禁止 force_positive 机制。

算法流程：
  针对每张选中的图像，依次尝试 crop_ratio ∈ [0.03, 0.06, 0.09, 0.12, 0.15]：
    1. 对图像进行边界裁剪
    2. 让 Agent A 进行真实推理
    3. 用 Levenshtein 判断是否发生了真实的边界漏字
    4. 只保留真实漏字的图像，立即停止当前图像的步进
    5. 若裁剪 15% 后仍未漏字，放弃该图像

输出：
  - data/geo/geotext_augmented.jsonl  (原始 + 有效增强样本)
  - data/geo/geotext_aug/             (裁剪后的图像，均为真实 Hard Case)
"""

import json
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import Levenshtein


# 步进裁剪比例序列（从小到大，逐步加深）
CROP_STEPS = [0.03, 0.06, 0.09, 0.12, 0.15]


def apply_crop(
    img: np.ndarray,
    crop_ratio: float,
    side: str,
) -> np.ndarray:
    """
    对图像进行单侧或双侧边界裁剪

    Args:
        img: 原始图像 (H, W, C)
        crop_ratio: 裁剪比例
        side: 'left' | 'right' | 'both'

    Returns:
        裁剪后的图像
    """
    H, W = img.shape[:2]
    left_px = 0
    right_px = W

    if side in ("left", "both"):
        left_px = max(1, int(W * crop_ratio))
    if side in ("right", "both"):
        right_px = min(W - 1, W - int(W * crop_ratio))

    if right_px <= left_px:
        return img  # 防止过度裁剪
    return img[:, left_px:right_px]


def check_boundary_deletion(T_A: str, T_GT: str, K: int = 2) -> bool:
    """
    使用 Levenshtein 判断是否发生了边界漏字

    Args:
        T_A:  Agent A 识别文本
        T_GT: Ground Truth 文本
        K:    边界窗口大小

    Returns:
        True 表示发生了边界漏字
    """
    if not T_A or not T_GT:
        return False
    # 如果识别结果变成乱码（长度差异过大），放弃
    if len(T_A) == 0 or abs(len(T_A) - len(T_GT)) > len(T_GT) * 0.5:
        return False

    ops = Levenshtein.editops(T_A, T_GT)
    for op_type, pos_A, pos_GT in ops:
        if op_type == "delete":
            if pos_GT < K or pos_GT >= len(T_GT) - K:
                return True
    return False


def synthesize_hard_case(
    img: np.ndarray,
    T_GT: str,
    recognizer,
    side: str,
    K: int = 2,
) -> Tuple[bool, Optional[np.ndarray], float]:
    """
    动态对抗裁剪：步进式尝试，直到 Agent A 真实漏字

    Args:
        img:        原始图像
        T_GT:       Ground Truth 文本
        recognizer: TextRecognizerWithLogits 实例
        side:       裁剪侧 ('left' | 'right' | 'both')
        K:          边界窗口大小

    Returns:
        (success, cropped_img, crop_ratio)
        success=True 表示找到了真实漏字的裁剪版本
    """
    for crop_ratio in CROP_STEPS:
        cropped = apply_crop(img, crop_ratio, side)

        # Agent A 真实推理
        try:
            output = recognizer([cropped])
            if not output or not output.get("results"):
                continue
            T_A, conf = output["results"][0]
        except Exception:
            continue

        # Levenshtein 验证：是否真实发生了边界漏字
        if check_boundary_deletion(T_A, T_GT, K=K):
            return True, cropped, crop_ratio

    # 所有步进均未触发漏字，放弃该图像
    return False, None, 0.0


def augment_hard_cases(
    input_jsonl: str,
    output_jsonl: str,
    aug_image_dir: str,
    augment_ratio: float = 0.20,
    recognizer_args=None,
    K: int = 2,
    seed: int = 42,
) -> None:
    """
    生成真实 Hard Case 增强样本

    Args:
        input_jsonl:      原始 JSONL 文件路径
        output_jsonl:     输出 JSONL 路径（原始 + 增强）
        aug_image_dir:    增强图像保存目录
        augment_ratio:    候选图像比例（默认 20%）
        recognizer_args:  TextRecognizerWithLogits 参数对象
        K:                边界窗口大小
        seed:             随机种子
    """
    from modules.paddle_engine.predict_rec_modified import TextRecognizerWithLogits

    random.seed(seed)
    np.random.seed(seed)

    input_path = Path(input_jsonl)
    output_path = Path(output_jsonl)
    aug_dir = Path(aug_image_dir).resolve()
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

    # 初始化 Agent A
    print("初始化 Agent A (TextRecognizerWithLogits)...")
    recognizer = TextRecognizerWithLogits(recognizer_args)

    # 随机挑选候选图像
    n_candidates = int(len(samples) * augment_ratio)
    candidate_indices = set(random.sample(range(len(samples)), n_candidates))
    print(f"候选图像: {n_candidates} 张（{augment_ratio:.0%}），逐步裁剪至真实漏字")
    print(f"裁剪步进序列: {[f'{r:.0%}' for r in CROP_STEPS]}")

    augmented_samples = []
    tried = 0
    success = 0
    failed = 0

    for i, sample in enumerate(tqdm(samples, desc="Dynamic Adversarial Cropping")):
        if i not in candidate_indices:
            continue

        tried += 1
        image_path = sample.get("image") or sample.get("image_path")
        T_GT = sample.get("gt_text") or sample.get("text") or sample.get("label", "")

        if not image_path or not T_GT:
            failed += 1
            continue

        img_path = Path(image_path)
        if not img_path.is_absolute():
            img_path = data_root / img_path

        img = cv2.imread(str(img_path))
        if img is None:
            failed += 1
            continue

        # 随机选择裁剪侧
        side = random.choice(["left", "right", "both"])

        # 动态对抗裁剪
        found, cropped_img, crop_ratio = synthesize_hard_case(
            img, T_GT, recognizer, side, K=K
        )

        if not found:
            failed += 1
            continue

        # 保存有效的 Hard Case 图像
        stem = img_path.stem
        aug_filename = f"{stem}_adv_{side}_{crop_ratio:.3f}.jpg"
        aug_save_path = aug_dir / aug_filename
        cv2.imwrite(str(aug_save_path), cropped_img)

        # 构造新样本
        try:
            rel_path = aug_save_path.relative_to(data_root)
        except ValueError:
            rel_path = aug_save_path

        new_sample = dict(sample)
        new_sample["image"] = str(rel_path)
        new_sample["augmented"] = True
        new_sample["crop_side"] = side
        new_sample["crop_ratio"] = round(crop_ratio, 4)
        # 无 force_positive：标签完全由 prepare_calibration_data.py 中的
        # generate_deletion_label 通过 Levenshtein 对齐真实计算得出

        augmented_samples.append(new_sample)
        success += 1

    print(f"\n动态对抗裁剪完成:")
    print(f"  尝试: {tried} 张")
    print(f"  成功（真实漏字）: {success} 张")
    print(f"  失败（未能触发漏字）: {failed} 张")
    print(f"  成功率: {success / tried:.1%}" if tried > 0 else "")

    # 写出合并 JSONL
    all_samples = samples + augmented_samples
    with open(output_path, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n✓ 合并数据集已保存: {output_path}")
    print(f"  原始样本: {len(samples)}")
    print(f"  有效增强样本: {success}")
    print(f"  合计: {len(all_samples)}")
    orig_pos = int(len(samples) * 0.0293)
    new_pos_ratio = (orig_pos + success) / len(all_samples)
    print(f"  预期正样本比例: ~{new_pos_ratio:.2%}")


def main():
    import argparse
    from tools.infer.utility import init_args

    parser = init_args()
    parser.add_argument("--input_jsonl", type=str, default="data/geo/geotext.jsonl")
    parser.add_argument("--output_jsonl", type=str, default="data/geo/geotext_augmented.jsonl")
    parser.add_argument("--aug_image_dir", type=str, default="data/geo/geotext_aug")
    parser.add_argument("--augment_ratio", type=float, default=0.20)
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_det", action="store_true")
    parser.add_argument("--det_model_dir", type=str, default="")

    args = parser.parse_args()

    if args.rec_model_dir is None:
        args.rec_model_dir = "./models/agent_a_ppocr/PP-OCRv5_server_rec_infer"

    augment_hard_cases(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        aug_image_dir=args.aug_image_dir,
        augment_ratio=args.augment_ratio,
        recognizer_args=args,
        K=args.K,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
