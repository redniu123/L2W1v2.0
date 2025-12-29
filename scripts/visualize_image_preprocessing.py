#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
可视化图像预处理流程 - 模拟 PP-OCRv5 压缩效果

用于理解图像在输入识别模型前的预处理过程
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import math

try:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARNING] matplotlib 未安装，将使用 OpenCV 显示")


def simulate_paddleocr_preprocessing(
    img: np.ndarray,
    rec_image_shape: tuple = (3, 48, 320),
    padding: int = 20,
    use_mean_padding: bool = True,
) -> tuple:
    """
    模拟 PaddleOCR 的预处理流程

    Args:
        img: 输入图像 (BGR 格式)
        rec_image_shape: 识别模型输入形状 (C, H, W)
        padding: padding 像素数
        use_mean_padding: 是否使用平均像素值填充

    Returns:
        (original_shape, padded_img, final_img, compression_info)
    """
    original_h, original_w = img.shape[:2]
    imgC, imgH, imgW = rec_image_shape

    # ============================================================
    # 步骤 1: 添加 Padding
    # ============================================================
    if padding > 0:
        if use_mean_padding:
            # 使用平均像素值填充
            mean_pixel = tuple(int(v) for v in cv2.mean(img)[:3])
            padding_value = mean_pixel
        else:
            # 使用白色填充
            padding_value = (255, 255, 255)

        img_padded = cv2.copyMakeBorder(
            img,
            top=padding,
            bottom=padding,
            left=padding,
            right=padding,
            borderType=cv2.BORDER_CONSTANT,
            value=padding_value,
        )
    else:
        img_padded = img.copy()
        padding_value = None

    padded_h, padded_w = img_padded.shape[:2]

    # ============================================================
    # 步骤 2: 压缩高度到固定值 (imgH = 48)
    # ============================================================
    # 计算宽高比
    ratio = padded_w / float(padded_h)

    # 计算目标宽度（保持宽高比，但不超过 imgW）
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW  # 超过最大宽度，截断
        is_truncated = True
    else:
        resized_w = int(math.ceil(imgH * ratio))
        is_truncated = False

    # 执行 resize
    img_resized = cv2.resize(
        img_padded, (resized_w, imgH), interpolation=cv2.INTER_LINEAR
    )

    # ============================================================
    # 步骤 3: 填充到固定宽度 (imgW = 320)
    # ============================================================
    if resized_w < imgW:
        # 右侧零填充
        padding_im = np.zeros((imgH, imgW, 3), dtype=np.float32)
        padding_im[:, :resized_w, :] = img_resized.astype(np.float32)
        img_final = padding_im.astype(np.uint8)
        has_zero_padding = True
    else:
        img_final = img_resized
        has_zero_padding = False

    # ============================================================
    # 计算压缩信息
    # ============================================================
    compression_info = {
        "original_shape": (original_w, original_h),
        "padded_shape": (padded_w, padded_h),
        "final_shape": (imgW, imgH),
        "resized_width": resized_w,
        "height_compression_ratio": imgH / padded_h,
        "width_compression_ratio": resized_w / padded_w
        if not is_truncated
        else imgW / padded_w,
        "is_truncated": is_truncated,
        "has_zero_padding": has_zero_padding,
        "padding_value": padding_value,
    }

    return (original_h, original_w), img_padded, img_final, compression_info


def visualize_comparison(
    original_img: np.ndarray,
    padded_img: np.ndarray,
    final_img: np.ndarray,
    compression_info: dict,
    save_path: str = None,
):
    """可视化对比结果"""

    if HAS_MATPLOTLIB:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 原始图像
        axes[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title(
            f"原始图像\n尺寸: {compression_info['original_shape'][0]}×{compression_info['original_shape'][1]}",
            fontsize=12,
        )
        axes[0].axis("off")

        # Padding 后
        axes[1].imshow(cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB))
        axes[1].set_title(
            f"Padding 后\n尺寸: {compression_info['padded_shape'][0]}×{compression_info['padded_shape'][1]}\n"
            f"Padding: 20px (RGB{compression_info['padding_value']})",
            fontsize=12,
        )
        axes[1].axis("off")

        # 最终压缩后
        axes[2].imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
        title = f"最终输入模型\n尺寸: {compression_info['final_shape'][0]}×{compression_info['final_shape'][1]}\n"
        if compression_info["is_truncated"]:
            title += "⚠️ 宽度被截断！\n"
        if compression_info["has_zero_padding"]:
            title += "⚠️ 右侧零填充\n"
        title += f"高度压缩比: {compression_info['height_compression_ratio']:.3f}\n"
        title += f"宽度压缩比: {compression_info['width_compression_ratio']:.3f}"
        axes[2].set_title(
            title,
            fontsize=12,
            color="red" if compression_info["is_truncated"] else "black",
        )
        axes[2].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[INFO] 结果已保存至: {save_path}")

        plt.show()

    else:
        # 使用 OpenCV 显示
        print("\n" + "=" * 60)
        print("图像预处理结果")
        print("=" * 60)
        print(
            f"原始尺寸: {compression_info['original_shape'][0]}×{compression_info['original_shape'][1]}"
        )
        print(
            f"Padding 后: {compression_info['padded_shape'][0]}×{compression_info['padded_shape'][1]}"
        )
        print(
            f"最终尺寸: {compression_info['final_shape'][0]}×{compression_info['final_shape'][1]}"
        )
        print(f"高度压缩比: {compression_info['height_compression_ratio']:.3f}")
        print(f"宽度压缩比: {compression_info['width_compression_ratio']:.3f}")
        if compression_info["is_truncated"]:
            print("⚠️  警告: 宽度被截断！")
        if compression_info["has_zero_padding"]:
            print("⚠️  警告: 右侧有零填充")
        print("=" * 60)

        # 水平拼接显示
        h1, w1 = original_img.shape[:2]
        h2, w2 = padded_img.shape[:2]
        h3, w3 = final_img.shape[:2]

        max_h = max(h1, h2, h3)

        # 调整高度一致
        img1_resized = cv2.resize(original_img, (int(w1 * max_h / h1), max_h))
        img2_resized = cv2.resize(padded_img, (int(w2 * max_h / h2), max_h))
        img3_resized = cv2.resize(final_img, (int(w3 * max_h / h3), max_h))

        # 拼接
        combined = np.hstack([img1_resized, img2_resized, img3_resized])

        cv2.imshow("原始 | Padding后 | 最终压缩", combined)
        print("\n按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="可视化 PP-OCRv5 图像预处理流程")
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="输入图像路径（HWDB 测试图像）",
    )
    parser.add_argument(
        "--rec_image_shape",
        type=str,
        default="3,48,320",
        help="识别模型输入形状 (C,H,W)，默认 3,48,320",
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=20,
        help="Padding 像素数，默认 20",
    )
    parser.add_argument(
        "--no_mean_padding",
        action="store_true",
        help="使用白色填充而非平均像素值",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="保存对比图像路径（如 comparison.png）",
    )

    args = parser.parse_args()

    # 解析 rec_image_shape
    rec_image_shape = tuple(int(x) for x in args.rec_image_shape.split(","))
    if len(rec_image_shape) != 3:
        raise ValueError("rec_image_shape 必须是 3 个值 (C,H,W)")

    # 读取图像
    image_path = Path(args.image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"无法读取图像: {image_path}")

    print(f"[INFO] 读取图像: {image_path}")
    print(f"[INFO] 原始尺寸: {img.shape[1]}×{img.shape[0]}")
    print(
        f"[INFO] 模型输入形状: {rec_image_shape[0]}×{rec_image_shape[1]}×{rec_image_shape[2]}"
    )
    print(
        f"[INFO] Padding: {args.padding}px ({'平均像素值' if not args.no_mean_padding else '白色'})"
    )
    print()

    # 执行预处理模拟
    original_shape, padded_img, final_img, compression_info = (
        simulate_paddleocr_preprocessing(
            img,
            rec_image_shape=rec_image_shape,
            padding=args.padding,
            use_mean_padding=not args.no_mean_padding,
        )
    )

    # 打印详细信息
    print("=" * 60)
    print("预处理步骤详情")
    print("=" * 60)
    print(
        f"1. 原始图像: {compression_info['original_shape'][0]}×{compression_info['original_shape'][1]}"
    )
    print(
        f"2. Padding 后: {compression_info['padded_shape'][0]}×{compression_info['padded_shape'][1]}"
    )
    if compression_info["padding_value"]:
        print(f"   Padding 颜色: RGB{compression_info['padding_value']}")
    print(
        f"3. 压缩高度: {compression_info['padded_shape'][1]} → {rec_image_shape[1]} (压缩比: {compression_info['height_compression_ratio']:.3f})"
    )
    print(
        f"4. 压缩宽度: {compression_info['padded_shape'][0]} → {compression_info['resized_width']} (压缩比: {compression_info['width_compression_ratio']:.3f})"
    )
    if compression_info["is_truncated"]:
        print(f"   ⚠️  警告: 宽度超过最大限制 {rec_image_shape[2]}，已被截断！")
    if compression_info["has_zero_padding"]:
        print(
            f"5. 右侧零填充: {compression_info['resized_width']} → {rec_image_shape[2]}"
        )
    print(
        f"6. 最终输入模型: {compression_info['final_shape'][0]}×{compression_info['final_shape'][1]}"
    )
    print("=" * 60)

    # 可视化
    visualize_comparison(img, padded_img, final_img, compression_info, args.output)


if __name__ == "__main__":
    main()
