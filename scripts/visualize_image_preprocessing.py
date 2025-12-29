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
import sys

try:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("[WARNING] matplotlib 未安装，将使用 OpenCV 显示")

# 尝试导入 PaddleOCR（可选）
try:
    from paddleocr import PaddleOCR

    HAS_PADDLEOCR = True
except ImportError:
    HAS_PADDLEOCR = False
    print("[WARNING] PaddleOCR 未安装，将跳过 OCR 置信度分析")


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


def run_ocr_analysis(
    img: np.ndarray,
    use_det: bool = False,
    ocr_instance: object = None,
) -> dict:
    """
    运行 OCR 推理并提取置信度信息

    Args:
        img: 输入图像
        use_det: 是否使用检测器
        ocr_instance: PaddleOCR 实例（如果为 None 则创建新的）

    Returns:
        dict: {
            'pred_text': str,
            'char_confidences': List[Dict],  # [{'char': c, 'score': s}, ...]
            'avg_confidence': float,
            'ocr_result': raw OCR result,
        }
    """
    if not HAS_PADDLEOCR:
        return None

    try:
        if ocr_instance is None:
            # 临时创建 OCR 实例（较慢，建议复用）
            ocr_params = {
                "ocr_version": "PP-OCRv5",
                "lang": "ch",
                "device": "gpu",
                "use_textline_orientation": True,
            }
            # 新版本可能不支持 show_log，尝试添加
            try:
                ocr_instance = PaddleOCR(**ocr_params, show_log=False)
            except TypeError:
                ocr_instance = PaddleOCR(**ocr_params)

        # 运行 OCR
        result = None
        if hasattr(ocr_instance, "ocr"):
            try:
                result = ocr_instance.ocr(img, det=use_det, rec=True)
            except TypeError:
                result = ocr_instance.predict(img)
        else:
            result = ocr_instance.predict(img)

        if result is None or len(result) == 0:
            return None

        # 解析结果
        pred_text = ""
        char_confidences = []
        avg_confidence = 0.0

        # 格式1: [{'rec_texts': [...], 'rec_scores': [...], ...}]
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            first_item = result[0]
            if "rec_texts" in first_item and "rec_scores" in first_item:
                rec_texts = first_item.get("rec_texts", [])
                rec_scores = first_item.get("rec_scores", [])
                pred_text = "".join(rec_texts)

                # 构建字符级置信度（每个文本块的平均置信度分配给每个字符）
                for text_idx, text in enumerate(rec_texts):
                    score = rec_scores[text_idx] if text_idx < len(rec_scores) else 0.0
                    for char in text:
                        char_confidences.append({"char": char, "score": float(score)})

                avg_confidence = (
                    float(sum(rec_scores) / len(rec_scores)) if rec_scores else 0.0
                )

        # 格式2: 旧版格式 [[box, (text, conf)], ...]
        elif isinstance(result, list) and len(result) > 0:
            texts = []
            scores = []
            for item in result:
                if isinstance(item, list) and len(item) >= 2:
                    text_info = item[1]
                    if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                        text = str(text_info[0])
                        conf = float(text_info[1])
                        texts.append(text)
                        scores.append(conf)

            pred_text = "".join(texts)
            for text, score in zip(texts, scores):
                for char in text:
                    char_confidences.append({"char": char, "score": score})
            avg_confidence = float(sum(scores) / len(scores)) if scores else 0.0

        return {
            "pred_text": pred_text,
            "char_confidences": char_confidences,
            "avg_confidence": avg_confidence,
            "ocr_result": result,
        }

    except Exception as e:
        print(f"[WARNING] OCR 推理失败: {e}")
        return None


def visualize_comparison(
    original_img: np.ndarray,
    padded_img: np.ndarray,
    final_img: np.ndarray,
    compression_info: dict,
    save_path: str = None,
    ocr_analysis: dict = None,
    gt_text: str = None,
):
    """可视化对比结果，包括 OCR 置信度分析"""

    has_ocr = ocr_analysis is not None and ocr_analysis.get("pred_text")

    if HAS_MATPLOTLIB:
        # 根据是否有 OCR 结果调整布局
        if has_ocr:
            fig = plt.figure(figsize=(20, 10))
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
            axes_img = [fig.add_subplot(gs[0, i]) for i in range(3)]
            axes_conf = fig.add_subplot(gs[1, :])
        else:
            fig, axes_img = plt.subplots(1, 3, figsize=(18, 6))

        # 原始图像
        axes_img[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        axes_img[0].set_title(
            f"原始图像\n尺寸: {compression_info['original_shape'][0]}×{compression_info['original_shape'][1]}",
            fontsize=12,
        )
        axes_img[0].axis("off")

        # Padding 后
        axes_img[1].imshow(cv2.cvtColor(padded_img, cv2.COLOR_BGR2RGB))
        axes_img[1].set_title(
            f"Padding 后\n尺寸: {compression_info['padded_shape'][0]}×{compression_info['padded_shape'][1]}\n"
            f"Padding: 20px (RGB{compression_info['padding_value']})",
            fontsize=12,
        )
        axes_img[1].axis("off")

        # 最终压缩后
        axes_img[2].imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
        title = f"最终输入模型\n尺寸: {compression_info['final_shape'][0]}×{compression_info['final_shape'][1]}\n"
        if compression_info["is_truncated"]:
            title += "⚠️ 宽度被截断！\n"
        if compression_info["has_zero_padding"]:
            title += "⚠️ 右侧零填充\n"
        title += f"高度压缩比: {compression_info['height_compression_ratio']:.3f}\n"
        title += f"宽度压缩比: {compression_info['width_compression_ratio']:.3f}"
        axes_img[2].set_title(
            title,
            fontsize=12,
            color="red" if compression_info["is_truncated"] else "black",
        )
        axes_img[2].axis("off")

        # OCR 置信度可视化
        if has_ocr:
            char_confidences = ocr_analysis["char_confidences"]
            pred_text = ocr_analysis["pred_text"]
            avg_conf = ocr_analysis["avg_confidence"]

            # 提取字符和置信度
            chars = [item["char"] for item in char_confidences]
            confidences = [item["score"] for item in char_confidences]

            # 创建字符位置索引
            x_positions = list(range(len(chars)))

            # 绘制置信度条形图
            colors = [
                "green" if c >= 0.8 else "orange" if c >= 0.6 else "red"
                for c in confidences
            ]
            bars = axes_conf.bar(
                x_positions,
                confidences,
                color=colors,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
            )

            # 添加平均置信度线
            axes_conf.axhline(
                y=avg_conf,
                color="blue",
                linestyle="--",
                linewidth=2,
                label=f"平均置信度: {avg_conf:.3f}",
            )

            # 标注低置信度区域（边缘字符）
            if len(chars) > 0:
                # 标注前3个和后3个字符（边缘字符）
                edge_indices = list(range(min(3, len(chars)))) + list(
                    range(max(0, len(chars) - 3), len(chars))
                )
                for idx in edge_indices:
                    if idx < len(chars):
                        axes_conf.text(
                            idx,
                            confidences[idx] + 0.02,
                            chars[idx],
                            ha="center",
                            va="bottom",
                            fontsize=8,
                            fontweight="bold",
                        )

            axes_conf.set_xlabel("字符位置索引", fontsize=12)
            axes_conf.set_ylabel("置信度", fontsize=12)
            axes_conf.set_title(
                f'字符级置信度分析 | 预测文本: "{pred_text}"\n'
                f"字符数: {len(chars)} | 平均置信度: {avg_conf:.3f}",
                fontsize=13,
                fontweight="bold",
            )
            axes_conf.set_ylim([0, 1.1])
            axes_conf.grid(True, alpha=0.3, axis="y")
            axes_conf.legend(loc="upper right")

            # X 轴标签（每隔几个字符显示一次，避免过于密集）
            step = max(1, len(chars) // 20)
            axes_conf.set_xticks(x_positions[::step])
            axes_conf.set_xticklabels(x_positions[::step])

            # 如果有 GT 文本，标注缺失的字符
            if gt_text:
                gt_chars = list(gt_text)
                missing_info = []
                for i, gt_char in enumerate(gt_chars):
                    if i >= len(chars) or (i < len(chars) and chars[i] != gt_char):
                        missing_info.append((i, gt_char))

                if missing_info:
                    info_text = f"缺失/错误字符: {', '.join([f'位置{i}:{c}' for i, c in missing_info[:10]])}"
                    if len(missing_info) > 10:
                        info_text += "..."
                    axes_conf.text(
                        0.02,
                        0.98,
                        info_text,
                        transform=axes_conf.transAxes,
                        fontsize=10,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.5),
                    )

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
    parser.add_argument(
        "--run_ocr",
        action="store_true",
        help="运行 OCR 推理并显示字符级置信度分析",
    )
    parser.add_argument(
        "--use_det",
        action="store_true",
        help="OCR 时启用文本检测（默认仅识别）",
    )
    parser.add_argument(
        "--gt_text",
        type=str,
        default=None,
        help="真值文本（用于对比分析）",
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

    # 运行 OCR 分析（如果启用）
    ocr_analysis = None
    if args.run_ocr:
        if not HAS_PADDLEOCR:
            print("\n[WARNING] PaddleOCR 未安装，跳过 OCR 分析")
        else:
            print("\n" + "=" * 60)
            print("运行 OCR 推理分析...")
            print("=" * 60)
            # 使用 padding 后的图像进行 OCR（与实际推理一致）
            ocr_analysis = run_ocr_analysis(padded_img, use_det=args.use_det)
            if ocr_analysis:
                print(f"预测文本: {ocr_analysis['pred_text']}")
                print(f"字符数: {len(ocr_analysis['char_confidences'])}")
                print(f"平均置信度: {ocr_analysis['avg_confidence']:.4f}")

                # 分析边缘字符置信度
                char_confs = ocr_analysis["char_confidences"]
                if len(char_confs) > 0:
                    edge_window = min(3, len(char_confs) // 2)
                    left_confs = [c["score"] for c in char_confs[:edge_window]]
                    right_confs = [c["score"] for c in char_confs[-edge_window:]]
                    center_confs = (
                        [c["score"] for c in char_confs[edge_window:-edge_window]]
                        if len(char_confs) > 2 * edge_window
                        else []
                    )

                    print(f"\n边缘字符置信度分析:")
                    if left_confs:
                        print(
                            f"  左侧 {edge_window} 个字符: 平均 {np.mean(left_confs):.4f}, 最小 {np.min(left_confs):.4f}"
                        )
                    if right_confs:
                        print(
                            f"  右侧 {edge_window} 个字符: 平均 {np.mean(right_confs):.4f}, 最小 {np.min(right_confs):.4f}"
                        )
                    if center_confs:
                        print(
                            f"  中间字符: 平均 {np.mean(center_confs):.4f}, 最小 {np.min(center_confs):.4f}"
                        )

                if args.gt_text:
                    print(f"\n真值文本: {args.gt_text}")
                    print(
                        f"文本匹配: {'✓' if ocr_analysis['pred_text'] == args.gt_text else '✗'}"
                    )
            else:
                print("[WARNING] OCR 推理失败或返回空结果")

    # 可视化
    visualize_comparison(
        img,
        padded_img,
        final_img,
        compression_info,
        args.output,
        ocr_analysis=ocr_analysis,
        gt_text=args.gt_text,
    )


if __name__ == "__main__":
    main()
