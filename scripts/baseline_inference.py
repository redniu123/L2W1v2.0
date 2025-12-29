#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
L2W1 基准测试脚本 - PP-OCRv5 在 HWDB 测试集上的基准表现

功能:
1. 使用 PP-OCRv5 对 HWDB 测试集进行推理
2. 计算编辑距离、CER、字符级置信度等指标
3. 输出结构化的 JSONL 结果文件

Usage:
    python scripts/baseline_inference.py
    python scripts/baseline_inference.py --use_gpu
    python scripts/baseline_inference.py --metadata_path ./data/raw/HWDB_Benchmark/test_metadata.jsonl
# 方式1: 正常模式 (使用检测器 + 左右 25px padding)
python scripts/baseline_inference.py \
    --metadata_path ./data/raw/HWDB_Benchmark/test_metadata.jsonl \
    --image_root ./data/raw/HWDB_Benchmark/ \
    --output_path ./results/baseline_results.jsonl

# 方式2: 直接识别模式 (禁用检测器，适合行文本)
python scripts/baseline_inference.py \
    --metadata_path ./data/raw/HWDB_Benchmark/test_metadata.jsonl \
    --image_root ./data/raw/HWDB_Benchmark/ \
    --output_path ./results/baseline_results_nodet.jsonl \
    --no_det
Author: L2W1 Team
Date: 2024
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict

# 添加项目路径
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 第三方库导入
try:
    from tqdm import tqdm
except ImportError:
    print("[WARNING] tqdm 未安装，使用简单进度显示")

    def tqdm(iterable, **kwargs):
        total = kwargs.get(
            "total", len(iterable) if hasattr(iterable, "__len__") else None
        )
        desc = kwargs.get("desc", "Processing")
        for i, item in enumerate(iterable):
            if total:
                print(f"\r{desc}: {i + 1}/{total}", end="", flush=True)
            yield item
        print()


try:
    import Levenshtein
except ImportError:
    print("[ERROR] Levenshtein 库未安装，请运行: pip install python-Levenshtein")
    sys.exit(1)

try:
    from paddleocr import PaddleOCR
except ImportError:
    print("[ERROR] paddleocr 库未安装，请运行: pip install paddleocr")
    sys.exit(1)

try:
    import cv2
    import numpy as np
except ImportError:
    print("[ERROR] opencv-python 或 numpy 未安装")
    sys.exit(1)


# =============================================================================
# 配置常量
# =============================================================================

DEFAULT_METADATA_PATH = "data/raw/HWDB_Benchmark/test_metadata.jsonl"
DEFAULT_IMAGE_ROOT = "data/raw/HWDB_Benchmark/"
DEFAULT_OUTPUT_PATH = "results/baseline_results.jsonl"


# =============================================================================
# 数据结构
# =============================================================================


@dataclass
class InferenceResult:
    """单条推理结果"""

    id: str
    image_path: str
    gt_text: str
    pred_text: str
    avg_confidence: float
    char_confidences: List[Dict[str, Any]]  # [{"char": "字", "score": 0.95}, ...]
    edit_distance: int
    cer: float

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# 核心功能类
# =============================================================================


class BaselineInference:
    """
    PP-OCRv5 基准测试推理器

    提取模型在测试集上的基准表现和不确定性指标
    """

    def __init__(
        self,
        metadata_path: str = DEFAULT_METADATA_PATH,
        image_root: str = DEFAULT_IMAGE_ROOT,
        output_path: str = DEFAULT_OUTPUT_PATH,
        use_gpu: bool = True,
        det: bool = True,  # 是否使用检测模型
    ):
        """
        初始化推理器

        Args:
            metadata_path: 测试集元数据文件路径
            image_root: 图像根目录
            output_path: 输出结果路径
            use_gpu: 是否使用 GPU
            det: 是否启用文本检测（对于行文本图像可设为 False）
        """
        self.metadata_path = Path(metadata_path)
        self.image_root = Path(image_root)
        self.output_path = Path(output_path)
        self.use_gpu = use_gpu
        self.use_det = det

        # 创建输出目录
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # 统计信息
        self.stats = {
            "total_samples": 0,
            "success_samples": 0,
            "failed_samples": 0,
            "total_cer": 0.0,
            "total_confidence": 0.0,
            "total_edit_distance": 0,
        }

        # 初始化 OCR 模型
        self._init_ocr()

    def _init_ocr(self):
        """初始化 PP-OCRv5 模型"""
        print("=" * 60)
        print("初始化 PP-OCRv5 模型...")
        print("=" * 60)

        try:
            # 新版 PaddleOCR API 参数精简
            # - device: "gpu" 或 "cpu" (代替 use_gpu)
            # - use_textline_orientation: 代替已废弃的 use_angle_cls
            # - 新版移除了 show_log 参数
            device = "gpu" if self.use_gpu else "cpu"

            # 构建初始化参数
            # 注意：新版 PaddleOCR API 精简了参数，rec_image_shape 不再支持
            # 识别分辨率由模型内部决定，我们通过 padding 策略弥补边界丢失
            ocr_params = {
                "ocr_version": "PP-OCRv5",
                "lang": "ch",
                "device": device,
                "use_textline_orientation": True,
            }

            # 如果启用检测，增加 unclip ratio 以捕获边界笔画
            if self.use_det:
                ocr_params["text_det_unclip_ratio"] = 2.5  # 新版 API 参数名

            self.ocr = PaddleOCR(**ocr_params)
            print("[INFO] PP-OCRv5 模型初始化成功")
            print(f"[INFO] 设备: {device.upper()}")
            print(
                f"[INFO] 文本检测: {'启用 (unclip_ratio=2.5)' if self.use_det else '禁用（直接识别）'}"
            )
            print()
        except Exception as e:
            print(f"[ERROR] 模型初始化失败: {e}")
            raise

    def _load_metadata(self) -> List[Dict]:
        """
        加载测试集元数据

        Returns:
            元数据列表
        """
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"元数据文件不存在: {self.metadata_path}")

        samples = []
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                    samples.append(item)
                except json.JSONDecodeError as e:
                    print(f"[WARNING] 第 {line_num} 行 JSON 解析失败: {e}")
                    continue

        print(f"[INFO] 加载了 {len(samples)} 个测试样本")
        return samples

    def _resolve_image_path(self, image_path: str) -> Optional[Path]:
        """
        解析图像路径

        Args:
            image_path: 元数据中的图像路径

        Returns:
            完整的图像路径，如果不存在则返回 None
        """
        # 标准化路径
        image_path = os.path.normpath(image_path)

        # 尝试多种路径解析方式
        candidates = [
            Path(image_path),  # 直接路径
            self.image_root / image_path,  # 相对于图像根目录
            self.image_root / Path(image_path).name,  # 仅文件名
        ]

        # 如果路径已包含 image_root，避免重复拼接
        image_path_str = image_path.replace("\\", "/").strip("./")
        image_root_str = str(self.image_root).replace("\\", "/").strip("./")

        if image_path_str.startswith(image_root_str):
            candidates.insert(0, Path(image_path))

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return None

    def _run_ocr(
        self, image_path: Path, gt_text: str = ""
    ) -> Tuple[str, List[Dict[str, Any]], float]:
        """
        运行 OCR 推理

        Args:
            image_path: 图像路径
            gt_text: 真值文本 (用于长度检查警告)

        Returns:
            Tuple[pred_text, char_confidences, avg_confidence]
        """
        try:
            # ================================================================
            # 策略 A: 智能图像 Padding - 解决边缘字符丢失问题
            # 四周均匀填充 20px，使用原图平均像素值保证背景一致性
            # ================================================================
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"[WARNING] 无法读取图像: {image_path}")
                return "", [], 0.0

            original_h, original_w = img.shape[:2]

            # 计算原图的平均像素值作为填充色（保证背景一致性）
            mean_pixel = tuple(int(v) for v in cv2.mean(img)[:3])

            # 四周均匀填充 20px
            padding = 20
            img_padded = cv2.copyMakeBorder(
                img,
                top=padding,
                bottom=padding,
                left=padding,
                right=padding,
                borderType=cv2.BORDER_CONSTANT,
                value=mean_pixel,  # 使用平均像素值填充
            )

            padded_h, padded_w = img_padded.shape[:2]

            # 调试：打印前 5 个样本的尺寸信息
            if not hasattr(self, "_debug_count"):
                self._debug_count = 0

            if self._debug_count < 5:
                self._debug_count += 1
                print(f"\n[DEBUG #{self._debug_count}] 图像: {image_path.name}")
                print(f"  原始尺寸: {original_w}x{original_h}")
                print(f"  填充后尺寸: {padded_w}x{padded_h} (四周各+{padding}px)")
                print(f"  填充颜色: RGB{mean_pixel}")
                print(f"  检测模式: {'启用' if self.use_det else '禁用 (直接识别)'}")

            # ================================================================
            # 策略 B: 尝试多种 API 调用方式
            # ================================================================
            result = None

            # 方式1: 尝试使用旧版 ocr() API (支持 det 参数)
            if hasattr(self.ocr, "ocr"):
                try:
                    result = self.ocr.ocr(img_padded, det=self.use_det, rec=True)
                except TypeError:
                    # 如果 ocr() 不接受这些参数，fallback 到 predict()
                    pass

            # 方式2: 使用 predict() API
            if result is None:
                result = self.ocr.predict(img_padded)

            # 调试：打印第一个样本的结果结构
            if not hasattr(self, "_debug_printed"):
                self._debug_printed = True
                print(f"\n[DEBUG] OCR 返回结果类型: {type(result)}")
                if isinstance(result, list) and len(result) > 0:
                    print(f"[DEBUG] 第一个元素类型: {type(result[0])}")
                    if isinstance(result[0], dict):
                        print(f"[DEBUG] 字典键: {result[0].keys()}")
                print()

            if result is None or len(result) == 0:
                return "", [], 0.0

            # ================================================================
            # 格式1 (最新版 PaddleOCR/PaddleX):
            # [{'rec_texts': ['文本1', '文本2'], 'rec_scores': [0.95, 0.87], ...}]
            # ================================================================
            if (
                isinstance(result, list)
                and len(result) > 0
                and isinstance(result[0], dict)
            ):
                first_item = result[0]

                # 检查是否为最新版格式 (rec_texts / rec_scores)
                if "rec_texts" in first_item and "rec_scores" in first_item:
                    rec_texts = first_item.get("rec_texts", [])
                    rec_scores = first_item.get("rec_scores", [])

                    # 拼接所有文本
                    all_text = "".join(rec_texts) if rec_texts else ""

                    # 构建字符级置信度列表
                    all_confidences = []
                    for text_idx, text in enumerate(rec_texts):
                        # 获取对应的置信度（如果索引越界则取最后一个或默认值）
                        score = (
                            rec_scores[text_idx]
                            if text_idx < len(rec_scores)
                            else (rec_scores[-1] if rec_scores else 0.0)
                        )
                        score = float(score)
                        for char in text:
                            all_confidences.append(
                                {"char": char, "score": round(score, 4)}
                            )

                    # 计算平均置信度
                    avg_conf = (
                        float(sum(rec_scores) / len(rec_scores)) if rec_scores else 0.0
                    )

                    return all_text, all_confidences, round(avg_conf, 4)

                # 检查单条识别结果格式 (rec_text / rec_score)
                if "rec_text" in first_item or "text" in first_item:
                    text = first_item.get("rec_text", first_item.get("text", ""))
                    conf = float(
                        first_item.get(
                            "rec_score",
                            first_item.get("confidence", first_item.get("score", 0.0)),
                        )
                    )
                    if text:
                        char_confs = [
                            {"char": c, "score": round(conf, 4)} for c in text
                        ]
                        return text, char_confs, round(conf, 4)
                    return "", [], 0.0

            # ================================================================
            # 格式2 (Fallback - 旧版格式):
            # [[ [[box], (text, conf)], ... ]] 或 [ {text:..., confidence:...}, ... ]
            # ================================================================

            # 如果是嵌套列表，取第一层
            lines = (
                result[0]
                if (
                    isinstance(result, list)
                    and len(result) > 0
                    and isinstance(result[0], list)
                )
                else result
            )

            if not lines:
                return "", [], 0.0

            all_text = ""
            all_confidences = []
            total_conf = 0.0
            char_count = 0

            # 按坐标排序：优先 Y（带阈值容差），然后 X
            # 适用于倾斜的手写行文本
            Y_THRESHOLD = 15  # Y 坐标容差阈值（像素）

            def get_box_coords(item):
                """提取文本框的左上角 (x, y) 坐标"""
                try:
                    if isinstance(item, dict):
                        box = item.get(
                            "text_box", item.get("box", item.get("dt_polys", []))
                        )
                        if box and len(box) >= 1:
                            if isinstance(box[0], (list, tuple)):
                                return (box[0][0], box[0][1])  # (x, y)
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        box = item[0]
                        if box and len(box) >= 1:
                            if isinstance(box[0], (list, tuple)):
                                return (box[0][0], box[0][1])  # (x, y)
                except Exception:
                    pass
                return (0, 0)

            def sort_key(item):
                """排序键：Y 坐标量化（按阈值分组），然后 X 坐标"""
                x, y = get_box_coords(item)
                # 将 Y 坐标按阈值量化，使得同一行的文本有相同的 Y 排序值
                y_quantized = int(y // Y_THRESHOLD)
                return (y_quantized, x)

            sorted_lines = sorted(lines, key=sort_key)

            for line in sorted_lines:
                text = ""
                conf = 0.0

                # 字典格式
                if isinstance(line, dict):
                    text = line.get("text", line.get("rec_text", ""))
                    conf = float(
                        line.get(
                            "confidence", line.get("rec_score", line.get("score", 0.0))
                        )
                    )
                # 列表/元组格式: [[box], (text, confidence)]
                elif isinstance(line, (list, tuple)) and len(line) >= 2:
                    text_info = line[1]
                    if isinstance(text_info, dict):
                        text = text_info.get("text", "")
                        conf = float(
                            text_info.get("confidence", text_info.get("score", 0.0))
                        )
                    elif isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                        text = str(text_info[0])
                        conf = float(text_info[1])

                if text:
                    all_text += text
                    for char in text:
                        all_confidences.append({"char": char, "score": round(conf, 4)})
                        total_conf += conf
                        char_count += 1

            avg_conf = total_conf / char_count if char_count > 0 else 0.0

            # ================================================================
            # 长度检查警告：识别结果远短于 GT 可能是边缘漏读
            # ================================================================
            if gt_text and len(all_text) < len(gt_text) * 0.5:
                if not hasattr(self, "_short_warn_count"):
                    self._short_warn_count = 0
                if self._short_warn_count < 10:  # 只打印前 10 个警告
                    self._short_warn_count += 1
                    print(f"\n[警告] 识别过短! {image_path.name}")
                    print(
                        f"  GT 长度: {len(gt_text)}, 识别长度: {len(all_text)} ({len(all_text) / len(gt_text) * 100:.1f}%)"
                    )
                    print(f"  GT: {gt_text[:30]}...")
                    print(f"  预测: {all_text[:30]}...")

            return all_text, all_confidences, round(avg_conf, 4)

        except Exception as e:
            print(f"[WARNING] OCR 推理失败 ({image_path}): {e}")
            return "", [], 0.0

    def _calculate_metrics(self, pred_text: str, gt_text: str) -> Tuple[int, float]:
        """
        计算编辑距离和 CER

        Args:
            pred_text: 预测文本
            gt_text: 真值文本

        Returns:
            Tuple[edit_distance, cer]
        """
        edit_distance = Levenshtein.distance(pred_text, gt_text)

        if len(gt_text) == 0:
            cer = 1.0 if len(pred_text) > 0 else 0.0
        else:
            cer = min(edit_distance / len(gt_text), 1.0)

        return edit_distance, cer

    def run(self) -> List[InferenceResult]:
        """
        运行基准测试

        Returns:
            推理结果列表
        """
        print("=" * 60)
        print("L2W1 基准测试 - PP-OCRv5 on HWDB")
        print("=" * 60)
        print(f"元数据文件: {self.metadata_path}")
        print(f"图像根目录: {self.image_root}")
        print(f"输出文件: {self.output_path}")
        print()

        # 加载元数据
        samples = self._load_metadata()
        self.stats["total_samples"] = len(samples)

        if not samples:
            print("[ERROR] 没有可用的测试样本")
            return []

        # 清空输出文件
        with open(self.output_path, "w", encoding="utf-8") as f:
            pass

        results = []

        # 逐个处理样本
        print("\n[推理进度]")
        for item in tqdm(samples, desc="OCR 推理"):
            sample_id = item.get(
                "id", item.get("sample_id", f"sample_{len(results):06d}")
            )
            image_path_str = item.get("image_path", item.get("image", ""))
            gt_text = item.get("gt_text", item.get("text", item.get("label", "")))

            # 解析图像路径
            image_path = self._resolve_image_path(image_path_str)

            if image_path is None:
                print(f"\n[WARNING] 图像不存在: {image_path_str}")
                self.stats["failed_samples"] += 1

                # 记录失败样本
                result = InferenceResult(
                    id=sample_id,
                    image_path=image_path_str,
                    gt_text=gt_text,
                    pred_text="",
                    avg_confidence=0.0,
                    char_confidences=[],
                    edit_distance=len(gt_text),
                    cer=1.0,
                )
                results.append(result)
                self._write_result(result)
                continue

            # 运行 OCR (传入 gt_text 用于长度检查警告)
            pred_text, char_confidences, avg_confidence = self._run_ocr(
                image_path, gt_text
            )

            # 计算指标
            edit_distance, cer = self._calculate_metrics(pred_text, gt_text)

            # 创建结果
            result = InferenceResult(
                id=sample_id,
                image_path=str(image_path),
                gt_text=gt_text,
                pred_text=pred_text,
                avg_confidence=round(avg_confidence, 4),
                char_confidences=char_confidences,
                edit_distance=edit_distance,
                cer=round(cer, 4),
            )

            results.append(result)
            self._write_result(result)

            # 更新统计
            self.stats["success_samples"] += 1
            self.stats["total_cer"] += cer
            self.stats["total_confidence"] += avg_confidence
            self.stats["total_edit_distance"] += edit_distance

        # 打印汇总统计
        self._print_summary()

        return results

    def _write_result(self, result: InferenceResult):
        """写入单条结果到 JSONL 文件"""
        with open(self.output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    def _print_summary(self):
        """打印汇总统计信息"""
        print("\n" + "=" * 60)
        print("测试集统计汇总")
        print("=" * 60)

        total = self.stats["total_samples"]
        success = self.stats["success_samples"]
        failed = self.stats["failed_samples"]

        print(f"总样本数: {total}")
        print(f"成功推理: {success}")
        print(f"失败样本: {failed}")
        print()

        if success > 0:
            avg_cer = self.stats["total_cer"] / success
            avg_conf = self.stats["total_confidence"] / success
            avg_edit = self.stats["total_edit_distance"] / success

            print(f"平均 CER: {avg_cer:.4f} ({avg_cer * 100:.2f}%)")
            print(f"平均置信度: {avg_conf:.4f}")
            print(f"平均编辑距离: {avg_edit:.2f}")
        else:
            print("[WARNING] 没有成功的推理样本，无法计算平均指标")

        print()
        print(f"结果已保存至: {self.output_path}")
        print("=" * 60)


# =============================================================================
# 命令行入口
# =============================================================================


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="PP-OCRv5 基准测试 - HWDB 测试集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 使用默认配置运行
    python scripts/baseline_inference.py
    
    # 指定自定义路径
    python scripts/baseline_inference.py \\
        --metadata_path ./data/raw/HWDB_Benchmark/test_metadata.jsonl \\
        --image_root ./data/raw/HWDB_Benchmark/ \\
        --output_path ./results/baseline_results.jsonl
    
    # 禁用 GPU
    python scripts/baseline_inference.py --no_gpu
    
    # 禁用文本检测（直接识别模式，适用于行文本图像）
    python scripts/baseline_inference.py --no_det
        """,
    )

    parser.add_argument(
        "--metadata_path",
        type=str,
        default=DEFAULT_METADATA_PATH,
        help=f"测试集元数据文件路径 (默认: {DEFAULT_METADATA_PATH})",
    )
    parser.add_argument(
        "--image_root",
        type=str,
        default=DEFAULT_IMAGE_ROOT,
        help=f"图像根目录 (默认: {DEFAULT_IMAGE_ROOT})",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help=f"输出结果路径 (默认: {DEFAULT_OUTPUT_PATH})",
    )
    parser.add_argument(
        "--use_gpu", action="store_true", default=True, help="使用 GPU (默认启用)"
    )
    parser.add_argument("--no_gpu", action="store_true", help="禁用 GPU，使用 CPU 推理")
    parser.add_argument(
        "--no_det",
        action="store_true",
        help="禁用文本检测，直接进行文本识别（适用于行文本图像）",
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()

    # 处理 GPU 参数
    use_gpu = args.use_gpu and not args.no_gpu
    use_det = not args.no_det

    # 创建推理器
    inferencer = BaselineInference(
        metadata_path=args.metadata_path,
        image_root=args.image_root,
        output_path=args.output_path,
        use_gpu=use_gpu,
        det=use_det,
    )

    # 运行推理
    results = inferencer.run()

    print(f"\n[完成] 共处理 {len(results)} 个样本")


if __name__ == "__main__":
    main()
