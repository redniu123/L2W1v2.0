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
        total = kwargs.get('total', len(iterable) if hasattr(iterable, '__len__') else None)
        desc = kwargs.get('desc', 'Processing')
        for i, item in enumerate(iterable):
            if total:
                print(f"\r{desc}: {i+1}/{total}", end="", flush=True)
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
            self.ocr = PaddleOCR(
                ocr_version='PP-OCRv5',
                lang='ch',
                use_gpu=self.use_gpu,
                use_angle_cls=True,  # 使用方向分类器
                show_log=False,      # 减少日志输出
            )
            print("[INFO] PP-OCRv5 模型初始化成功")
            print(f"[INFO] GPU 模式: {'启用' if self.use_gpu else '禁用'}")
            print(f"[INFO] 文本检测: {'启用' if self.use_det else '禁用（直接识别）'}")
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
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
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
        image_path_str = image_path.replace('\\', '/').strip('./')
        image_root_str = str(self.image_root).replace('\\', '/').strip('./')
        
        if image_path_str.startswith(image_root_str):
            candidates.insert(0, Path(image_path))
        
        for candidate in candidates:
            if candidate.exists():
                return candidate
        
        return None
    
    def _run_ocr(self, image_path: Path) -> Tuple[str, List[Dict[str, Any]], float]:
        """
        运行 OCR 推理
        
        Args:
            image_path: 图像路径
            
        Returns:
            Tuple[pred_text, char_confidences, avg_confidence]
        """
        try:
            # 读取图像
            img = cv2.imread(str(image_path))
            if img is None:
                return "", [], 0.0
            
            # 运行 OCR
            result = self.ocr.ocr(img, det=self.use_det, cls=True)
            
            if result is None or len(result) == 0 or result[0] is None:
                return "", [], 0.0
            
            # 解析结果
            # PaddleOCR 返回格式: [[[box], (text, confidence)], ...]
            lines = result[0]
            
            if not lines:
                return "", [], 0.0
            
            # 按纵坐标排序（从上到下）
            def get_y_coord(item):
                if isinstance(item, list) and len(item) >= 2:
                    box = item[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    if box and len(box) >= 1:
                        return box[0][1] if isinstance(box[0], list) else 0
                return 0
            
            sorted_lines = sorted(lines, key=get_y_coord)
            
            # 提取文本和置信度
            all_text = ""
            all_confidences = []
            total_conf = 0.0
            char_count = 0
            
            for line in sorted_lines:
                if not isinstance(line, list) or len(line) < 2:
                    continue
                
                text_info = line[1]  # (text, confidence)
                if not isinstance(text_info, (list, tuple)) or len(text_info) < 2:
                    continue
                
                text = text_info[0]
                conf = float(text_info[1])
                
                if text:
                    all_text += text
                    
                    # 为每个字符分配相同的置信度（行级别）
                    # 注意: PaddleOCR 默认只提供行级置信度，不提供字符级置信度
                    for char in text:
                        all_confidences.append({
                            "char": char,
                            "score": round(conf, 4)
                        })
                        total_conf += conf
                        char_count += 1
            
            avg_conf = total_conf / char_count if char_count > 0 else 0.0
            
            return all_text, all_confidences, avg_conf
            
        except Exception as e:
            print(f"[WARNING] OCR 推理失败 ({image_path}): {e}")
            return "", [], 0.0
    
    def _calculate_metrics(
        self, 
        pred_text: str, 
        gt_text: str
    ) -> Tuple[int, float]:
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
        with open(self.output_path, 'w', encoding='utf-8') as f:
            pass
        
        results = []
        
        # 逐个处理样本
        print("\n[推理进度]")
        for item in tqdm(samples, desc="OCR 推理"):
            sample_id = item.get('id', item.get('sample_id', f"sample_{len(results):06d}"))
            image_path_str = item.get('image_path', item.get('image', ''))
            gt_text = item.get('gt_text', item.get('text', item.get('label', '')))
            
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
            
            # 运行 OCR
            pred_text, char_confidences, avg_confidence = self._run_ocr(image_path)
            
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
        with open(self.output_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + '\n')
    
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
            
            print(f"平均 CER: {avg_cer:.4f} ({avg_cer*100:.2f}%)")
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
        """
    )
    
    parser.add_argument(
        "--metadata_path", 
        type=str, 
        default=DEFAULT_METADATA_PATH,
        help=f"测试集元数据文件路径 (默认: {DEFAULT_METADATA_PATH})"
    )
    parser.add_argument(
        "--image_root", 
        type=str, 
        default=DEFAULT_IMAGE_ROOT,
        help=f"图像根目录 (默认: {DEFAULT_IMAGE_ROOT})"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default=DEFAULT_OUTPUT_PATH,
        help=f"输出结果路径 (默认: {DEFAULT_OUTPUT_PATH})"
    )
    parser.add_argument(
        "--use_gpu", 
        action="store_true", 
        default=True,
        help="使用 GPU (默认启用)"
    )
    parser.add_argument(
        "--no_gpu", 
        action="store_true",
        help="禁用 GPU，使用 CPU 推理"
    )
    parser.add_argument(
        "--no_det", 
        action="store_true",
        help="禁用文本检测，直接进行文本识别（适用于行文本图像）"
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

