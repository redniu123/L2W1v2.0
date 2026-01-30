#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal TextDetector wrapper for SH-DA++ v4.0

依赖：paddleocr (PaddleOCR Python 包)
用途：在 REC 前获取 dt_boxes，用于旋转裁剪矫正
"""

from typing import List


class TextDetector:
    """
    轻量 TextDetector：使用 PaddleOCR 内置检测器
    """

    def __init__(self, args):
        self.use_gpu = getattr(args, "use_gpu", True)
        self.det_model_dir = getattr(args, "det_model_dir", None)
        self._ocr = None
        self._init_ocr()

    def _init_ocr(self):
        try:
            from paddleocr import PaddleOCR
        except Exception as e:
            raise ImportError(
                "paddleocr 未安装，无法启用检测器。"
                "请在云端环境安装 paddleocr 或提供自定义检测器。"
            ) from e

        # 基础参数（所有版本应该都支持）
        base_kwargs = {
            "det": True,
            "rec": False,
            "cls": False,
        }
        if self.det_model_dir:
            base_kwargs["det_model_dir"] = self.det_model_dir

        # 可选参数列表（按优先级尝试）
        optional_params_list = [
            {"show_log": False, "use_gpu": self.use_gpu},  # 尝试1: 全部可选参数
            {"use_gpu": self.use_gpu},                      # 尝试2: 去掉 show_log
            {"show_log": False},                            # 尝试3: 去掉 use_gpu
            {},                                             # 尝试4: 仅基础参数
        ]

        last_error = None
        for optional_params in optional_params_list:
            try:
                kwargs = {**base_kwargs, **optional_params}
                self._ocr = PaddleOCR(**kwargs)
                return  # 成功则退出
            except TypeError as e:
                last_error = e
                continue
            except Exception as e:
                # 非参数错误，直接抛出
                if "Unknown argument" not in str(e):
                    raise
                last_error = e
                continue

        # 所有尝试都失败
        if last_error:
            raise last_error

    def __call__(self, img) -> List:
        """
        返回 dt_boxes: List[box], box = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        """
        if self._ocr is None:
            return []
        result = self._ocr.ocr(img, det=True, rec=False, cls=False)
        if result is None:
            return []

        # PaddleOCR 返回结构可能是 [boxes] 或 [[box, score], ...]
        if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
            result = result[0]

        boxes = []
        for item in result:
            if not isinstance(item, (list, tuple)) or len(item) == 0:
                continue
            # item 可能是 box 或 [box, score]
            if isinstance(item[0], (list, tuple)) and len(item[0]) == 2:
                box = item
            else:
                box = item[0]
            boxes.append(box)
        return boxes
