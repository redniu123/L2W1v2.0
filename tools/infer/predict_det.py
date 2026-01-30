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
    兼容不同版本 PaddleOCR API
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

        # 按优先级尝试不同的参数组合
        # 不同版本 PaddleOCR 支持的参数不同
        init_attempts = [
            # 尝试1: 新版 API (仅 det_model_dir)
            {"det_model_dir": self.det_model_dir} if self.det_model_dir else {},
            # 尝试2: 旧版 API (det/rec/cls 在初始化时指定)
            {"det": True, "rec": False, "cls": False, "det_model_dir": self.det_model_dir} if self.det_model_dir else {"det": True, "rec": False, "cls": False},
        ]

        # 可选参数（逐个去掉尝试）
        optional_keys = ["show_log", "use_gpu"]
        optional_values = [False, self.use_gpu]

        last_error = None
        for base_kwargs in init_attempts:
            # 尝试加上所有可选参数 → 逐个去掉 → 最后只用 base
            optional_combos = [
                dict(zip(optional_keys, optional_values)),  # 全部
                {"use_gpu": self.use_gpu},                  # 只 use_gpu
                {"show_log": False},                        # 只 show_log
                {},                                          # 无
            ]
            for opt in optional_combos:
                kwargs = {**base_kwargs, **opt}
                # 移除 None 值
                kwargs = {k: v for k, v in kwargs.items() if v is not None}
                try:
                    self._ocr = PaddleOCR(**kwargs)
                    return  # 成功
                except TypeError as e:
                    last_error = e
                    continue
                except Exception as e:
                    err_str = str(e)
                    if "Unknown argument" in err_str or "unexpected keyword" in err_str.lower():
                        last_error = e
                        continue
                    raise

        # 最后一搏：完全无参数初始化
        try:
            self._ocr = PaddleOCR()
            return
        except Exception as e:
            last_error = e

        if last_error:
            raise last_error

    def __call__(self, img) -> List:
        """
        返回 dt_boxes: List[box], box = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        """
        if self._ocr is None:
            return []

        # 尝试不同的调用方式
        result = None
        call_attempts = [
            {"det": True, "rec": False, "cls": False},  # 新版：参数在调用时指定
            {},                                          # 旧版：参数已在初始化时指定
        ]

        for call_kwargs in call_attempts:
            try:
                result = self._ocr.ocr(img, **call_kwargs)
                break
            except TypeError:
                continue
            except Exception as e:
                if "Unknown argument" in str(e) or "unexpected keyword" in str(e).lower():
                    continue
                raise

        if result is None:
            return []

        # PaddleOCR 返回结构可能是 [boxes] 或 [[box, score], ...] 或 [[[box], text], ...]
        # 需要兼容多种格式
        return self._parse_boxes(result)

    def _parse_boxes(self, result) -> List:
        """解析不同版本 PaddleOCR 返回的检测结果"""
        if result is None:
            return []

        # 展开嵌套列表 [[...]] -> [...]
        if isinstance(result, list) and len(result) == 1 and isinstance(result[0], list):
            result = result[0]

        if result is None:
            return []

        boxes = []
        for item in result:
            if item is None:
                continue
            if not isinstance(item, (list, tuple)) or len(item) == 0:
                continue

            box = None
            # 格式1: item 直接是 box [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            if self._is_box(item):
                box = item
            # 格式2: item 是 [box, text_info] 或 [box, score]
            elif isinstance(item[0], (list, tuple)):
                if self._is_box(item[0]):
                    box = item[0]
                # 格式3: item[0] 是 [box]
                elif len(item[0]) > 0 and self._is_box(item[0][0]):
                    box = item[0][0]

            if box is not None:
                boxes.append(box)

        return boxes

    def _is_box(self, obj) -> bool:
        """判断是否是 4 点 box 格式"""
        if not isinstance(obj, (list, tuple)):
            return False
        if len(obj) != 4:
            return False
        # 每个点应该是 [x, y]
        for pt in obj:
            if not isinstance(pt, (list, tuple)) or len(pt) != 2:
                return False
        return True
