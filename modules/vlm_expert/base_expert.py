#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v5.1: Base VLM Expert Interface
所有本地 VLM 专家的抽象基类。
统一接口: chat_with_image(image_path, prompt_text) -> str
"""
import re
from abc import ABC, abstractmethod
from typing import Dict, Union
import numpy as np


class BaseVLMExpert(ABC):

    @abstractmethod
    def chat_with_image(self, image_path: Union[str, np.ndarray], prompt_text: str) -> str:
        """核心推理接口：给定图像路径和提示词，返回模型原始输出。"""
        ...

    @abstractmethod
    def get_model_info(self) -> Dict:
        """返回模型元信息（backend, model_type, model_path 等）"""
        ...

    def process_hard_sample(self, image: Union[str, np.ndarray], manifest: Dict) -> Dict:
        """处理困难样本（标准 pipeline 接口）"""
        ocr_text = manifest.get("ocr_text", "")
        suspicious_index = manifest.get("suspicious_index", -1)
        suspicious_char = manifest.get("suspicious_char", "")

        hint_lines = []
        if suspicious_index is not None and suspicious_index >= 0 and suspicious_char:
            hint_lines.append(
                f"其中第 {suspicious_index + 1} 个字符 '{suspicious_char}' 的机器置信度极低，请重点关注。"
            )
        hint_lines.append("本文本属于【地质勘探】领域，请留意专业术语的准确性。")
        hint_block = "系统检测到该文本可能存在识别错误。\n" + "\n".join(hint_lines) + "\n\n"

        prompt = (
            f"你是一个严格的 OCR 纠错专家。以下是初步的单行文本识别结果：\n"
            f"【 {ocr_text} 】\n\n"
            f"{hint_block}"
            f"请结合提供的图像，修正上述文本中的错别字或漏字。\n"
            f"**最高约束红线：**\n"
            f"1. 尽可能保持原句原貌，绝对禁止润色、改写或大幅度增删。\n"
            f"2. 如果认为没有错误，请直接原样输出。\n\n"
            f"请直接输出修正后的完整文本，不要任何解释或多余字符："
        )

        raw = self.chat_with_image(image, prompt)
        corrected_text = self._parse_output(raw, ocr_text)

        return {
            "original_text": ocr_text,
            "corrected_text": corrected_text,
            "suspicious_index": suspicious_index,
            "suspicious_char": suspicious_char,
            "is_corrected": corrected_text != ocr_text,
        }

    def _parse_output(self, raw_text: str, fallback: str = "") -> str:
        """解析模型输出：只取第一行，去除引号，防止解释文字混入。"""
        if not raw_text:
            return fallback
        text = raw_text.strip()
        patterns = [
            r"修正后的文本[:\uff1a]\s*(.+)",
            r"修正后的完整文本[:\uff1a]\s*(.+)",
            r"输出[:\uff1a]\s*(.+)",
        ]
        for pattern in patterns:
            m = re.search(pattern, text)
            if m:
                text = m.group(1).strip()
                break
        text = text.split("\n")[0].strip()
        text = text.strip('"\'\u201c\u201d\u2018\u2019')
        return text if text else fallback
