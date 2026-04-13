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
            f"你是一个极度保守的 OCR 纠错专家。以下是初步的单行文本识别结果：\n"
            f"【 {ocr_text} 】\n\n"
            f"{hint_block}"
            f"请结合提供的图像，只在你高度确定存在 OCR 错误时，做最小必要修改。\n"
            f"**最高约束红线：**\n"
            f"1. 除非你能从图像中高度确认 OCR 有误，否则必须逐字原样输出。\n"
            f"2. 绝对禁止润色、扩写、缩写、改写语序、补全你推测但看不清的内容。\n"
            f"3. 绝对禁止仅出于排版美观而修改全半角、括号、句号、冒号、编号样式。\n"
            f"4. 如需修改，只允许做最小必要字符级改动；若拿不准，保持原样。\n\n"
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

        # 去除 Qwen3 思维链 <think>...</think> 块
        import re as _re
        text = _re.sub(r'<think>.*?</think>', '', text, flags=_re.DOTALL).strip()
        # 去除 "Thinking Process:" 等思维链残留
        text = _re.sub(r'^(Thinking Process|思考过程|思维链)[:\uff1a].*', '', text, flags=_re.MULTILINE).strip()

        # 过滤英文分析输出（模型无视指令时降级）
        if _re.search(r'^\s*(\d+\.\s*)?\*{0,2}(Analyze|analyze|Analysis|Step|Correction|Result)', text):
            return fallback
        # 过滤纯英文输出（OCR 任务应输出中文）
        chinese_chars = len(_re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(_re.sub(r'\s', '', text))
        if total_chars > 4 and chinese_chars / max(total_chars, 1) < 0.3:
            return fallback

        patterns = [
            r"修正后的文本[:\uff1a]\s*(.+)",
            r"修正后的完整文本[:\uff1a]\s*(.+)",
            r"输出[:\uff1a]\s*(.+)",
        ]
        for pattern in patterns:
            m = _re.search(pattern, text)
            if m:
                text = m.group(1).strip()
                break
        text = text.split("\n")[0].strip()
        text = text.strip('"\'\u201c\u201d\u2018\u2019')
        text = text.strip()
        # 防复述检测
        PROMPT_FRAGMENTS = [
            "尽可能保持原句原貌", "绝对禁止润色", "绝对禁止对句子",
            "最高约束红线", "请直接输出修正后", "系统检测到该文本",
            "修正上述文本中的错别字", "如有错别字请修正", "只输出文字",
            "否则原样输出", "不要解释",
            "The text in the image", "OCR (Optical Character", "written in Chinese",
            "The OCR", "reads:", "Thinking Process", "思考过程",
        ]
        for frag in PROMPT_FRAGMENTS:
            if frag in text:
                return fallback
        return text if text else fallback
