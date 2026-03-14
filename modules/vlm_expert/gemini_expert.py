#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v5.1: Gemini-3-Flash API Expert

通过 OpenAI 兼容接口调用 Gemini-3-Flash，支持：
- 100 个 API Key 轮询（并发安全）
- 指数退避重试（最多 2 次）
- 容错降级（失败返回 T_A）
"""

import base64
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import requests
from PIL import Image


class APIKeyManager:
    """
    API Key 管理器（并发安全）
    
    功能：
    - 从 key.txt 读取 100 个 API Key
    - Round-robin 轮询
    - 线程安全（使用 threading.Lock）
    """
    
    def __init__(self, key_file: str = "docs/key.txt"):
        """
        Args:
            key_file: Key 文件路径（相对于项目根目录）
        """
        self.key_file = Path(key_file)
        self.keys = self._load_keys()
        self._index = 0
        self._lock = threading.Lock()
        
        if not self.keys:
            raise ValueError(f"No API keys found in {key_file}")
        
        print(f"[APIKeyManager] Loaded {len(self.keys)} API keys")
    
    def _load_keys(self) -> List[str]:
        """从 key.txt 读取 API Keys"""
        keys = []
        
        if not self.key_file.exists():
            print(f"[Warning] Key file not found: {self.key_file}")
            return keys
        
        with open(self.key_file, "r", encoding="utf-8") as f:
            in_list = False
            for line in f:
                line = line.strip()
                
                # 检测 API_KEYS = [ 开始
                if "API_KEYS" in line and "[" in line:
                    in_list = True
                    continue
                
                # 检测 ] 结束
                if in_list and "]" in line:
                    break
                
                # 提取 Key
                if in_list and line.startswith('"sk-'):
                    # 移除引号和逗号
                    key = line.strip('",')
                    if key.startswith("sk-"):
                        keys.append(key)
        
        return keys
    
    def get_next_key(self) -> str:
        """
        获取下一个 API Key（线程安全）
        
        Returns:
            API Key 字符串
        """
        with self._lock:
            key = self.keys[self._index]
            self._index = (self._index + 1) % len(self.keys)
            return key
    
    def get_key_count(self) -> int:
        """获取 Key 总数"""
        return len(self.keys)
    
    def get_current_index(self) -> int:
        """获取当前索引（调试用）"""
        with self._lock:
            return self._index


@dataclass
class GeminiConfig:
    """Gemini API 配置"""

    base_url: str = "https://new.lemonapi.site/v1"
    model_name: str = "gemini-3-flash-preview"
    key_file: str = "docs/key.txt"  # Key 文件路径
    temperature: float = 0.1
    max_tokens: int = 256
    max_retries: int = 2
    timeout: int = 180  # 3 分钟

    def __post_init__(self):
        # 初始化 APIKeyManager
        self.key_manager = APIKeyManager(self.key_file)
        print(f"[GeminiConfig] Using {self.key_manager.get_key_count()} API keys")


class GeminiAgentB:
    """Gemini-3-Flash Agent B（OpenAI 兼容接口）"""

    def __init__(self, config: GeminiConfig = None):
        self.config = config or GeminiConfig()
        self._initialized = True
        print(f"[Gemini Agent B] Initialized with {self.config.key_manager.get_key_count()} API keys")

    def _encode_image(self, image_path: str) -> str:
        """将图像编码为 base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _call_api(
        self, prompt: str, image_base64: str, api_key: str
    ) -> Optional[str]:
        """调用 Gemini API（OpenAI 兼容格式）"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.config.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                        },
                    ],
                }
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        try:
            response = requests.post(
                f"{self.config.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.config.timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"[Gemini] Rate limit (429), will retry with next key")
            else:
                print(f"[Gemini] HTTP error {e.response.status_code}: {e}")
            return None
        except Exception as e:
            print(f"[Gemini] API call failed: {e}")
            return None

    def _parse_output(self, raw_text: str) -> str:
        """解析模型输出，提取修正后的文本"""
        import re

        text = raw_text.strip()
        patterns = [
            r"修正后的文本[：:]\s*(.+)",
            r"修正后的完整文本[：:]\s*(.+)",
            r"输出[：:]\s*(.+)",
        ]
        for pattern in patterns:
            m = re.search(pattern, text, re.DOTALL)
            if m:
                text = m.group(1).strip()
                break
        text = text.strip('"\'""''【】')
        text = re.sub(r"\s+", "", text)
        return text

    def process_hard_sample(
        self, image: Union[str, np.ndarray], manifest: Dict
    ) -> Dict:
        """
        处理困难样本（与 Qwen 接口兼容）

        Args:
            image: 图像路径或 numpy 数组
            manifest: 包含 ocr_text, suspicious_index, suspicious_char, risk_level

        Returns:
            dict: {original_text, corrected_text, is_corrected, ...}
        """
        ocr_text = manifest.get("ocr_text", "")
        suspicious_index = manifest.get("suspicious_index", -1)
        suspicious_char = manifest.get("suspicious_char", "")
        risk_level = manifest.get("risk_level", "medium")

        # 构建 Targeted Correction 提示词
        hint_lines = []
        if suspicious_index >= 0 and suspicious_char:
            hint_lines.append(
                f"其中第 {suspicious_index + 1} 个字符 '{suspicious_char}' 的机器置信度极低，请重点关注。"
            )
        hint_lines.append("本文本属于【地质勘探】领域，请留意专业术语的准确性。")

        hint_block = (
            "系统检测到该文本可能存在识别错误。\n" + "\n".join(hint_lines) + "\n\n"
        )

        prompt = (
            f"你是一个严格的 OCR 纠错专家。以下是初步的单行文本识别结果：\n"
            f"【 {ocr_text} 】\n\n"
            f"{hint_block}"
            f"请结合提供的图像，修正上述文本中的错别字或漏字。\n"
            f"**最高约束红线：**\n"
            f"1. 尽可能保持原句原貌，绝对禁止对句子进行润色、改写或大幅度增删。\n"
            f"2. 如果认为没有错误，请直接原样输出。\n\n"
            f"请直接输出修正后的完整文本，不要任何解释或多余字符："
        )

        # 处理图像
        if isinstance(image, str):
            image_path = image
        elif isinstance(image, np.ndarray):
            import tempfile
            import cv2

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                cv2.imwrite(tmp.name, image)
                image_path = tmp.name
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        image_base64 = self._encode_image(image_path)

        # 指数退避重试（简化日志）
        corrected_text = None
        for attempt in range(self.config.max_retries):
            api_key = self.config.key_manager.get_next_key()
            corrected_text = self._call_api(prompt, image_base64, api_key)
            if corrected_text:
                break
            if attempt < self.config.max_retries - 1:
                sleep_time = min(4, 2**attempt) + random.uniform(0, 0.5)
                time.sleep(sleep_time)

        # 容错降级（静默失败）
        if not corrected_text:
            corrected_text = ocr_text

        # 解析输出
        corrected_text = self._parse_output(corrected_text)

        return {
            "original_text": ocr_text,
            "corrected_text": corrected_text,
            "suspicious_index": suspicious_index,
            "suspicious_char": suspicious_char,
            "is_corrected": corrected_text != ocr_text,
        }

    def correct_line(
        self, image: Union[str, np.ndarray], ocr_text: str, suspicious_idx: int, suspicious_char: str
    ) -> str:
        """便捷接口：行级纠错"""
        manifest = {
            "ocr_text": ocr_text,
            "suspicious_index": suspicious_idx,
            "suspicious_char": suspicious_char,
            "risk_level": "medium",
        }
        result = self.process_hard_sample(image, manifest)
        return result["corrected_text"]

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "backend": "gemini",
            "model_name": self.config.model_name,
            "base_url": self.config.base_url,
            "num_keys": self.config.key_manager.get_key_count(),
            "initialized": self._initialized,
        }


def create_gemini_agent(config: GeminiConfig = None) -> GeminiAgentB:
    """工厂函数：创建 Gemini Agent B"""
    return GeminiAgentB(config or GeminiConfig())
