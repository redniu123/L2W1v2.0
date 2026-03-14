#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v5.1: MiniCPM-V 2.6 Expert Wrapper

支持 openbmb/MiniCPM-V-2_6 本地推理。
显存策略: bfloat16 + device_map=auto。
"""

from typing import Dict, Union

import numpy as np

from .base_expert import BaseVLMExpert


class MiniCPMVExpert(BaseVLMExpert):
    """
    MiniCPM-V 2.6 专家包装器

    模型特点:
    - 使用 AutoModel + AutoTokenizer
    - 使用 model.chat() 接口
    - trust_remote_code=True
    """

    def __init__(self, model_path: str, dtype: str = "bfloat16", max_new_tokens: int = 128):
        self.model_path = model_path
        self.dtype_str = dtype
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None
        self._initialized = False
        self._init_model()

    def _init_model(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        dtype = torch.bfloat16 if self.dtype_str == "bfloat16" else torch.float16
        print(f"[MiniCPM-V] Loading {self.model_path} ({self.dtype_str})...")

        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        self._initialized = True
        print(f"[MiniCPM-V] Ready.")

    def chat_with_image(self, image: Union[str, np.ndarray], prompt: str) -> str:
        from PIL import Image as PILImage

        if isinstance(image, str):
            pil_img = PILImage.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = PILImage.fromarray(image).convert("RGB")
        else:
            pil_img = image.convert("RGB")

        msgs = [{"role": "user", "content": [pil_img, prompt]}]

        response = self.model.chat(
            image=None,
            msgs=msgs,
            tokenizer=self.tokenizer,
            sampling=False,
            max_new_tokens=self.max_new_tokens,
        )
        return response

    def get_model_info(self) -> Dict:
        return {
            "backend": "local_vlm",
            "model_type": "minicpm_v",
            "model_path": self.model_path,
            "dtype": self.dtype_str,
            "initialized": self._initialized,
        }
