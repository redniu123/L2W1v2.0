#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v5.1: Qwen2.5-VL Expert Wrapper

支持 Qwen2.5-VL-7B-Instruct 本地推理。
显存策略: bfloat16 + device_map=auto，22GB 显存下安全运行。
"""

from pathlib import Path
from typing import Dict, Union

import numpy as np

from .base_expert import BaseVLMExpert


class QwenVLExpert(BaseVLMExpert):
    """
    Qwen2.5-VL 专家包装器

    模型特点:
    - 使用 AutoProcessor + Qwen2_5_VLForConditionalGeneration
    - 动态分辨率: min_pixels/max_pixels 控制
    - 支持 Flash Attention 2
    """

    def __init__(self, model_path: str, dtype: str = "bfloat16", max_new_tokens: int = 128):
        """
        Args:
            model_path: 本地模型路径，如 ./models/agent_b_vlm/Qwen2.5-VL-7B-Instruct
            dtype: 推理精度，bfloat16 或 float16
            max_new_tokens: 最大生成 token 数
        """
        self.model_path = model_path
        self.dtype_str = dtype
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None
        self._initialized = False
        self._init_model()

    def _init_model(self):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        dtype = torch.bfloat16 if self.dtype_str == "bfloat16" else torch.float16

        print(f"[QwenVL] Loading {self.model_path} ({self.dtype_str})...")

        # 尝试启用 Flash Attention 2
        attn = None
        try:
            import flash_attn  # noqa
            attn = "flash_attention_2"
            print("[QwenVL] Flash Attention 2 enabled")
        except ImportError:
            print("[QwenVL] Flash Attention 2 not available, using sdpa")
            attn = "sdpa"

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation=attn,
            trust_remote_code=True,
        )
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
        self._initialized = True
        print(f"[QwenVL] Ready.")

    def chat_with_image(self, image: Union[str, np.ndarray], prompt: str) -> str:
        import torch
        from PIL import Image as PILImage

        if isinstance(image, str):
            pil_img = PILImage.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = PILImage.fromarray(image).convert("RGB")
        else:
            pil_img = image.convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[pil_img], padding=True, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, generated_ids)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def get_model_info(self) -> Dict:
        return {
            "backend": "local_vlm",
            "model_type": "qwen2.5_vl",
            "model_path": self.model_path,
            "dtype": self.dtype_str,
            "initialized": self._initialized,
        }
