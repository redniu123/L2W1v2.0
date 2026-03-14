#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v5.1: InternVL2 Expert Wrapper

支持 OpenGVLab/InternVL2-8B 本地推理。
显存策略: bfloat16 + device_map=auto。
"""

from typing import Dict, Union

import numpy as np

from .base_expert import BaseVLMExpert


class InternVL2Expert(BaseVLMExpert):
    """
    InternVL2 专家包装器

    模型特点:
    - 使用 AutoModel + AutoTokenizer
    - 图像预处理: dynamic_preprocess (448x448 tiles)
    - 支持 trust_remote_code
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
        print(f"[InternVL2] Loading {self.model_path} ({self.dtype_str})...")

        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )
        self._initialized = True
        print(f"[InternVL2] Ready.")

    def _load_image(self, image: Union[str, np.ndarray]):
        """InternVL2 专用图像加载（dynamic preprocess）"""
        import torch
        import torchvision.transforms as T
        from PIL import Image as PILImage
        from torchvision.transforms.functional import InterpolationMode

        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)
        IMG_SIZE = 448

        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

        if isinstance(image, str):
            pil_img = PILImage.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            pil_img = PILImage.fromarray(image).convert("RGB")
        else:
            pil_img = image.convert("RGB")

        pixel_values = transform(pil_img).unsqueeze(0)  # (1, 3, H, W)
        return pixel_values

    def chat_with_image(self, image: Union[str, np.ndarray], prompt: str) -> str:
        import torch

        pixel_values = self._load_image(image)
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        pixel_values = pixel_values.to(device=device, dtype=dtype)

        generation_config = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )

        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            prompt,
            generation_config,
        )
        return response

    def get_model_info(self) -> Dict:
        return {
            "backend": "local_vlm",
            "model_type": "internvl2",
            "model_path": self.model_path,
            "dtype": self.dtype_str,
            "initialized": self._initialized,
        }
