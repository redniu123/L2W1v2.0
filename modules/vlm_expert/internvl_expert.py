#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v5.1: InternVL2 Expert
兼容 InternVL2-8B-AWQ 和 InternVL2_5-8B-AWQ。
AWQ 量化直接加载，无需 bitsandbytes。
"""
from typing import Dict, Union
import numpy as np
from .base_expert import BaseVLMExpert


class InternVL2Expert(BaseVLMExpert):

    def __init__(self, model_path: str, torch_dtype: str = "float16", max_new_tokens: int = 128):
        self.model_path = model_path
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None
        self._init_model()

    def _init_model(self):
        import torch
        from transformers import AutoModel, AutoTokenizer, GenerationMixin, GenerationConfig
        dtype = torch.float16 if self.torch_dtype == "float16" else torch.bfloat16
        is_awq = "awq" in self.model_path.lower()
        print(f"[InternVL2] Loading {self.model_path} (AWQ={is_awq}, {self.torch_dtype})...")
        self.model = AutoModel.from_pretrained(
            self.model_path,
            dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        ).eval()

        # transformers >=4.50 不再自动继承 GenerationMixin，手动 patch
        lm = getattr(self.model, "language_model", None)
        if lm is not None and not isinstance(lm, GenerationMixin):
            lm.__class__ = type(
                lm.__class__.__name__,
                (lm.__class__, GenerationMixin),
                {},
            )
            print("[InternVL2] Applied GenerationMixin patch")

        # 修复 generation_config=None 导致的 AttributeError
        if lm is not None and getattr(lm, "generation_config", None) is None:
            try:
                lm.generation_config = GenerationConfig.from_model_config(lm.config)
            except Exception:
                lm.generation_config = GenerationConfig()
            print("[InternVL2] Initialized generation_config")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        print("[InternVL2] Ready.")

    def _preprocess_image(self, image_path: Union[str, np.ndarray]):
        import torch
        import torchvision.transforms as T
        from PIL import Image as PILImage
        from torchvision.transforms.functional import InterpolationMode
        IMG_SIZE = 448
        transform = T.Compose([
            T.Lambda(lambda img: img.convert("RGB")),
            T.Resize((IMG_SIZE, IMG_SIZE), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        if isinstance(image_path, str):
            pil = PILImage.open(image_path).convert("RGB")
        elif isinstance(image_path, np.ndarray):
            pil = PILImage.fromarray(image_path).convert("RGB")
        else:
            pil = image_path.convert("RGB")
        tensor = transform(pil).unsqueeze(0)  # (1, 3, H, W)
        assert tensor is not None and tensor.shape[0] == 1, f"preprocess failed: {tensor}"
        return tensor

    def chat_with_image(self, image_path: Union[str, np.ndarray], prompt_text: str) -> str:
        import torch
        try:
            pixel_values = self._preprocess_image(image_path)
            device = next(self.model.parameters()).device
            dtype = next(self.model.parameters()).dtype
            pixel_values = pixel_values.to(device=device, dtype=dtype)
            assert pixel_values is not None and len(pixel_values.shape) == 4, \
                f"pixel_values invalid: {pixel_values}"
            gen_cfg = dict(max_new_tokens=self.max_new_tokens, do_sample=False)
            response = self.model.chat(self.tokenizer, pixel_values, prompt_text, gen_cfg)
            return response
        finally:
            torch.cuda.empty_cache()

    def get_model_info(self) -> Dict:
        return {"backend": "local_vlm", "model_type": "internvl2",
                "model_path": self.model_path, "torch_dtype": self.torch_dtype}
