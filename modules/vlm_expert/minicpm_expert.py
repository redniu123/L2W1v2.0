#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v5.1: MiniCPM-V Expert
兼容 MiniCPM-V-2_6-int4 / MiniCPM-V 4.5。
"""
from typing import Dict, Union
import numpy as np
from .base_expert import BaseVLMExpert


class MiniCPMVExpert(BaseVLMExpert):
    """MiniCPM-V 专家。"""

    def __init__(self, model_path: str, torch_dtype: str = "float16", max_new_tokens: int = 128):
        self.model_path = model_path
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None
        self._init_model()

    def _init_model(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        is_int4 = "int4" in self.model_path.lower()
        print(f"[MiniCPM-V] Loading {self.model_path} (int4={is_int4})...")

        load_kwargs = dict(
            device_map=None,
            trust_remote_code=True,
            low_cpu_mem_usage=False,
        )
        if not is_int4:
            load_kwargs["torch_dtype"] = torch.float16 if self.torch_dtype == "float16" else torch.bfloat16

        if hasattr(torch, "set_default_device"):
            torch.set_default_device("cpu")

        self.model = AutoModel.from_pretrained(self.model_path, **load_kwargs)
        if not hasattr(self.model, "all_tied_weights_keys"):
            self.model.all_tied_weights_keys = list(getattr(self.model, "_tied_weights_keys", []) or [])
        self.model = self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.to("cuda:0")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        print("[MiniCPM-V] Ready.")

    def chat_with_image(self, image_path: Union[str, np.ndarray], prompt_text: str) -> str:
        import torch
        from PIL import Image as PILImage

        try:
            if isinstance(image_path, str):
                pil_img = PILImage.open(image_path).convert("RGB")
            elif isinstance(image_path, np.ndarray):
                pil_img = PILImage.fromarray(image_path).convert("RGB")
            else:
                pil_img = image_path.convert("RGB")

            msgs = [{"role": "user", "content": [pil_img, prompt_text]}]
            response = self.model.chat(
                image=None,
                msgs=msgs,
                tokenizer=self.tokenizer,
                sampling=False,
                max_new_tokens=self.max_new_tokens,
            )
            return response
        finally:
            torch.cuda.empty_cache()

    def get_model_info(self) -> Dict:
        return {
            "backend": "local_vlm",
            "model_type": "minicpm_v",
            "model_path": self.model_path,
            "torch_dtype": self.torch_dtype,
        }
