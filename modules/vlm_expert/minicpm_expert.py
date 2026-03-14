#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v5.1: MiniCPM-V Expert
兼容 MiniCPM-V-2_6-int4（原生 int4，直接加载）。
"""
from typing import Dict, Union
import numpy as np
from .base_expert import BaseVLMExpert


class MiniCPMVExpert(BaseVLMExpert):
    """
    MiniCPM-V 2.6 专家
    int4 版本直接加载（无需 bitsandbytes，模型已预量化）。
    """

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
            device_map="auto",
            trust_remote_code=True,
        )
        # int4 预量化版本不需要指定 torch_dtype，直接加载
        if not is_int4:
            import torch
            load_kwargs["torch_dtype"] = torch.float16 if self.torch_dtype == "float16" else torch.bfloat16
        self.model = AutoModel.from_pretrained(self.model_path, **load_kwargs).eval()
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

            # MiniCPM-V int4 内部直接操作设备，需要确保在同一设备上
            # 用 cuda:0 相对索引（CUDA_VISIBLE_DEVICES 已做映射）
            with torch.cuda.device(0):
                response = self.model.chat(
                    image=None, msgs=msgs, tokenizer=self.tokenizer,
                    sampling=False, max_new_tokens=self.max_new_tokens,
                )
            return response
        finally:
            torch.cuda.empty_cache()

    def get_model_info(self) -> Dict:
        return {"backend": "local_vlm", "model_type": "minicpm_v",
                "model_path": self.model_path, "torch_dtype": self.torch_dtype}
