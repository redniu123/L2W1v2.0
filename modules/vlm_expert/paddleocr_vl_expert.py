#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v5.1: PaddleOCR-VL Expert
兼容 PaddlePaddle/PaddleOCR-VL 和 PaddleOCR-VL-1.5。
使用 transformers AutoModel 加载（PaddleNLP 风格）。
"""
from typing import Dict, Union
import numpy as np
from .base_expert import BaseVLMExpert


class PaddleOCRVLExpert(BaseVLMExpert):
    """
    PaddleOCR-VL 专家包装器。
    PaddleOCR-VL 系列已支持 transformers 加载方式。
    """

    def __init__(self, model_path: str, torch_dtype: str = "float16", max_new_tokens: int = 128):
        self.model_path = model_path
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None
        self._init_model()

    def _init_model(self):
        import torch
        from transformers import AutoModel, AutoProcessor
        dtype = torch.float16 if self.torch_dtype == "float16" else torch.bfloat16
        print(f"[PaddleOCR-VL] Loading {self.model_path} ({self.torch_dtype})...")
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        print("[PaddleOCR-VL] Ready.")

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

            messages = [{"role": "user", "content": [
                {"type": "image", "image": pil_img},
                {"type": "text", "text": prompt_text},
            ]}]

            # 尝试标准 chat_template 方式
            try:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.processor(
                    text=[text], images=[pil_img], return_tensors="pt", padding=True
                ).to(self.model.device)
                with torch.no_grad():
                    gen_ids = self.model.generate(
                        **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
                    )
                trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
                return self.processor.batch_decode(
                    trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]
            except Exception:
                # 降级：尝试 model.chat() 接口
                response = self.model.chat(
                    self.processor, pil_img, prompt_text,
                    max_new_tokens=self.max_new_tokens
                )
                return response if isinstance(response, str) else str(response)
        finally:
            torch.cuda.empty_cache()

    def get_model_info(self) -> Dict:
        return {"backend": "local_vlm", "model_type": "paddleocr_vl",
                "model_path": self.model_path, "torch_dtype": self.torch_dtype}
