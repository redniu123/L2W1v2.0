#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v5.1: SmolVLM Expert
兼容 HuggingFaceTB/SmolVLM-500M-Instruct。
轻量级 500M 模型，适合快速冒烟测试。
"""
from typing import Dict, Union
import numpy as np
from .base_expert import BaseVLMExpert


class SmolVLMExpert(BaseVLMExpert):
    """
    SmolVLM-500M 专家包装器。
    使用 AutoModelForVision2Seq + AutoProcessor。
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
        from transformers import AutoModelForVision2Seq, AutoProcessor
        dtype = torch.float16 if self.torch_dtype == "float16" else torch.bfloat16
        print(f"[SmolVLM] Loading {self.model_path} ({self.torch_dtype})...")
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map="auto",
            _attn_implementation="eager",  # 500M 不支持 flash_attention_2
        ).eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        print("[SmolVLM] Ready.")

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

            # SmolVLM 能力有限，使用极简 prompt 提升指令遵循率
            # 从 prompt_text 中提取 OCR 文本（【 xxx 】格式），构建简短指令
            import re
            m = re.search(r'\u3010\s*(.+?)\s*\u3011', prompt_text)
            ocr_text = m.group(1).strip() if m else prompt_text[:50]
            simple_prompt = f'This is an OCR result: "{ocr_text}". Look at the image and correct any errors. Output only the corrected text, nothing else.'

            messages = [{"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": simple_prompt},
            ]}]
            prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = self.processor(
                text=prompt, images=[pil_img], return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                gen_ids = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
                )
            new_tokens = gen_ids[0][inputs["input_ids"].shape[1]:]
            return self.processor.decode(new_tokens, skip_special_tokens=True).strip()
        finally:
            torch.cuda.empty_cache()

    def get_model_info(self) -> Dict:
        return {"backend": "local_vlm", "model_type": "smolvlm",
                "model_path": self.model_path, "torch_dtype": self.torch_dtype}
