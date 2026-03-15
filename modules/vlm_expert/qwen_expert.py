#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v5.1: Qwen Expert
兼容 Qwen2.5-VL-7B-Instruct 和 Qwen3.5-9B。
显存策略: float16 + device_map=auto (2080Ti Turing 禁用 bfloat16)。
"""
from typing import Dict, Union
import numpy as np
from .base_expert import BaseVLMExpert


class QwenVLExpert(BaseVLMExpert):
    """Qwen2.5-VL 多模态专家"""

    def __init__(self, model_path: str, torch_dtype: str = "float16", max_new_tokens: int = 128):
        self.model_path = model_path
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.processor = None
        self._init_model()

    def _init_model(self):
        import torch
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        dtype = torch.float16 if self.torch_dtype == "float16" else torch.bfloat16
        print(f"[QwenVL] Loading {self.model_path} ({self.torch_dtype})...")
        attn = "sdpa"
        try:
            import flash_attn  # noqa
            attn = "flash_attention_2"
            print("[QwenVL] Flash Attention 2 enabled")
        except ImportError:
            pass
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype=dtype, device_map="auto",
            attn_implementation=attn, trust_remote_code=True,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=True,
            min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28,
        )
        print("[QwenVL] Ready.")

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
            text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = self.processor(text=[text], images=[pil_img], padding=True, return_tensors="pt")
            inputs = inputs.to(self.model.device)
            with torch.no_grad():
                gen_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
            trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen_ids)]
            return self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        finally:
            torch.cuda.empty_cache()

    def get_model_info(self) -> Dict:
        return {"backend": "local_vlm", "model_type": "qwen2.5_vl",
                "model_path": self.model_path, "torch_dtype": self.torch_dtype}


class Qwen35Expert(BaseVLMExpert):
    """Qwen3.5-9B 纯文本专家（无视觉，提示词中嵌入 OCR 文本）"""

    def __init__(self, model_path: str, torch_dtype: str = "float16", max_new_tokens: int = 128):
        self.model_path = model_path
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None
        self._init_model()

    def _init_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        dtype = torch.float16 if self.torch_dtype == "float16" else torch.bfloat16
        print(f"[Qwen3.5] Loading {self.model_path} ({self.torch_dtype})...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, torch_dtype=dtype, device_map="auto", trust_remote_code=True,
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        print("[Qwen3.5] Ready.")

    def chat_with_image(self, image_path: Union[str, np.ndarray], prompt_text: str) -> str:
        """Qwen3.5 无视觉，仅使用文本 prompt，加 /no_think 禁用思维链"""
        import torch
        try:
            # /no_think 告知 Qwen3.5 直接输出答案，不输出思维链
            no_think_prompt = prompt_text.rstrip() + " /no_think"
            messages = [{"role": "user", "content": no_think_prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                gen_ids = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
                )
            out = gen_ids[0][inputs.input_ids.shape[1]:]
            return self.tokenizer.decode(out, skip_special_tokens=True)
        finally:
            torch.cuda.empty_cache()

    def get_model_info(self) -> Dict:
        return {"backend": "local_vlm", "model_type": "qwen3.5",
                "model_path": self.model_path, "torch_dtype": self.torch_dtype}
