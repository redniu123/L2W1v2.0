# -*- coding: utf-8 -*-
"""
SH-DA++ v5.1: VLM Expert Factory

根据 router_config.yaml 中的 agent_b 配置动态实例化 VLM 专家。
支持 9 款模型的热切换，只需修改 model_type 一行配置。

  model_type       expert class        典型模型
  -----------      ----------------    --------------------------------
  qwen2.5_vl       QwenVLExpert        Qwen2.5-VL-7B-Instruct / Qwen3-VL-8B
  qwen3.5          Qwen35Expert        Qwen3.5-9B
  internvl2        InternVL2Expert     InternVL2-8B-AWQ
  internvl2_5      InternVL2Expert     InternVL2_5-8B-AWQ
  minicpm_v        MiniCPMVExpert      MiniCPM-V-2_6-int4
  paddleocr_vl     PaddleOCRVLExpert   PaddleOCR-VL / PaddleOCR-VL-1.5
  llava            LLaVAExpert         llava-1.5-7b-hf
  smolvlm          SmolVLMExpert       SmolVLM-500M-Instruct
"""

from .base_expert import BaseVLMExpert
from .constrained_prompter import ConstrainedPrompter
from .gemini_expert import GeminiAgentB, GeminiConfig


class AgentBFactory:
    """
    Agent B 工厂类。
    根据 YAML config dict 中的 agent_b.backend / model_type 动态路由。
    """

    @staticmethod
    def create(config: dict) -> BaseVLMExpert:
        """
        Args:
            config: 顶层 YAML config dict（包含 agent_b 节点）
        Returns:
            BaseVLMExpert 子类实例
        """
        cfg = config.get("agent_b", {})
        backend = cfg.get("backend", "gemini")

        if backend == "gemini":
            return AgentBFactory._create_gemini(cfg)
        elif backend == "local_vlm":
            return AgentBFactory._create_local_vlm(cfg)
        else:
            raise ValueError(f"[AgentBFactory] Unknown backend: {backend}")

    @staticmethod
    def _create_gemini(cfg: dict) -> GeminiAgentB:
        gemini_config = GeminiConfig(model_name=cfg.get("model_name", "[V]gemini-3-flash-preview"))
        agent = GeminiAgentB(config=gemini_config)
        print(f"[AgentBFactory] Gemini: {gemini_config.model_name}")
        return agent

    @staticmethod
    def _create_local_vlm(cfg: dict) -> BaseVLMExpert:
        model_type = cfg.get("model_type", "qwen2.5_vl")
        model_path = cfg.get("model_path", "")
        torch_dtype = cfg.get("torch_dtype", "float16")  # 2080Ti 红线: float16
        max_new_tokens = cfg.get("max_new_tokens", 128)

        if not model_path:
            raise ValueError("[AgentBFactory] agent_b.model_path is required for local_vlm")

        print(f"[AgentBFactory] local_vlm | type={model_type} | dtype={torch_dtype} | path={model_path}")

        kwargs = dict(model_path=model_path, torch_dtype=torch_dtype, max_new_tokens=max_new_tokens)

        if model_type == "qwen2.5_vl":
            from .qwen_expert import QwenVLExpert
            return QwenVLExpert(**kwargs)

        elif model_type == "qwen3.5":
            from .qwen_expert import Qwen35Expert
            return Qwen35Expert(**kwargs)

        elif model_type in ("internvl2", "internvl2_5"):
            from .internvl_expert import InternVL2Expert
            return InternVL2Expert(**kwargs)

        elif model_type == "minicpm_v":
            from .minicpm_expert import MiniCPMVExpert
            return MiniCPMVExpert(**kwargs)

        elif model_type == "paddleocr_vl":
            from .paddleocr_vl_expert import PaddleOCRVLExpert
            return PaddleOCRVLExpert(**kwargs)

        elif model_type == "llava":
            from .llava_expert import LLaVAExpert
            return LLaVAExpert(**kwargs)

        elif model_type == "smolvlm":
            from .smolvlm_expert import SmolVLMExpert
            return SmolVLMExpert(**kwargs)

        else:
            raise ValueError(f"[AgentBFactory] Unknown model_type: {model_type}")


# 向后兼容旧导出
from .agent_b_expert import (
    AgentBExpert,
    AgentBExpertMock,
    AgentBConfig,
    EIPPromptTemplate,
    create_agent_b,
    process_hard_sample,
)

__all__ = [
    "AgentBFactory",
    "BaseVLMExpert",
    "AgentBExpert",
    "AgentBExpertMock",
    "AgentBConfig",
    "EIPPromptTemplate",
    "create_agent_b",
    "process_hard_sample",
    "ConstrainedPrompter",
]
