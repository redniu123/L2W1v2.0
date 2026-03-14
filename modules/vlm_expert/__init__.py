# -*- coding: utf-8 -*-
"""
SH-DA++ v5.1: VLM Expert Factory

根据 router_config.yaml 中的 agent_b 配置，动态实例化对应的 VLM 专家。

用法:
    from modules.vlm_expert import AgentBFactory

    expert = AgentBFactory.create(config)
    result = expert.process_hard_sample(image, manifest)
"""

from .base_expert import BaseVLMExpert
from .constrained_prompter import ConstrainedPrompter
from .gemini_expert import GeminiAgentB, GeminiConfig


class AgentBFactory:
    """
    Agent B 工厂类

    根据 YAML config dict 中的 agent_b.backend 和 agent_b.model_type
    动态实例化并返回对应的专家对象。

    支持的 backend:
      - "gemini"     : Gemini API (GeminiAgentB)
      - "local_vlm"  : 本地模型，由 model_type 决定具体实现
          - "qwen2.5_vl" : QwenVLExpert
          - "internvl2"  : InternVL2Expert
          - "minicpm_v"  : MiniCPMVExpert
    """

    @staticmethod
    def create(config: dict) -> BaseVLMExpert:
        """
        根据配置创建 VLM 专家实例。

        Args:
            config: 完整的 YAML config dict（顶层，包含 agent_b 节点）

        Returns:
            BaseVLMExpert 子类实例

        Raises:
            ValueError: 未知的 backend 或 model_type
        """
        agent_b_cfg = config.get("agent_b", {})
        backend = agent_b_cfg.get("backend", "gemini")

        if backend == "gemini":
            return AgentBFactory._create_gemini(agent_b_cfg)
        elif backend == "local_vlm":
            return AgentBFactory._create_local_vlm(agent_b_cfg)
        else:
            raise ValueError(f"[AgentBFactory] Unknown backend: {backend}")

    @staticmethod
    def _create_gemini(cfg: dict) -> GeminiAgentB:
        """创建 Gemini API 专家"""
        gemini_config = GeminiConfig(
            model_name=cfg.get("model_name", "[V]gemini-3-flash-preview"),
        )
        agent = GeminiAgentB(config=gemini_config)
        print(f"[AgentBFactory] Gemini backend: {gemini_config.model_name}")
        return agent

    @staticmethod
    def _create_local_vlm(cfg: dict) -> BaseVLMExpert:
        """创建本地 VLM 专家"""
        model_type = cfg.get("model_type", "qwen2.5_vl")
        model_path = cfg.get("model_path", "")
        dtype = cfg.get("dtype", "bfloat16")
        max_new_tokens = cfg.get("max_new_tokens", 128)

        if not model_path:
            raise ValueError("[AgentBFactory] agent_b.model_path is required for local_vlm backend")

        print(f"[AgentBFactory] local_vlm backend: {model_type} @ {model_path}")

        if model_type == "qwen2.5_vl":
            from .qwen_expert import QwenVLExpert
            return QwenVLExpert(model_path=model_path, dtype=dtype, max_new_tokens=max_new_tokens)

        elif model_type == "internvl2":
            from .internvl_expert import InternVL2Expert
            return InternVL2Expert(model_path=model_path, dtype=dtype, max_new_tokens=max_new_tokens)

        elif model_type == "minicpm_v":
            from .minicpm_expert import MiniCPMVExpert
            return MiniCPMVExpert(model_path=model_path, dtype=dtype, max_new_tokens=max_new_tokens)

        else:
            raise ValueError(f"[AgentBFactory] Unknown model_type: {model_type}")


# 保持向后兼容的旧导出
from .agent_b_expert import (
    AgentBExpert,
    AgentBExpertMock,
    AgentBConfig,
    EIPPromptTemplate,
    create_agent_b,
    process_hard_sample,
)

__all__ = [
    # 新工厂接口
    "AgentBFactory",
    "BaseVLMExpert",
    # 旧接口（向后兼容）
    "AgentBExpert",
    "AgentBExpertMock",
    "AgentBConfig",
    "EIPPromptTemplate",
    "create_agent_b",
    "process_hard_sample",
    # 工具
    "ConstrainedPrompter",
]
