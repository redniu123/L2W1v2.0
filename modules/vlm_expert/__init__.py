# Agent B: 动态分辨率配置与显式索引推理逻辑

"""
L2W1 Agent B 视觉专家模块

核心组件:
- AgentBExpert: 基于 Qwen2.5-VL-3B 的视觉专家
- AgentBExpertMock: 模拟版本（用于测试）
- AgentBConfig: 配置类
- EIPPromptTemplate: 显式索引提示模板

关键特性:
1. 4-bit 量化：适配 11GB 显存 (RTX 2080Ti)
2. 动态分辨率：保留原始长宽比，避免几何失真
3. Flash Attention 2：加速长序列推理
4. EIP 策略：显式索引引导的定点纠错

动态分辨率配置:
- min_pixels: 256 × 28 × 28 = 200,704
- max_pixels: 1280 × 28 × 28 = 1,003,520
- 支持极端长宽比 (最高 20:1)

使用示例:
```python
from L2W1.modules.vlm_expert import AgentBExpert, AgentBConfig

# 配置
config = AgentBConfig(
    model_path="Qwen/Qwen2.5-VL-3B-Instruct",
    use_4bit=True,
    use_flash_attention=True,
)

# 初始化
agent = AgentBExpert(config)

# 处理困难样本
result = agent.process_hard_sample(
    image="./line_image.jpg",
    manifest={
        'ocr_text': "在时间的未尾",
        'suspicious_index': 4,
        'suspicious_char': '未',
        'risk_level': 'high',
    }
)

print(f"修正结果: {result['corrected_text']}")
```
"""

from .agent_b_expert import (
    # 核心类
    AgentBExpert,
    AgentBExpertMock,
    AgentBConfig,
    EIPPromptTemplate,
    
    # 便捷函数
    create_agent_b,
    process_hard_sample,
)

__all__ = [
    # 核心类
    'AgentBExpert',
    'AgentBExpertMock',
    'AgentBConfig',
    'EIPPromptTemplate',
    
    # 便捷函数
    'create_agent_b',
    'process_hard_sample',
]
