# L2W1 核心模块

"""
L2W1 分层多智能体 OCR 系统

核心模块:
- paddle_engine: Agent A (PP-OCRv5) - 全量扫描与 Logits 导出
- router: Router - 视觉熵计算与路由决策
- vlm_expert: Agent B (Qwen2.5-VL) - 视觉专家精准重写

完整流水线:
- L2W1Pipeline: 端到端推理流水线，整合三个组件

架构图:
    Input Image
         │
         ▼
    ┌─────────────┐
    │   Agent A   │ ──► text, logits
    │  (PP-OCRv5) │
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │   Router    │ ──► is_hard, suspicious_index
    │ (Entropy+PPL)│
    └─────────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
  Easy      Hard
    │         │
    │    ┌────┴────────┐
    │    │   Agent B   │ ──► corrected_text
    │    │ (Qwen2.5-VL)│
    │    └─────────────┘
    │         │
    └────┬────┘
         ▼
    Final Output

使用示例:
```python
from L2W1.modules import L2W1Pipeline, PipelineConfig

# 配置流水线
config = PipelineConfig(
    use_mock=False,
    agent_b_use_4bit=True,
)

# 创建流水线
pipeline = L2W1Pipeline(config)

# 处理图像
result = pipeline.process("./line_image.jpg")

print(f"最终文本: {result.final_text}")
print(f"是否经过 Agent B: {result.routed_to_agent_b}")
```
"""

# 导入子模块
from . import paddle_engine
from . import router
from . import vlm_expert

# 导入流水线
from .pipeline import L2W1Pipeline, PipelineConfig, PipelineResult

__all__ = [
    # 子模块
    'paddle_engine',
    'router',
    'vlm_expert',
    
    # 流水线
    'L2W1Pipeline',
    'PipelineConfig',
    'PipelineResult',
]
