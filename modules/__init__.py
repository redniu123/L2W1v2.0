# L2W1 核心模块

"""
SH-DA++ v4.0 分层多智能体 OCR 系统

核心模块:
- paddle_engine: Agent A (PP-OCRv5) - 全量扫描与 Logits 导出
- router: Router - 风险评分与路由决策
- vlm_expert: Agent B (Qwen2.5-VL) - 视觉专家精准重写

Stage 2 新增:
- pipeline_stage2: 集成 Pipeline（严格回填、熔断监控）
"""

# 导入子模块
from . import paddle_engine
from . import router
from . import vlm_expert

# 导入 Stage 2 Pipeline
from .pipeline_stage2 import SHDAPipeline, PipelineConfig, PipelineResult

__all__ = [
    # 子模块
    'paddle_engine',
    'router',
    'vlm_expert',

    # Stage 2 Pipeline
    'SHDAPipeline',
    'PipelineConfig',
    'PipelineResult',
]
