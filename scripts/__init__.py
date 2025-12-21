# L2W1 执行脚本模块

"""
脚本模块说明:

1. data_pipeline.py - 自动化数据处理流水线
   - 加载 HCTR 数据集
   - 使用 Agent A 推理
   - 难例挖掘与错误索引定位
   - 生成 Agent B SFT 训练数据
   
   Usage:
       python data_pipeline.py --test  # 测试模式
       python data_pipeline.py --data_dir ./data/raw --batch_size 16

2. train_agent_b.py - Agent B SFT 训练脚本
   - 使用 SFT 数据微调 Qwen2.5-VL

3. evaluate.py - 评估指标计算脚本
   - CER (Character Error Rate)
   - OCR-R (Over-Correction Rate)
"""

from .data_pipeline import (
    DataPipeline,
    PipelineConfig,
    HCTRDatasetLoader,
    ErrorAnalyzer,
    SFTGenerator,
    DataSample,
    SFTConversation,
)

__all__ = [
    'DataPipeline',
    'PipelineConfig',
    'HCTRDatasetLoader',
    'ErrorAnalyzer',
    'SFTGenerator',
    'DataSample',
    'SFTConversation',
]
