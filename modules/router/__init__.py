# Router: 视觉熵计算与语义 PPL 评估器

"""
L2W1 不确定性路由器模块

核心组件:
- UncertaintyRouter: 完整路由器，整合视觉熵和语义 PPL
- CTCAligner: CTC 时间步对齐器
- VisualEntropyCalculator: 视觉熵计算器
- SemanticPPLCalculator: 语义困惑度计算器

便捷函数:
- calculate_visual_entropy(logits, text): 计算视觉熵并返回存疑位置
- calculate_ppl(text): 计算文本困惑度
- create_routing_manifest(logits, text, confidence): 创建路由 Manifest

配置类:
- RouterConfig: 路由器阈值配置
- RoutingResult: 路由结果数据类

使用示例:
```python
from L2W1.modules.router import UncertaintyRouter, calculate_visual_entropy

# 方式 1: 使用便捷函数
result = calculate_visual_entropy(logits, text)
print(f"存疑位置: {result['suspicious_index']}")
print(f"存疑字符: {result['suspicious_char']}")

# 方式 2: 使用完整路由器
router = UncertaintyRouter()
routing = router.route(logits, text, confidence=0.85)
if routing.is_hard:
    print(f"需要 Agent B 处理: 第 {routing.suspicious_index + 1} 个字符存疑")
```
"""

from .uncertainty_router import (
    # 核心类
    UncertaintyRouter,
    CTCAligner,
    VisualEntropyCalculator,
    SemanticPPLCalculator,
    
    # 配置和结果类
    RouterConfig,
    RoutingResult,
    RiskLevel,
    
    # 便捷函数
    calculate_visual_entropy,
    calculate_ppl,
    create_routing_manifest,
)

__all__ = [
    # 核心类
    'UncertaintyRouter',
    'CTCAligner',
    'VisualEntropyCalculator',
    'SemanticPPLCalculator',
    
    # 配置和结果类
    'RouterConfig',
    'RoutingResult',
    'RiskLevel',
    
    # 便捷函数
    'calculate_visual_entropy',
    'calculate_ppl',
    'create_routing_manifest',
]
