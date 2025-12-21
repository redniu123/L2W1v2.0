"""
L2W1 完整推理流水线

将 Agent A、Router、Agent B 三个组件串联，实现端到端的手写识别与纠错

流程:
1. Agent A (PP-OCRv5): 全量行级扫描，输出文本 + Logits
2. Router: 计算视觉熵和语义 PPL，决策是否调用 Agent B
3. Agent B (Qwen2.5-VL): 对困难样本进行精准视觉重写

架构:
    Input Image
         │
         ▼
    ┌─────────────┐
    │   Agent A   │ ──────► text, logits
    └─────────────┘
         │
         ▼
    ┌─────────────┐
    │   Router    │ ──────► is_hard, suspicious_index
    └─────────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
  Easy      Hard
    │         │
    │    ┌────┴────┐
    │    │ Agent B │ ──────► corrected_text
    │    └─────────┘
    │         │
    └────┬────┘
         ▼
    Final Output
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import numpy as np

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class PipelineConfig:
    """流水线配置"""
    # Agent A 配置
    agent_a_model_dir: str = ""
    agent_a_batch_size: int = 16
    
    # Router 配置
    entropy_threshold_low: float = 2.0
    entropy_threshold_high: float = 4.0
    ppl_threshold_low: float = 50.0
    ppl_threshold_high: float = 200.0
    
    # Agent B 配置
    agent_b_model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    agent_b_use_4bit: bool = True
    agent_b_use_flash_attention: bool = True
    
    # 流水线配置
    use_mock: bool = False  # 是否使用模拟模式
    verbose: bool = False   # 是否打印详细日志


@dataclass
class PipelineResult:
    """流水线结果"""
    # 输入信息
    image_path: str = ""
    
    # Agent A 输出
    agent_a_text: str = ""
    agent_a_confidence: float = 0.0
    agent_a_logits: Optional[np.ndarray] = None
    
    # Router 输出
    is_hard: bool = False
    suspicious_index: int = -1
    suspicious_char: str = ""
    risk_level: str = "low"
    visual_entropy: float = 0.0
    semantic_ppl: float = 0.0
    
    # Agent B 输出
    agent_b_text: str = ""
    agent_b_is_corrected: bool = False
    
    # 最终输出
    final_text: str = ""
    
    # 元信息
    routed_to_agent_b: bool = False
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "image_path": self.image_path,
            "agent_a": {
                "text": self.agent_a_text,
                "confidence": self.agent_a_confidence,
            },
            "router": {
                "is_hard": self.is_hard,
                "suspicious_index": self.suspicious_index,
                "suspicious_char": self.suspicious_char,
                "risk_level": self.risk_level,
                "visual_entropy": round(self.visual_entropy, 4),
                "semantic_ppl": round(self.semantic_ppl, 4),
            },
            "agent_b": {
                "text": self.agent_b_text,
                "is_corrected": self.agent_b_is_corrected,
            },
            "final_text": self.final_text,
            "routed_to_agent_b": self.routed_to_agent_b,
        }


class L2W1Pipeline:
    """
    L2W1 完整推理流水线
    
    实现分层多智能体架构：
    - Agent A (System 1): 快速扫描
    - Router: 不确定性评估与路由
    - Agent B (System 2): 精细重写
    """
    
    def __init__(self, config: PipelineConfig = None):
        """
        Args:
            config: 流水线配置
        """
        self.config = config or PipelineConfig()
        
        # 组件（延迟初始化）
        self._agent_a = None
        self._router = None
        self._agent_b = None
        
        # 统计
        self.stats = {
            "total_processed": 0,
            "routed_to_agent_b": 0,
            "corrected_by_agent_b": 0,
        }
    
    @property
    def agent_a(self):
        """获取 Agent A"""
        if self._agent_a is None:
            self._init_agent_a()
        return self._agent_a
    
    @property
    def router(self):
        """获取 Router"""
        if self._router is None:
            self._init_router()
        return self._router
    
    @property
    def agent_b(self):
        """获取 Agent B"""
        if self._agent_b is None:
            self._init_agent_b()
        return self._agent_b
    
    def _init_agent_a(self):
        """初始化 Agent A"""
        if self.config.use_mock:
            self._agent_a = MockAgentA()
        else:
            try:
                from modules.paddle_engine.predict_rec_modified import TextRecognizerWithLogits
                # TODO: 实际初始化逻辑
                self._agent_a = MockAgentA()  # 暂时使用 Mock
            except ImportError:
                print("[Warning] 无法加载 Agent A，使用模拟模式")
                self._agent_a = MockAgentA()
    
    def _init_router(self):
        """初始化 Router"""
        from modules.router import UncertaintyRouter, RouterConfig
        
        router_config = RouterConfig(
            entropy_threshold_low=self.config.entropy_threshold_low,
            entropy_threshold_high=self.config.entropy_threshold_high,
            ppl_threshold_low=self.config.ppl_threshold_low,
            ppl_threshold_high=self.config.ppl_threshold_high,
        )
        self._router = UncertaintyRouter(config=router_config)
    
    def _init_agent_b(self):
        """初始化 Agent B"""
        if self.config.use_mock:
            from modules.vlm_expert import AgentBExpertMock
            self._agent_b = AgentBExpertMock()
        else:
            try:
                from modules.vlm_expert import AgentBExpert, AgentBConfig
                
                agent_b_config = AgentBConfig(
                    model_path=self.config.agent_b_model_path,
                    use_4bit=self.config.agent_b_use_4bit,
                    use_flash_attention=self.config.agent_b_use_flash_attention,
                )
                self._agent_b = AgentBExpert(config=agent_b_config)
            except Exception as e:
                print(f"[Warning] 无法加载 Agent B: {e}，使用模拟模式")
                from modules.vlm_expert import AgentBExpertMock
                self._agent_b = AgentBExpertMock()
    
    def process(
        self,
        image: Union[str, np.ndarray],
        image_path: str = ""
    ) -> PipelineResult:
        """
        处理单张图像
        
        Args:
            image: 图像路径或 numpy 数组
            image_path: 图像路径（用于记录）
            
        Returns:
            PipelineResult: 处理结果
        """
        import time
        start_time = time.time()
        
        result = PipelineResult(image_path=image_path or str(image))
        
        # Step 1: Agent A 推理
        if self.config.verbose:
            print("[Pipeline] Step 1: Agent A 推理...")
        
        agent_a_output = self.agent_a.recognize(image)
        result.agent_a_text = agent_a_output.get('text', '')
        result.agent_a_confidence = agent_a_output.get('confidence', 0.0)
        result.agent_a_logits = agent_a_output.get('logits', None)
        
        # Step 2: Router 决策
        if self.config.verbose:
            print("[Pipeline] Step 2: Router 决策...")
        
        if result.agent_a_logits is not None:
            routing_result = self.router.route(
                logits=result.agent_a_logits,
                text=result.agent_a_text,
                confidence=result.agent_a_confidence
            )
            
            result.is_hard = routing_result.is_hard
            result.suspicious_index = routing_result.suspicious_index
            result.suspicious_char = routing_result.suspicious_char
            result.risk_level = routing_result.risk_level
            result.visual_entropy = routing_result.visual_entropy
            result.semantic_ppl = routing_result.semantic_ppl
        else:
            # 无 logits 时，使用置信度判断
            result.is_hard = result.agent_a_confidence < 0.8
        
        # Step 3: 条件调用 Agent B
        if result.is_hard:
            if self.config.verbose:
                print("[Pipeline] Step 3: 调用 Agent B...")
            
            result.routed_to_agent_b = True
            self.stats["routed_to_agent_b"] += 1
            
            manifest = {
                'ocr_text': result.agent_a_text,
                'suspicious_index': result.suspicious_index,
                'suspicious_char': result.suspicious_char,
                'risk_level': result.risk_level,
            }
            
            agent_b_output = self.agent_b.process_hard_sample(image, manifest)
            result.agent_b_text = agent_b_output.get('corrected_text', '')
            result.agent_b_is_corrected = agent_b_output.get('is_corrected', False)
            
            if result.agent_b_is_corrected:
                self.stats["corrected_by_agent_b"] += 1
            
            # 最终文本采用 Agent B 输出
            result.final_text = result.agent_b_text
        else:
            if self.config.verbose:
                print("[Pipeline] Step 3: 跳过 Agent B (简单样本)")
            
            # 最终文本采用 Agent A 输出
            result.final_text = result.agent_a_text
        
        result.processing_time = time.time() - start_time
        self.stats["total_processed"] += 1
        
        return result
    
    def process_batch(
        self,
        images: List[Union[str, np.ndarray]],
        image_paths: List[str] = None
    ) -> List[PipelineResult]:
        """
        批量处理
        
        Args:
            images: 图像列表
            image_paths: 图像路径列表
            
        Returns:
            结果列表
        """
        if image_paths is None:
            image_paths = [str(img) for img in images]
        
        results = []
        for image, path in zip(images, image_paths):
            result = self.process(image, path)
            results.append(result)
        
        return results
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = self.stats.copy()
        
        if stats["total_processed"] > 0:
            stats["agent_b_routing_rate"] = stats["routed_to_agent_b"] / stats["total_processed"]
            stats["agent_b_correction_rate"] = (
                stats["corrected_by_agent_b"] / stats["routed_to_agent_b"]
                if stats["routed_to_agent_b"] > 0 else 0.0
            )
        
        return stats


class MockAgentA:
    """Agent A 模拟版本"""
    
    def recognize(self, image: Union[str, np.ndarray]) -> Dict:
        """模拟识别"""
        import random
        
        # 模拟识别结果
        sample_texts = [
            "中国科学院计算技术研究所",
            "在时间的未尾",
            "深度学习模型训练",
            "自然语言处理技术",
        ]
        
        text = random.choice(sample_texts)
        
        # 模拟 logits
        seq_len = 80
        vocab_size = 6625
        logits = np.random.randn(seq_len, vocab_size).astype(np.float32) * 0.5
        
        return {
            'text': text,
            'confidence': random.uniform(0.7, 0.95),
            'logits': logits,
        }


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("L2W1 完整推理流水线测试")
    print("=" * 60)
    
    # 配置
    config = PipelineConfig(
        use_mock=True,  # 使用模拟模式
        verbose=True,
    )
    
    # 创建流水线
    pipeline = L2W1Pipeline(config)
    
    # 测试单张图像
    print("\n[测试] 处理单张图像...")
    result = pipeline.process("./test_image.jpg")
    
    print(f"\n[结果]")
    print(f"  Agent A 输出: '{result.agent_a_text}' (置信度: {result.agent_a_confidence:.2%})")
    print(f"  Router 决策: is_hard={result.is_hard}, risk_level={result.risk_level}")
    
    if result.routed_to_agent_b:
        print(f"  Agent B 输出: '{result.agent_b_text}' (已修正: {result.agent_b_is_corrected})")
    
    print(f"  最终文本: '{result.final_text}'")
    print(f"  处理时间: {result.processing_time:.3f}s")
    
    # 测试批量处理
    print("\n[测试] 批量处理...")
    images = ["./img1.jpg", "./img2.jpg", "./img3.jpg"]
    results = pipeline.process_batch(images)
    
    for i, r in enumerate(results):
        status = "→ Agent B" if r.routed_to_agent_b else "→ Agent A"
        print(f"  样本 {i+1}: '{r.final_text}' {status}")
    
    # 统计信息
    print("\n[统计]")
    stats = pipeline.get_stats()
    print(f"  总处理: {stats['total_processed']}")
    print(f"  路由到 Agent B: {stats['routed_to_agent_b']}")
    print(f"  Agent B 修正: {stats['corrected_by_agent_b']}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

