"""
L2W1 不确定性路由器 (Uncertainty Router)

核心功能:
1. 视觉不确定性 (U_vis): 计算 CTC Logits 的 Shannon Entropy，并对齐到字符位置
2. 语义不确定性 (U_sem): 计算 OCR 文本的 Perplexity (PPL)
3. 路由决策: 判断是否需要调用 Agent B 进行精细处理

技术核心 - CTC 时间步对齐:
- Agent A 输出的 logits 序列长度固定为 ~80
- 识别出的文本可能只有 10 个字符
- 需要利用 CTC Decoder 逻辑，将熵值精确映射到字符位置

公式:
- Visual Entropy: H = -Σ(P · log(P + ε))
- Semantic PPL: PPL = exp(1/M · Σ CrossEntropy(T_ocr | LM))
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    """风险等级枚举"""
    LOW = "low"           # 低风险：Agent A 结果可信
    MEDIUM = "medium"     # 中风险：单一指标超阈值
    HIGH = "high"         # 高风险：多指标超阈值，需 Agent B 介入
    CRITICAL = "critical" # 极高风险：严重不确定性


@dataclass
class RouterConfig:
    """路由器配置"""
    # 视觉熵阈值
    entropy_threshold_low: float = 2.0     # 低于此值视为高置信度
    entropy_threshold_high: float = 4.0    # 高于此值视为低置信度
    
    # 语义困惑度阈值
    ppl_threshold_low: float = 50.0        # 低于此值视为流畅文本
    ppl_threshold_high: float = 200.0      # 高于此值视为异常文本
    
    # CTC 解码参数
    blank_idx: int = 0                     # CTC blank 符号索引
    
    # 熵计算参数
    epsilon: float = 1e-10                 # 防止 log(0)


@dataclass
class RoutingResult:
    """路由结果"""
    is_hard: bool                          # 是否为困难样本
    suspicious_index: int                  # 存疑字符位置 (0-indexed)
    suspicious_char: str                   # 存疑字符
    risk_level: str                        # 风险等级
    visual_entropy: float                  # 视觉熵均值
    max_char_entropy: float                # 最大字符熵值
    semantic_ppl: float                    # 语义困惑度
    entropy_sequence: List[float]          # 字符级熵序列
    
    def to_dict(self) -> Dict:
        """转换为 Manifest Task JSON 格式"""
        return {
            "is_hard": self.is_hard,
            "suspicious_index": self.suspicious_index,
            "suspicious_char": self.suspicious_char,
            "risk_level": self.risk_level,
            "metrics": {
                "visual_entropy": round(self.visual_entropy, 4),
                "max_char_entropy": round(self.max_char_entropy, 4),
                "semantic_ppl": round(self.semantic_ppl, 4),
            },
            "entropy_sequence": [round(e, 4) for e in self.entropy_sequence]
        }


class CTCAligner:
    """
    CTC 时间步对齐器
    
    将固定长度的 logits 序列 (e.g., 80) 映射到可变长度的字符序列 (e.g., 10)
    
    核心逻辑:
    1. 对每个时间步取 argmax 得到预测索引
    2. 应用 CTC 解码规则：移除连续重复 + 移除 blank
    3. 记录每个最终字符对应的时间步范围
    4. 将时间步的熵值聚合到对应字符
    """
    
    def __init__(self, blank_idx: int = 0):
        """
        Args:
            blank_idx: CTC blank 符号的索引，PaddleOCR 默认为 0
        """
        self.blank_idx = blank_idx
    
    def align(self, logits: np.ndarray, text: str) -> List[Tuple[int, List[int]]]:
        """
        对齐 logits 时间步到字符位置
        
        Args:
            logits: 原始 logits，形状 [Seq_Len, Vocab_Size]
            text: 识别出的文本字符串
            
        Returns:
            List[Tuple[char_idx, List[timestep_indices]]]:
                每个字符对应的时间步索引列表
        """
        seq_len, vocab_size = logits.shape
        
        # Step 1: 对每个时间步取 argmax
        pred_indices = np.argmax(logits, axis=-1)  # [Seq_Len,]
        
        # Step 2: CTC 解码 - 找到非 blank 且发生字符变更的时间步
        char_to_timesteps = []
        current_char_idx = -1
        current_timesteps = []
        prev_idx = -1
        
        for t, idx in enumerate(pred_indices):
            if idx == self.blank_idx:
                # 遇到 blank，结束当前字符的时间步收集
                if current_timesteps:
                    char_to_timesteps.append((current_char_idx, current_timesteps))
                    current_timesteps = []
                prev_idx = idx
                continue
            
            if idx != prev_idx:
                # 字符变更，开始新字符
                if current_timesteps:
                    char_to_timesteps.append((current_char_idx, current_timesteps))
                current_char_idx += 1
                current_timesteps = [t]
            else:
                # 连续相同字符，累加时间步
                current_timesteps.append(t)
            
            prev_idx = idx
        
        # 处理最后一个字符
        if current_timesteps:
            char_to_timesteps.append((current_char_idx, current_timesteps))
        
        # Step 3: 验证对齐结果
        # 如果解码出的字符数与文本长度不匹配，使用均匀分配
        if len(char_to_timesteps) != len(text):
            return self._fallback_align(seq_len, text)
        
        return char_to_timesteps
    
    def _fallback_align(self, seq_len: int, text: str) -> List[Tuple[int, List[int]]]:
        """
        回退对齐策略：均匀分配时间步到字符
        
        当 CTC 解码结果与文本长度不匹配时使用
        """
        if len(text) == 0:
            return []
        
        char_to_timesteps = []
        steps_per_char = seq_len / len(text)
        
        for char_idx in range(len(text)):
            start_t = int(char_idx * steps_per_char)
            end_t = int((char_idx + 1) * steps_per_char)
            timesteps = list(range(start_t, min(end_t, seq_len)))
            if not timesteps:
                timesteps = [min(start_t, seq_len - 1)]
            char_to_timesteps.append((char_idx, timesteps))
        
        return char_to_timesteps


class VisualEntropyCalculator:
    """
    视觉不确定性计算器
    
    基于 CTC Logits 计算 Shannon Entropy，并对齐到字符级别
    """
    
    def __init__(self, config: RouterConfig = None):
        self.config = config or RouterConfig()
        self.aligner = CTCAligner(blank_idx=self.config.blank_idx)
    
    def compute_timestep_entropy(self, logits: np.ndarray) -> np.ndarray:
        """
        计算每个时间步的 Shannon Entropy
        
        Args:
            logits: 原始 logits，形状 [Seq_Len, Vocab_Size]
            
        Returns:
            entropy: 熵序列，形状 [Seq_Len,]
        """
        # Softmax 归一化 (数值稳定版本)
        logits_max = np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits - logits_max)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        
        # Shannon Entropy: H = -Σ(P · log(P + ε))
        entropy = -np.sum(probs * np.log(probs + self.config.epsilon), axis=-1)
        
        return entropy
    
    def compute_char_entropy(
        self, 
        logits: np.ndarray, 
        text: str
    ) -> Tuple[List[float], int, float]:
        """
        计算字符级熵，并找出最高熵的字符位置
        
        Args:
            logits: 原始 logits，形状 [Seq_Len, Vocab_Size]
            text: 识别出的文本
            
        Returns:
            Tuple[char_entropies, suspicious_idx, max_entropy]:
                - char_entropies: 每个字符的熵值列表
                - suspicious_idx: 最高熵字符的索引
                - max_entropy: 最高熵值
        """
        if len(text) == 0:
            return [], -1, 0.0
        
        # 计算时间步级熵
        timestep_entropy = self.compute_timestep_entropy(logits)
        
        # CTC 对齐：时间步 -> 字符
        alignment = self.aligner.align(logits, text)
        
        # 聚合每个字符的熵值（取最大值，因为我们关注最不确定的时刻）
        char_entropies = []
        for char_idx, timesteps in alignment:
            if timesteps:
                # 使用最大熵值代表该字符的不确定性
                char_entropy = np.max(timestep_entropy[timesteps])
                char_entropies.append(float(char_entropy))
            else:
                char_entropies.append(0.0)
        
        # 找出最高熵的字符
        if char_entropies:
            suspicious_idx = int(np.argmax(char_entropies))
            max_entropy = max(char_entropies)
        else:
            suspicious_idx = -1
            max_entropy = 0.0
        
        return char_entropies, suspicious_idx, max_entropy


class SemanticPPLCalculator:
    """
    语义困惑度计算器
    
    使用语言模型评估 OCR 文本的流畅度
    困惑度高 = 文本不流畅/可能有识别错误
    """
    
    def __init__(self, model_path: str = None, use_gpu: bool = True):
        """
        Args:
            model_path: 语言模型路径 (e.g., Qwen2.5-0.5B)
            use_gpu: 是否使用 GPU
        """
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.model = None
        self.tokenizer = None
        self._initialized = False
    
    def _lazy_init(self):
        """延迟初始化模型"""
        if self._initialized:
            return
        
        if self.model_path:
            try:
                self._init_transformer_model()
            except Exception as e:
                print(f"[Warning] Failed to load LM model: {e}")
                print("[Fallback] Using simple n-gram based PPL estimation")
        
        self._initialized = True
    
    def _init_transformer_model(self):
        """初始化 Transformer 语言模型"""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.use_gpu else torch.float32,
                device_map="auto" if self.use_gpu else "cpu"
            )
            self.model.eval()
        except ImportError:
            raise ImportError("transformers and torch are required for PPL calculation")
    
    def calculate(self, text: str) -> float:
        """
        计算文本的困惑度
        
        Args:
            text: OCR 识别文本
            
        Returns:
            ppl: 困惑度值，越低表示文本越流畅
        """
        if not text or len(text) < 2:
            return 1.0
        
        self._lazy_init()
        
        if self.model is not None:
            return self._calculate_transformer_ppl(text)
        else:
            return self._calculate_simple_ppl(text)
    
    def _calculate_transformer_ppl(self, text: str) -> float:
        """使用 Transformer 模型计算 PPL"""
        import torch
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt")
            if self.use_gpu:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
            
            ppl = torch.exp(loss).item()
            return min(ppl, 10000.0)  # 限制最大值
        except Exception as e:
            print(f"[Warning] PPL calculation failed: {e}")
            return self._calculate_simple_ppl(text)
    
    def _calculate_simple_ppl(self, text: str) -> float:
        """
        简单的 PPL 估计（基于字符频率和 n-gram）
        
        这是一个占位实现，用于在无 LM 模型时提供基本的语义评估
        """
        # 常见中文字符集（高频字）
        common_chars = set(
            "的一是不了在人有我他这个们中来上大为和国地到以说时要就出会可也你对生能而子那得于着下自之年过发后作里用道行所然家种事"
            "成方多经么同面当起与好看学进着种将还等此心前为所以因把第二三四五六七八九十百千万"
        )
        
        # 计算非常见字符比例
        if len(text) == 0:
            return 1.0
        
        uncommon_count = sum(1 for c in text if c not in common_chars and '\u4e00' <= c <= '\u9fff')
        uncommon_ratio = uncommon_count / len(text)
        
        # 计算字符重复率（识别错误常导致重复）
        if len(text) >= 2:
            repeat_count = sum(1 for i in range(1, len(text)) if text[i] == text[i-1])
            repeat_ratio = repeat_count / (len(text) - 1)
        else:
            repeat_ratio = 0.0
        
        # 简单的 PPL 估计公式
        base_ppl = 50.0
        ppl = base_ppl * (1 + uncommon_ratio * 5) * (1 + repeat_ratio * 3)
        
        return min(ppl, 10000.0)


class UncertaintyRouter:
    """
    L2W1 不确定性路由器
    
    整合视觉熵和语义 PPL，进行路由决策
    """
    
    def __init__(self, config: RouterConfig = None, lm_model_path: str = None):
        """
        Args:
            config: 路由器配置
            lm_model_path: 语言模型路径（用于 PPL 计算）
        """
        self.config = config or RouterConfig()
        self.visual_calculator = VisualEntropyCalculator(self.config)
        self.semantic_calculator = SemanticPPLCalculator(lm_model_path)
    
    def calculate_visual_entropy(
        self, 
        logits: np.ndarray, 
        text: str
    ) -> Tuple[List[float], int, float]:
        """
        计算视觉不确定性
        
        Args:
            logits: 原始 logits，形状 [Seq_Len, Vocab_Size]
            text: 识别文本
            
        Returns:
            Tuple[char_entropies, suspicious_idx, max_entropy]
        """
        return self.visual_calculator.compute_char_entropy(logits, text)
    
    def calculate_ppl(self, text: str) -> float:
        """
        计算语义困惑度
        
        Args:
            text: 识别文本
            
        Returns:
            ppl: 困惑度值
        """
        return self.semantic_calculator.calculate(text)
    
    def should_reroute(
        self, 
        u_vis: float, 
        u_sem: float
    ) -> Tuple[bool, str]:
        """
        路由决策：判断是否需要调用 Agent B
        
        决策逻辑:
        - 如果 U_vis > τ_vis_high 或 U_sem > τ_sem_high: HIGH 风险
        - 如果 U_vis > τ_vis_low 或 U_sem > τ_sem_low: MEDIUM 风险
        - 否则: LOW 风险
        
        Args:
            u_vis: 视觉不确定性（最大字符熵）
            u_sem: 语义不确定性（PPL）
            
        Returns:
            Tuple[is_hard, risk_level]
        """
        # 高风险判定
        if u_vis > self.config.entropy_threshold_high or u_sem > self.config.ppl_threshold_high:
            return True, RiskLevel.HIGH.value
        
        # 中风险判定
        if u_vis > self.config.entropy_threshold_low or u_sem > self.config.ppl_threshold_low:
            return True, RiskLevel.MEDIUM.value
        
        # 低风险
        return False, RiskLevel.LOW.value
    
    def route(
        self, 
        logits: np.ndarray, 
        text: str,
        confidence: float = 1.0
    ) -> RoutingResult:
        """
        完整路由流程
        
        Args:
            logits: 原始 logits，形状 [Seq_Len, Vocab_Size]
            text: 识别文本
            confidence: Agent A 的置信度分数
            
        Returns:
            RoutingResult: 路由结果
        """
        # Step 1: 计算视觉不确定性
        char_entropies, suspicious_idx, max_entropy = self.calculate_visual_entropy(logits, text)
        visual_entropy_mean = np.mean(char_entropies) if char_entropies else 0.0
        
        # Step 2: 计算语义不确定性
        semantic_ppl = self.calculate_ppl(text)
        
        # Step 3: 路由决策
        is_hard, risk_level = self.should_reroute(max_entropy, semantic_ppl)
        
        # Step 4: 确定存疑字符
        suspicious_char = ""
        if suspicious_idx >= 0 and suspicious_idx < len(text):
            suspicious_char = text[suspicious_idx]
        
        # Step 5: 考虑置信度因素
        if confidence < 0.8 and risk_level == RiskLevel.LOW.value:
            # 置信度低但熵/PPL 低，可能是边界情况
            risk_level = RiskLevel.MEDIUM.value
            is_hard = True
        
        return RoutingResult(
            is_hard=is_hard,
            suspicious_index=suspicious_idx,
            suspicious_char=suspicious_char,
            risk_level=risk_level,
            visual_entropy=visual_entropy_mean,
            max_char_entropy=max_entropy,
            semantic_ppl=semantic_ppl,
            entropy_sequence=char_entropies
        )
    
    def route_batch(
        self,
        logits_list: List[np.ndarray],
        texts: List[str],
        confidences: List[float] = None
    ) -> List[RoutingResult]:
        """
        批量路由
        
        Args:
            logits_list: logits 列表
            texts: 文本列表
            confidences: 置信度列表
            
        Returns:
            List[RoutingResult]: 路由结果列表
        """
        if confidences is None:
            confidences = [1.0] * len(texts)
        
        results = []
        for logits, text, conf in zip(logits_list, texts, confidences):
            if logits is None:
                # 无 logits 的情况（非 CTC 算法）
                result = RoutingResult(
                    is_hard=False,
                    suspicious_index=-1,
                    suspicious_char="",
                    risk_level=RiskLevel.LOW.value,
                    visual_entropy=0.0,
                    max_char_entropy=0.0,
                    semantic_ppl=self.calculate_ppl(text),
                    entropy_sequence=[]
                )
            else:
                result = self.route(logits, text, conf)
            results.append(result)
        
        return results


# ==================== 便捷函数 ====================

def calculate_visual_entropy(logits: np.ndarray, text: str) -> Dict:
    """
    便捷函数：计算视觉熵并返回分析结果
    
    Args:
        logits: 原始 logits，形状 [Seq_Len, Vocab_Size]
        text: 识别文本
        
    Returns:
        dict: {
            'char_entropies': List[float],
            'suspicious_index': int,
            'suspicious_char': str,
            'max_entropy': float,
            'mean_entropy': float
        }
    """
    calculator = VisualEntropyCalculator()
    char_entropies, suspicious_idx, max_entropy = calculator.compute_char_entropy(logits, text)
    
    suspicious_char = ""
    if 0 <= suspicious_idx < len(text):
        suspicious_char = text[suspicious_idx]
    
    return {
        'char_entropies': char_entropies,
        'suspicious_index': suspicious_idx,
        'suspicious_char': suspicious_char,
        'max_entropy': max_entropy,
        'mean_entropy': np.mean(char_entropies) if char_entropies else 0.0
    }


def calculate_ppl(text: str, model_path: str = None) -> float:
    """
    便捷函数：计算文本困惑度
    
    Args:
        text: 输入文本
        model_path: 语言模型路径（可选）
        
    Returns:
        ppl: 困惑度值
    """
    calculator = SemanticPPLCalculator(model_path)
    return calculator.calculate(text)


def create_routing_manifest(
    logits: np.ndarray,
    text: str,
    confidence: float = 1.0,
    config: RouterConfig = None
) -> Dict:
    """
    创建路由 Manifest（符合 L2W1 Task JSON 规范）
    
    Args:
        logits: 原始 logits
        text: 识别文本
        confidence: 置信度
        config: 路由配置
        
    Returns:
        dict: Manifest Task JSON
    """
    router = UncertaintyRouter(config)
    result = router.route(logits, text, confidence)
    return result.to_dict()


# ==================== 测试代码 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("L2W1 Uncertainty Router 测试")
    print("=" * 60)
    
    # 模拟数据
    seq_len = 80
    vocab_size = 6625
    test_text = "中国科学院计算技术研究所"
    
    print(f"\n测试文本: '{test_text}' (长度: {len(test_text)})")
    print(f"Logits 形状: [{seq_len}, {vocab_size}]")
    
    # Case 1: 高置信度 logits（低熵）
    print("\n--- Case 1: 高置信度样本 ---")
    high_conf_logits = np.random.randn(seq_len, vocab_size) * 0.1
    # 模拟 CTC 输出：在特定位置有高置信度
    for i, char_idx in enumerate(range(0, seq_len, seq_len // len(test_text))):
        if i < len(test_text):
            high_conf_logits[char_idx, 100 + i] = 10.0
    
    result1 = calculate_visual_entropy(high_conf_logits, test_text)
    print(f"  字符熵序列: {[f'{e:.2f}' for e in result1['char_entropies']]}")
    print(f"  平均熵: {result1['mean_entropy']:.4f}")
    print(f"  最大熵: {result1['max_entropy']:.4f}")
    print(f"  存疑位置: {result1['suspicious_index']} ('{result1['suspicious_char']}')")
    
    # Case 2: 低置信度 logits（高熵）
    print("\n--- Case 2: 低置信度样本 ---")
    low_conf_logits = np.random.randn(seq_len, vocab_size) * 0.5
    # 在某个位置引入高不确定性
    uncertain_pos = 30
    low_conf_logits[uncertain_pos, :] = np.random.randn(vocab_size) * 0.1
    
    result2 = calculate_visual_entropy(low_conf_logits, test_text)
    print(f"  字符熵序列: {[f'{e:.2f}' for e in result2['char_entropies']]}")
    print(f"  平均熵: {result2['mean_entropy']:.4f}")
    print(f"  最大熵: {result2['max_entropy']:.4f}")
    print(f"  存疑位置: {result2['suspicious_index']} ('{result2['suspicious_char']}')")
    
    # Case 3: 完整路由测试
    print("\n--- Case 3: 完整路由决策 ---")
    router = UncertaintyRouter()
    routing_result = router.route(low_conf_logits, test_text, confidence=0.75)
    
    print(f"  是否困难样本: {routing_result.is_hard}")
    print(f"  风险等级: {routing_result.risk_level}")
    print(f"  存疑字符: 第 {routing_result.suspicious_index + 1} 个 '{routing_result.suspicious_char}'")
    print(f"  视觉熵均值: {routing_result.visual_entropy:.4f}")
    print(f"  语义 PPL: {routing_result.semantic_ppl:.4f}")
    
    print("\n--- Manifest JSON 输出 ---")
    manifest = routing_result.to_dict()
    import json
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
