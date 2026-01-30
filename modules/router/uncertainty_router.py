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

    LOW = "low"  # 低风险：Agent A 结果可信
    MEDIUM = "medium"  # 中风险：单一指标超阈值
    HIGH = "high"  # 高风险：多指标超阈值，需 Agent B 介入
    CRITICAL = "critical"  # 极高风险：严重不确定性


@dataclass
class RouterConfig:
    """路由器配置"""

    # 视觉熵阈值
    entropy_threshold_low: float = 2.0  # 低于此值视为高置信度
    entropy_threshold_high: float = 4.0  # 高于此值视为低置信度

    # 语义困惑度阈值
    ppl_threshold_low: float = 50.0  # 低于此值视为流畅文本
    ppl_threshold_high: float = 200.0  # 高于此值视为异常文本

    # CTC 解码参数
    blank_idx: int = 0  # CTC blank 符号索引

    # 熵计算参数
    epsilon: float = 1e-10  # 防止 log(0)

    # ========== SH-DA++ v4.0: 边界敏感检测配置已移除 ==========


@dataclass
class RoutingResult:
    """路由结果"""

    is_hard: bool  # 是否为困难样本
    suspicious_index: int  # 存疑字符位置 (0-indexed)
    suspicious_char: str  # 存疑字符
    risk_level: str  # 风险等级
    visual_entropy: float  # 视觉熵均值
    max_char_entropy: float  # 最大字符熵值
    semantic_ppl: float  # 语义困惑度
    entropy_sequence: List[float]  # 字符级熵序列

    # ========== SH-DA++ v4.0: 边界敏感检测结果字段已移除 ==========

    def to_dict(self) -> Dict:
        """转换为 Manifest Task JSON 格式"""
        result = {
            "is_hard": self.is_hard,
            "suspicious_index": self.suspicious_index,
            "suspicious_char": self.suspicious_char,
            "risk_level": self.risk_level,
            "metrics": {
                "visual_entropy": round(self.visual_entropy, 4),
                "max_char_entropy": round(self.max_char_entropy, 4),
                "semantic_ppl": round(self.semantic_ppl, 4),
            },
            "entropy_sequence": [round(e, 4) for e in self.entropy_sequence],
        }

        # SH-DA++ v4.0: 边界敏感检测结果已移除

        return result


class CTCAligner:
    """
    CTC 时间步对齐器 (加固版)

    将固定长度的 logits 序列 (e.g., 80) 映射到可变长度的字符序列 (e.g., 10)

    核心逻辑:
    1. 对每个时间步取 argmax 得到预测索引
    2. 应用 CTC 解码规则：移除连续重复 + 移除 blank
    3. 记录每个最终字符对应的时间步范围
    4. 将时间步的熵值聚合到对应字符

    加固策略 (v5.0.1):
    - 容错窗口: 允许 ±2 字符的长度误差
    - 贪婪映射: 长度不匹配时，基于熵权重的动态调整
    - 极端回退: 仅在误差 >30% 时使用均匀分配
    """

    # 容错配置
    TOLERANCE_WINDOW = 2  # 允许的字符数差异
    EXTREME_MISMATCH_RATIO = 0.3  # 极端不匹配比例阈值 (30%)

    def __init__(self, blank_idx: int = 0):
        """
        Args:
            blank_idx: CTC blank 符号的索引，PaddleOCR 默认为 0
        """
        self.blank_idx = blank_idx

    def align(
        self, logits: np.ndarray, text: str, timestep_entropy: np.ndarray = None
    ) -> List[Tuple[int, List[int]]]:
        """
        对齐 logits 时间步到字符位置 (加固版)

        Args:
            logits: 原始 logits，形状 [Seq_Len, Vocab_Size]
            text: 识别出的文本字符串
            timestep_entropy: 预计算的时间步熵 (可选，用于贪婪映射)

        Returns:
            List[Tuple[char_idx, List[timestep_indices]]]:
                每个字符对应的时间步索引列表
        """
        seq_len, vocab_size = logits.shape
        text_len = len(text)

        # 边界条件: 空文本
        if text_len == 0:
            return []

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

        decoded_len = len(char_to_timesteps)

        # Step 3: 验证对齐结果 (加固策略)
        if decoded_len == text_len:
            # 完美匹配
            return char_to_timesteps

        # 计算长度差异
        length_diff = abs(decoded_len - text_len)
        mismatch_ratio = length_diff / max(text_len, 1)

        # 策略 1: 容错窗口 (±2 字符)
        if length_diff <= self.TOLERANCE_WINDOW:
            return self._tolerant_align(
                char_to_timesteps, text_len, seq_len, logits, timestep_entropy
            )

        # 策略 2: 中等误差 - 贪婪映射
        if mismatch_ratio <= self.EXTREME_MISMATCH_RATIO:
            return self._greedy_align(
                char_to_timesteps, text_len, seq_len, logits, timestep_entropy
            )

        # 策略 3: 极端误差 (>30%) - 均匀回退
        return self._fallback_align(seq_len, text)

    def _tolerant_align(
        self,
        decoded_alignment: List[Tuple[int, List[int]]],
        target_len: int,
        seq_len: int,
        logits: np.ndarray = None,
        timestep_entropy: np.ndarray = None,
    ) -> List[Tuple[int, List[int]]]:
        """
        容错对齐: 处理 ±2 字符的长度误差

        策略:
        - 解码长度 > 目标: 截断末尾 (通常是重复字符)
        - 解码长度 < 目标: 填充末尾 (使用最后一个时间步附近)
        """
        decoded_len = len(decoded_alignment)

        if decoded_len == target_len:
            return decoded_alignment

        if decoded_len > target_len:
            # 截断: 保留前 target_len 个
            result = []
            for i in range(target_len):
                if i < len(decoded_alignment):
                    result.append((i, decoded_alignment[i][1]))
                else:
                    # 不应该发生，但做个保护
                    result.append((i, [seq_len - 1]))
            return result

        else:  # decoded_len < target_len
            # 填充: 复用解码结果，并为缺失的字符分配时间步
            result = []

            # 先复制已解码的
            for i, (_, timesteps) in enumerate(decoded_alignment):
                result.append((i, timesteps))

            # 为缺失的字符分配时间步
            # 策略: 从序列末尾均匀分配
            last_timestep = decoded_alignment[-1][1][-1] if decoded_alignment else 0
            remaining_steps = list(range(last_timestep + 1, seq_len))

            num_missing = target_len - decoded_len
            if remaining_steps:
                steps_per_missing = max(1, len(remaining_steps) // num_missing)

                for i in range(num_missing):
                    char_idx = decoded_len + i
                    start = i * steps_per_missing
                    end = min(start + steps_per_missing, len(remaining_steps))

                    if start < len(remaining_steps):
                        timesteps = [remaining_steps[j] for j in range(start, end)]
                    else:
                        timesteps = [seq_len - 1]

                    result.append((char_idx, timesteps))
            else:
                # 没有剩余时间步，复用最后一个
                for i in range(num_missing):
                    result.append((decoded_len + i, [seq_len - 1]))

            return result

    def _greedy_align(
        self,
        decoded_alignment: List[Tuple[int, List[int]]],
        target_len: int,
        seq_len: int,
        logits: np.ndarray = None,
        timestep_entropy: np.ndarray = None,
    ) -> List[Tuple[int, List[int]]]:
        """
        贪婪对齐: 基于熵权重的动态调整

        策略:
        - 计算每个时间步的熵值
        - 根据高熵区域动态划分字符边界
        - 确保峰值熵时间步落在合理的字符索引上
        """
        # 如果没有预计算熵，使用均匀分配
        if logits is None:
            return self._fallback_align(seq_len, target_len)

        # 计算时间步熵
        if timestep_entropy is None:
            # 计算 softmax
            logits_max = np.max(logits, axis=-1, keepdims=True)
            exp_logits = np.exp(logits - logits_max)
            probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

            # Shannon Entropy
            epsilon = 1e-10
            timestep_entropy = -np.sum(probs * np.log(probs + epsilon), axis=-1)

        # 找到熵的局部峰值 (这些是不确定的位置)
        peaks = []
        for t in range(1, seq_len - 1):
            if (
                timestep_entropy[t] > timestep_entropy[t - 1]
                and timestep_entropy[t] > timestep_entropy[t + 1]
            ):
                peaks.append((t, timestep_entropy[t]))

        # 按熵值降序排列
        peaks.sort(key=lambda x: x[1], reverse=True)

        # 使用均匀分配作为基础，但调整高熵区域
        result = []
        steps_per_char = seq_len / target_len

        for char_idx in range(target_len):
            start_t = int(char_idx * steps_per_char)
            end_t = int((char_idx + 1) * steps_per_char)

            # 确保至少有一个时间步
            if start_t >= seq_len:
                start_t = seq_len - 1
            if end_t > seq_len:
                end_t = seq_len
            if end_t <= start_t:
                end_t = start_t + 1

            timesteps = list(range(start_t, min(end_t, seq_len)))
            if not timesteps:
                timesteps = [min(start_t, seq_len - 1)]

            result.append((char_idx, timesteps))

        # 将最高熵峰值关联到最近的字符
        # (这确保 suspicious_index 指向真正不确定的位置)
        if peaks and result:
            peak_t, peak_entropy = peaks[0]  # 最高熵峰值

            # 找到包含这个时间步的字符
            for char_idx, timesteps in result:
                if peak_t in timesteps:
                    break  # 已经正确关联
            else:
                # 没有字符包含这个峰值，找最近的
                min_dist = float("inf")
                closest_char = 0
                for char_idx, timesteps in result:
                    for t in timesteps:
                        dist = abs(t - peak_t)
                        if dist < min_dist:
                            min_dist = dist
                            closest_char = char_idx

                # 将峰值时间步添加到最近的字符
                if closest_char < len(result):
                    result[closest_char] = (
                        closest_char,
                        sorted(set(result[closest_char][1] + [peak_t])),
                    )

        return result

    def _fallback_align(self, seq_len: int, text_or_len) -> List[Tuple[int, List[int]]]:
        """
        回退对齐策略：均匀分配时间步到字符

        仅在极端误差 (>30%) 时使用

        Args:
            seq_len: 序列长度
            text_or_len: 文本字符串或目标长度
        """
        if isinstance(text_or_len, str):
            text_len = len(text_or_len)
        else:
            text_len = text_or_len

        if text_len == 0:
            return []

        char_to_timesteps = []
        steps_per_char = seq_len / text_len

        for char_idx in range(text_len):
            start_t = int(char_idx * steps_per_char)
            end_t = int((char_idx + 1) * steps_per_char)
            timesteps = list(range(start_t, min(end_t, seq_len)))
            if not timesteps:
                timesteps = [min(start_t, seq_len - 1)]
            char_to_timesteps.append((char_idx, timesteps))

        return char_to_timesteps

    def get_alignment_stats(self, logits: np.ndarray, text: str) -> Dict:
        """
        获取对齐统计信息 (用于调试和分析)

        Returns:
            dict: 对齐统计信息
        """
        seq_len, vocab_size = logits.shape
        pred_indices = np.argmax(logits, axis=-1)

        # 解码
        decoded_chars = 0
        prev_idx = -1
        for idx in pred_indices:
            if idx != self.blank_idx and idx != prev_idx:
                decoded_chars += 1
            prev_idx = idx

        text_len = len(text)
        length_diff = abs(decoded_chars - text_len)
        mismatch_ratio = length_diff / max(text_len, 1)

        # 判断使用哪种策略
        if decoded_chars == text_len:
            strategy = "perfect_match"
        elif length_diff <= self.TOLERANCE_WINDOW:
            strategy = "tolerant_align"
        elif mismatch_ratio <= self.EXTREME_MISMATCH_RATIO:
            strategy = "greedy_align"
        else:
            strategy = "fallback_align"

        return {
            "seq_len": seq_len,
            "text_len": text_len,
            "decoded_chars": decoded_chars,
            "length_diff": length_diff,
            "mismatch_ratio": round(mismatch_ratio, 4),
            "strategy": strategy,
        }


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
        self, logits: np.ndarray, text: str
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
                self.model_path, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.use_gpu else torch.float32,
                device_map="auto" if self.use_gpu else "cpu",
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

    # _calculate_simple_ppl 方法已删除 - SH-DA++ v4.0 不再使用


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
        self, logits: np.ndarray, text: str
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

    def should_reroute(self, u_vis: float, u_sem: float) -> Tuple[bool, str]:
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
        if (
            u_vis > self.config.entropy_threshold_high
            or u_sem > self.config.ppl_threshold_high
        ):
            return True, RiskLevel.HIGH.value

        # 中风险判定
        if (
            u_vis > self.config.entropy_threshold_low
            or u_sem > self.config.ppl_threshold_low
        ):
            return True, RiskLevel.MEDIUM.value

        # 低风险
        return False, RiskLevel.LOW.value

    # check_boundary_sensitivity 方法已删除 - SH-DA++ v4.0 不再使用

    def route(
        self,
        logits: np.ndarray,
        text: str,
        confidence: float = 1.0,
    ) -> RoutingResult:
        """
        完整路由流程 (SH-DA++ v4.0)

        Args:
            logits: 原始 logits，形状 [Seq_Len, Vocab_Size]
            text: 识别文本
            confidence: Agent A 的置信度分数

        Returns:
            RoutingResult: 路由结果
        """
        # ========== 边界条件守护 (Boundary Case Guarding) ==========

        # Guard 1: 空文本 → 直接标记为 CRITICAL
        if len(text) == 0:
            return RoutingResult(
                is_hard=True,
                suspicious_index=-1,
                suspicious_char="",
                risk_level=RiskLevel.CRITICAL.value,
                visual_entropy=0.0,
                max_char_entropy=0.0,
                semantic_ppl=float("inf"),
                entropy_sequence=[],
            )

        # Guard 2: 单字符文本 → 简化处理，跳过复杂对齐
        if len(text) == 1:
            # 计算整体熵
            timestep_entropy = self.visual_calculator.compute_timestep_entropy(logits)
            max_entropy = float(np.max(timestep_entropy))

            is_hard = (
                max_entropy > self.config.entropy_threshold_low or confidence < 0.8
            )
            risk_level = RiskLevel.MEDIUM.value if is_hard else RiskLevel.LOW.value

            return RoutingResult(
                is_hard=is_hard,
                suspicious_index=0,  # 唯一的字符
                suspicious_char=text[0],
                risk_level=risk_level,
                visual_entropy=max_entropy,
                max_char_entropy=max_entropy,
                semantic_ppl=1.0,  # 单字符无法计算 PPL
                entropy_sequence=[max_entropy],
            )

        # SH-DA++ v4.0: 极端长宽比检测已移除

        # ========== 正常路由流程 ==========

        # Step 1: 计算视觉不确定性
        char_entropies, suspicious_idx, max_entropy = self.calculate_visual_entropy(
            logits, text
        )
        visual_entropy_mean = np.mean(char_entropies) if char_entropies else 0.0

        # Step 2: 计算语义不确定性
        semantic_ppl = self.calculate_ppl(text)

        # Step 3: 路由决策
        is_hard, risk_level = self.should_reroute(max_entropy, semantic_ppl)

        # Step 4: 确定存疑字符 (使用安全索引访问)
        suspicious_char = ""
        if 0 <= suspicious_idx < len(text):
            suspicious_char = text[suspicious_idx]
        elif suspicious_idx >= len(text):
            # 索引越界保护：钳制到最后一个字符
            suspicious_idx = len(text) - 1
            suspicious_char = text[suspicious_idx]

        # Step 5: 考虑置信度因素
        if confidence < 0.8 and risk_level == RiskLevel.LOW.value:
            # 置信度低但熵/PPL 低，可能是边界情况
            risk_level = RiskLevel.MEDIUM.value
            is_hard = True

        # ========== SH-DA++ v4.0: 边界敏感检测已移除 ==========

        return RoutingResult(
            is_hard=is_hard,
            suspicious_index=suspicious_idx,
            suspicious_char=suspicious_char,
            risk_level=risk_level,
            visual_entropy=visual_entropy_mean,
            max_char_entropy=max_entropy,
            semantic_ppl=semantic_ppl,
            entropy_sequence=char_entropies,
        )

    def route_batch(
        self,
        logits_list: List[np.ndarray],
        texts: List[str],
        confidences: List[float] = None,
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
                    entropy_sequence=[],
                )
            else:
                result = self.route(logits, text, conf)
            results.append(result)

        return results


# ==================== SH-DA++ v4.0: RuleOnlyScorer ====================


class RouteType(Enum):
    """分诊类型枚举 (SH-DA++ v4.0)"""
    
    NONE = "none"           # 无需路由：s_b < λ 且 s_a < λ
    BOUNDARY = "boundary"   # 边界风险：s_b ≥ λ 且 s_a < λ
    AMBIGUITY = "ambiguity" # 识别歧义：s_b < λ 且 s_a ≥ λ
    BOTH = "both"           # 双重风险：s_b ≥ λ 且 s_a ≥ λ


@dataclass
class RuleScorerConfig:
    """
    RuleOnlyScorer 配置类 (SH-DA++ v4.0)
    
    Attributes:
        v_min: v_edge Min-max 归一化下界
        v_max: v_edge Min-max 归一化上界
        lambda_threshold: 分诊阈值 λ
        eta: 综合优先级中 r_d 的权重系数 η
    """
    # v_edge 归一化参数
    v_min: float = 0.0      # 视觉熵下界
    v_max: float = 5.0      # 视觉熵上界（典型 CTC 熵范围）
    
    # 分诊阈值
    lambda_threshold: float = 0.5  # λ: 判定是否需要路由的阈值
    
    # 综合优先级权重
    eta: float = 0.5        # η: r_d 权重系数

    # 地质语义风险
    geology_dict_path: str = "data/dicts/Geology.txt"
    geology_min_len: int = 2
    geology_risk_weight: float = 1.0


@dataclass
class ScoringResult:
    """
    RuleOnlyScorer 评分结果 (SH-DA++ v4.0)
    
    Attributes:
        s_b: 边界风险评分 ∈ [0, 1]
        s_a: 识别歧义评分 ∈ [0, 1]
        q: 综合优先级 = max(s_b, s_a) + η·r_d
        route_type: 分诊类型
        details: 计算细节（用于调试和审计）
    """
    s_b: float              # 边界风险评分
    s_a: float              # 识别歧义评分
    q: float                # 综合优先级
    route_type: RouteType   # 分诊类型
    details: Dict           # 计算细节
    
    def to_dict(self) -> Dict:
        """转换为 JSON 可序列化格式"""
        return {
            "s_b": round(self.s_b, 6),
            "s_a": round(self.s_a, 6),
            "q": round(self.q, 6),
            "route_type": self.route_type.value,
            "details": self.details,
        }


class RuleOnlyScorer:
    """
    SH-DA++ v4.0 规则评分器
    
    基于 Stage 0 导出的 Emission 信号，计算边界风险和识别歧义评分。
    
    核心公式:
    1. 边界风险评分: s_b = clip(1/3·(v_edge·b_edge) + 1/3·b_edge + 1/3·drop, 0, 1)
    2. 识别歧义评分: s_a = clip(1 - min(m_i), 0, 1), where m_i = p_i^(1) - p_i^(2)
    3. 综合优先级: q = max(s_b, s_a) + η·r_d
    4. 分诊类型: 基于 s_b, s_a 与 λ 的比较
    
    References:
        - SH-DA++ v4.0 技术规范 Stage 1
    """
    
    def __init__(self, config: RuleScorerConfig = None):
        """
        初始化 RuleOnlyScorer
        
        Args:
            config: 评分器配置，默认使用 RuleScorerConfig()
        """
        self.config = config or RuleScorerConfig()
        self._geology = None
        try:
            from modules.router.domain_knowledge import GeologyKnowledge

            self._geology = GeologyKnowledge(
                dict_path=self.config.geology_dict_path,
                min_len=self.config.geology_min_len,
            )
        except Exception:
            # 地质词库缺失或加载失败时，默认关闭领域风险
            self._geology = None
    
    def _normalize_v_edge(self, v_edge: float) -> float:
        """
        对 v_edge 进行 Min-max 归一化
        
        公式: v_norm = clip((v_edge - v_min) / (v_max - v_min), 0, 1)
        
        Args:
            v_edge: 原始边界视觉熵值
            
        Returns:
            归一化后的值 ∈ [0, 1]
        """
        v_min, v_max = self.config.v_min, self.config.v_max
        
        # 防止除零
        if v_max <= v_min:
            return 0.5  # 配置无效时返回中间值
        
        v_norm = (v_edge - v_min) / (v_max - v_min)
        return float(np.clip(v_norm, 0.0, 1.0))
    
    def compute_boundary_score(
        self,
        boundary_stats: Dict,
        v_edge: float = None,
        char_count: int = 0,
        expected_char_count: int = 0,
    ) -> Tuple[float, Dict]:
        """
        计算边界风险评分 s_b
        
        公式: s_b = clip(1/3·(v_edge_norm·b_edge) + 1/3·b_edge + 1/3·drop, 0, 1)
        
        其中:
        - v_edge_norm: 归一化后的边界视觉熵
        - b_edge: 边界 blank 概率（取 L/R 中较大值）
        - drop: 字符丢失比例指标
        
        Args:
            boundary_stats: Stage 0 导出的边界统计量
                - blank_mean_L, blank_mean_R: 左右边界 blank 均值
                - blank_peak_L, blank_peak_R: 左右边界 blank 峰值
                - valid: 统计是否有效
            v_edge: 边界区域的视觉熵（可选，若无则从 boundary_stats 推断）
            char_count: 实际识别字符数
            expected_char_count: 预期字符数（基于图像宽度估计）
            
        Returns:
            Tuple[s_b, details]: 边界风险评分和计算细节
        """
        details = {
            "v_edge_raw": v_edge,
            "v_edge_norm": 0.0,
            "b_edge": 0.0,
            "drop": 0.0,
            "valid": False,
        }
        
        # 检查边界统计有效性
        if not boundary_stats or not boundary_stats.get("valid", False):
            # 统计无效时，返回中等风险评分
            details["reason"] = "boundary_stats_invalid"
            return 0.5, details
        
        details["valid"] = True
        
        # 1. 计算 b_edge：取左右边界 blank 概率的最大值
        blank_mean_L = boundary_stats.get("blank_mean_L", 0.0)
        blank_mean_R = boundary_stats.get("blank_mean_R", 0.0)
        blank_peak_L = boundary_stats.get("blank_peak_L", 0.0)
        blank_peak_R = boundary_stats.get("blank_peak_R", 0.0)
        
        # 使用均值和峰值的加权组合
        b_edge_L = 0.6 * blank_mean_L + 0.4 * blank_peak_L
        b_edge_R = 0.6 * blank_mean_R + 0.4 * blank_peak_R
        b_edge = max(b_edge_L, b_edge_R)
        details["b_edge"] = float(b_edge)
        details["b_edge_L"] = float(b_edge_L)
        details["b_edge_R"] = float(b_edge_R)
        
        # 2. 计算 v_edge_norm：归一化边界视觉熵
        if v_edge is not None:
            v_edge_norm = self._normalize_v_edge(v_edge)
        else:
            # 若未提供 v_edge，使用 blank 峰值作为代理指标
            v_edge_proxy = max(blank_peak_L, blank_peak_R) * 5.0  # 映射到 [0, 5] 范围
            v_edge_norm = self._normalize_v_edge(v_edge_proxy)
            details["v_edge_proxy"] = float(v_edge_proxy)
        
        details["v_edge_norm"] = float(v_edge_norm)
        
        # 3. 计算 drop：字符丢失比例
        if expected_char_count > 0 and char_count >= 0:
            drop = max(0.0, (expected_char_count - char_count) / expected_char_count)
        else:
            drop = 0.0  # 无法计算时假设无丢失
        
        details["drop"] = float(drop)
        details["char_count"] = char_count
        details["expected_char_count"] = expected_char_count
        
        # 4. 计算 s_b
        # 公式: s_b = clip(1/3·(v_edge_norm·b_edge) + 1/3·b_edge + 1/3·drop, 0, 1)
        term1 = (1.0 / 3.0) * (v_edge_norm * b_edge)
        term2 = (1.0 / 3.0) * b_edge
        term3 = (1.0 / 3.0) * drop
        
        s_b_raw = term1 + term2 + term3
        s_b = float(np.clip(s_b_raw, 0.0, 1.0))
        
        details["term1_v_b"] = float(term1)
        details["term2_b"] = float(term2)
        details["term3_drop"] = float(term3)
        details["s_b_raw"] = float(s_b_raw)
        
        return s_b, details
    
    def compute_ambiguity_score(
        self,
        top2_info: Dict,
    ) -> Tuple[float, Dict]:
        """
        计算识别歧义评分 s_a
        
        公式:
        - 若 top2_status == 'available': m_i = p_i^(1) - p_i^(2), s_a = clip(1 - min(m_i), 0, 1)
        - 若 Top-2 缺失: s_a = clip(1 - min(p_i^(1)), 0, 1)
        
        Args:
            top2_info: Stage 0 导出的 Top-2 信息
                - top2_status: 'available' | 'available_no_chars' | 'missing'
                - top1_conf_mean: Top-1 置信度均值
                - top2_conf_mean: Top-2 置信度均值
                - conf_gap_mean: 置信度差距均值
                - top1_probs: Top-1 概率序列 [T]
                - top2_probs: Top-2 概率序列 [T]
                
        Returns:
            Tuple[s_a, details]: 识别歧义评分和计算细节
        """
        details = {
            "top2_status": "missing",
            "method": "fallback",
            "min_margin": None,
            "min_conf": None,
        }
        
        top2_status = top2_info.get("top2_status", "missing")
        details["top2_status"] = top2_status
        
        if top2_status == "available":
            # 方法 1: 使用 Top-2 差距
            top1_probs = top2_info.get("top1_probs")
            top2_probs = top2_info.get("top2_probs")
            
            if top1_probs is not None and top2_probs is not None:
                top1_probs = np.array(top1_probs)
                top2_probs = np.array(top2_probs)
                
                # m_i = p_i^(1) - p_i^(2)
                margins = top1_probs - top2_probs
                min_margin = float(np.min(margins))
                
                # s_a = clip(1 - min(m_i), 0, 1)
                s_a = float(np.clip(1.0 - min_margin, 0.0, 1.0))
                
                details["method"] = "top2_margin"
                details["min_margin"] = min_margin
                details["mean_margin"] = float(np.mean(margins))
                details["margins_std"] = float(np.std(margins))
                
                return s_a, details
            
            # 若概率序列不可用，尝试使用聚合统计
            conf_gap_mean = top2_info.get("conf_gap_mean")
            if conf_gap_mean is not None:
                # 使用均值近似最小值（保守估计）
                approx_min_margin = max(0.0, conf_gap_mean - 0.1)  # 留 0.1 余量
                s_a = float(np.clip(1.0 - approx_min_margin, 0.0, 1.0))
                
                details["method"] = "top2_margin_approx"
                details["conf_gap_mean"] = conf_gap_mean
                details["approx_min_margin"] = approx_min_margin
                
                return s_a, details
        
        # 方法 2: Top-2 缺失，退化为 1 - min(p_i^(1))
        top1_probs = top2_info.get("top1_probs")
        top1_conf_mean = top2_info.get("top1_conf_mean")
        
        if top1_probs is not None:
            top1_probs = np.array(top1_probs)
            min_conf = float(np.min(top1_probs))
            s_a = float(np.clip(1.0 - min_conf, 0.0, 1.0))
            
            details["method"] = "top1_min"
            details["min_conf"] = min_conf
            details["mean_conf"] = float(np.mean(top1_probs))
            
            return s_a, details
        
        if top1_conf_mean is not None:
            # 使用均值近似
            approx_min_conf = max(0.0, top1_conf_mean - 0.15)  # 留 0.15 余量
            s_a = float(np.clip(1.0 - approx_min_conf, 0.0, 1.0))
            
            details["method"] = "top1_mean_approx"
            details["top1_conf_mean"] = top1_conf_mean
            details["approx_min_conf"] = approx_min_conf
            
            return s_a, details
        
        # 完全无信息时，返回高歧义评分（保守策略）
        details["method"] = "no_info_fallback"
        return 0.8, details
    
    def determine_route_type(
        self,
        s_b: float,
        s_a: float,
        lambda_threshold: float = None,
    ) -> RouteType:
        """
        根据评分判定分诊类型
        
        规则:
        - NONE: s_b < λ 且 s_a < λ
        - BOUNDARY: s_b ≥ λ 且 s_a < λ
        - AMBIGUITY: s_b < λ 且 s_a ≥ λ
        - BOTH: s_b ≥ λ 且 s_a ≥ λ
        
        Args:
            s_b: 边界风险评分
            s_a: 识别歧义评分
            lambda_threshold: 分诊阈值（默认使用配置值）
            
        Returns:
            RouteType: 分诊类型
        """
        lam = lambda_threshold if lambda_threshold is not None else self.config.lambda_threshold
        
        b_high = s_b >= lam
        a_high = s_a >= lam
        
        if b_high and a_high:
            return RouteType.BOTH
        elif b_high:
            return RouteType.BOUNDARY
        elif a_high:
            return RouteType.AMBIGUITY
        else:
            return RouteType.NONE
    
    def score(
        self,
        boundary_stats: Dict,
        top2_info: Dict,
        r_d: float = 0.0,
        v_edge: float = None,
        char_count: int = 0,
        expected_char_count: int = 0,
        agent_a_text: str = "",
    ) -> ScoringResult:
        """
        综合评分入口
        
        计算边界风险、识别歧义、综合优先级，并判定分诊类型。
        
        Args:
            boundary_stats: Stage 0 边界统计量
            top2_info: Stage 0 Top-2 信息
            r_d: 额外风险因子（如检测器置信度、几何异常等）
            v_edge: 边界区域视觉熵（可选）
            char_count: 实际识别字符数
            expected_char_count: 预期字符数
            
        Returns:
            ScoringResult: 评分结果
        """
        # 1. 计算边界风险评分 s_b
        s_b, s_b_details = self.compute_boundary_score(
            boundary_stats=boundary_stats,
            v_edge=v_edge,
            char_count=char_count,
            expected_char_count=expected_char_count,
        )
        
        # 2. 计算识别歧义评分 s_a
        s_a, s_a_details = self.compute_ambiguity_score(top2_info=top2_info)
        
        # 3. 计算领域风险 (s_d) 并合成 r_d
        s_d = 0.0
        geology_details = {}
        if self._geology is not None and agent_a_text:
            s_d, geology_details = self._geology.detect_geology_risk(agent_a_text)

        r_d_geology = self.config.geology_risk_weight * s_d
        r_d_total = float(r_d) + r_d_geology

        # 4. 计算综合优先级 q = max(s_b, s_a) + η·r_d_total
        eta = self.config.eta
        q = max(s_b, s_a) + eta * r_d_total
        
        # 5. 判定分诊类型
        route_type = self.determine_route_type(s_b, s_a)
        
        # 6. 汇总详情
        details = {
            "boundary_score_details": s_b_details,
            "ambiguity_score_details": s_a_details,
            "r_d": float(r_d_total),
            "r_d_base": float(r_d),
            "r_d_geology": float(r_d_geology),
            "s_d": float(s_d),
            "geology_details": geology_details,
            "eta": eta,
            "lambda": self.config.lambda_threshold,
            "config": {
                "v_min": self.config.v_min,
                "v_max": self.config.v_max,
            },
        }
        
        return ScoringResult(
            s_b=s_b,
            s_a=s_a,
            q=q,
            route_type=route_type,
            details=details,
        )


# ==================== SH-DA++ v4.0: OnlineBudgetController ====================


@dataclass
class BudgetControllerConfig:
    """
    OnlineBudgetController 配置类 (SH-DA++ v4.0)
    
    Attributes:
        window_size: 滑动窗口大小 W，用于计算实际调用率
        k: 比例系数，控制阈值更新步长
        lambda_min: 阈值下界
        lambda_max: 阈值上界
        lambda_init: 阈值初始值
        target_budget: 目标调用率 B ∈ [0, 1]
    """
    window_size: int = 200          # W: 滑动窗口大小
    k: float = 0.05                 # 比例系数
    lambda_min: float = 0.0         # λ_min
    lambda_max: float = 2.0         # λ_max
    lambda_init: float = 0.5        # λ 初始值
    target_budget: float = 0.2      # B: 目标调用率 (默认 20%)


class OnlineBudgetController:
    """
    SH-DA++ v4.0 在线预算控制器
    
    动态调整分诊阈值 λ，使实际 VLM 调用率逼近目标预算 B。
    
    核心公式:
        λ_{t+1} = clip(λ_t + k(B̄ - B), λ_min, λ_max)
    
    其中:
        - B̄: 过去 W 个样本的实际调用率
        - B: 目标调用率
        - k: 比例系数（步长因子）
    
    Warmup 机制:
        - 前 W 行不更新 λ，但记录调用情况
        - Warmup 期间使用初始阈值 λ_init
    
    References:
        - SH-DA++ v4.0 技术规范 Stage 1
    """
    
    def __init__(self, config: BudgetControllerConfig = None):
        """
        初始化 OnlineBudgetController
        
        Args:
            config: 预算控制器配置
        """
        self.config = config or BudgetControllerConfig()
        
        # 当前阈值
        self._lambda = self.config.lambda_init
        
        # 滑动窗口：记录过去 W 次决策结果 (True=升级, False=不升级)
        self._history: List[bool] = []
        
        # 样本计数器
        self._sample_count: int = 0
        
        # 统计信息
        self._total_upgrades: int = 0
        self._lambda_history: List[float] = [self._lambda]
    
    @property
    def current_lambda(self) -> float:
        """当前阈值 λ"""
        return self._lambda
    
    @property
    def is_warmup(self) -> bool:
        """是否处于 warmup 阶段"""
        return self._sample_count < self.config.window_size
    
    @property
    def actual_budget(self) -> float:
        """
        实际调用率 B̄
        
        Returns:
            过去 W 个样本的升级比例，若样本不足则返回当前累计比例
        """
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)
    
    @property
    def total_budget(self) -> float:
        """
        总体调用率（从开始到当前）
        """
        if self._sample_count == 0:
            return 0.0
        return self._total_upgrades / self._sample_count
    
    def decide(self, q: float, lambda_override: float = None) -> bool:
        """
        决策函数：判断是否需要升级（调用 VLM）
        
        Args:
            q: 综合优先级分数（来自 RuleOnlyScorer.score()）
            lambda_override: 可选的阈值覆盖值（用于测试）
            
        Returns:
            upgrade: True 表示需要升级（q ≥ λ），False 表示不升级
        """
        lam = lambda_override if lambda_override is not None else self._lambda
        return q >= lam
    
    def update(self, upgrade_decision: bool) -> Dict:
        """
        更新控制器状态
        
        在每次决策后调用，记录决策结果并更新阈值。
        
        Args:
            upgrade_decision: 本次决策结果（True=升级, False=不升级）
            
        Returns:
            Dict: 更新详情，包含 lambda_before, lambda_after, actual_budget 等
        """
        # 记录决策
        self._sample_count += 1
        if upgrade_decision:
            self._total_upgrades += 1
        
        # 维护滑动窗口
        self._history.append(upgrade_decision)
        if len(self._history) > self.config.window_size:
            self._history.pop(0)  # 移除最旧的记录
        
        # 更新详情
        details = {
            "sample_count": self._sample_count,
            "lambda_before": self._lambda,
            "actual_budget": self.actual_budget,
            "target_budget": self.config.target_budget,
            "is_warmup": self.is_warmup,
            "updated": False,
        }
        
        # Warmup 期间不更新 λ
        if self.is_warmup:
            details["lambda_after"] = self._lambda
            details["reason"] = "warmup"
            return details
        
        # 计算阈值更新
        # λ_{t+1} = clip(λ_t + k(B̄ - B), λ_min, λ_max)
        B_bar = self.actual_budget      # 实际调用率
        B = self.config.target_budget   # 目标调用率
        k = self.config.k               # 比例系数
        
        delta = k * (B_bar - B)
        lambda_new = self._lambda + delta
        lambda_new = float(np.clip(lambda_new, self.config.lambda_min, self.config.lambda_max))
        
        details["delta"] = delta
        details["lambda_after"] = lambda_new
        details["updated"] = True
        
        # 更新阈值
        self._lambda = lambda_new
        self._lambda_history.append(self._lambda)
        
        return details
    
    def step(self, q: float) -> Tuple[bool, Dict]:
        """
        单步执行：决策 + 更新
        
        便捷方法，等价于：
            upgrade = decide(q)
            details = update(upgrade)
        
        Args:
            q: 综合优先级分数
            
        Returns:
            Tuple[upgrade, details]: 决策结果和更新详情
        """
        upgrade = self.decide(q)
        details = self.update(upgrade)
        details["q"] = q
        details["upgrade"] = upgrade
        return upgrade, details
    
    def reset(self):
        """
        重置控制器状态
        
        用于开始新的评估 epoch 或测试。
        """
        self._lambda = self.config.lambda_init
        self._history.clear()
        self._sample_count = 0
        self._total_upgrades = 0
        self._lambda_history = [self._lambda]
    
    def get_stats(self) -> Dict:
        """
        获取统计信息
        
        Returns:
            Dict: 包含 lambda_current, lambda_history, actual_budget, total_budget 等
        """
        return {
            "lambda_current": self._lambda,
            "lambda_init": self.config.lambda_init,
            "lambda_min": self.config.lambda_min,
            "lambda_max": self.config.lambda_max,
            "lambda_history_len": len(self._lambda_history),
            "lambda_history_last_10": self._lambda_history[-10:],
            "sample_count": self._sample_count,
            "total_upgrades": self._total_upgrades,
            "actual_budget_window": self.actual_budget,
            "total_budget": self.total_budget,
            "target_budget": self.config.target_budget,
            "is_warmup": self.is_warmup,
            "window_size": self.config.window_size,
            "k": self.config.k,
        }


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
    char_entropies, suspicious_idx, max_entropy = calculator.compute_char_entropy(
        logits, text
    )

    suspicious_char = ""
    if 0 <= suspicious_idx < len(text):
        suspicious_char = text[suspicious_idx]

    return {
        "char_entropies": char_entropies,
        "suspicious_index": suspicious_idx,
        "suspicious_char": suspicious_char,
        "max_entropy": max_entropy,
        "mean_entropy": np.mean(char_entropies) if char_entropies else 0.0,
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
    logits: np.ndarray, text: str, confidence: float = 1.0, config: RouterConfig = None
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
    print(
        f"  存疑字符: 第 {routing_result.suspicious_index + 1} 个 '{routing_result.suspicious_char}'"
    )
    print(f"  视觉熵均值: {routing_result.visual_entropy:.4f}")
    print(f"  语义 PPL: {routing_result.semantic_ppl:.4f}")

    print("\n--- Manifest JSON 输出 ---")
    manifest = routing_result.to_dict()
    import json

    print(json.dumps(manifest, ensure_ascii=False, indent=2))

    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
