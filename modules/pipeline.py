"""
L2W1 完整推理流水线 (SH-DA++ v4.0)

将 Agent A、Router、Agent B 三个组件串联，实现端到端的手写识别与纠错

流程:
1. Agent A (PP-OCRv5): 全量行级扫描，输出文本 + Logits + Emission
2. Router (RuleOnlyScorer + OnlineBudgetController): 计算 s_b/s_a 评分，动态预算控制
3. Agent B (Qwen2.5-VL): 对困难样本进行精准视觉重写（异步调用，支持超时回退）

架构:
    Input Image
         │
         ▼
    ┌─────────────┐
    │   Agent A   │ ──────► text, logits, boundary_stats, top2_info
    └─────────────┘
         │
         ▼
    ┌─────────────────────────────────┐
    │   RuleOnlyScorer               │ ──────► s_b, s_a, q, route_type
    │   OnlineBudgetController       │ ──────► upgrade, λ_t
    └─────────────────────────────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
  Pass     Upgrade
    │         │
    │    ┌────┴─────────────────┐
    │    │ Agent B (Async)      │
    │    │ Timeout: 500ms       │ ──────► corrected_text
    │    │ Fallback: T_A        │
    │    └──────────────────────┘
    │         │
    └────┬────┘
         ▼
    Final Output → router_features.jsonl
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import time as time_module
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import numpy as np

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class PipelineConfig:
    """流水线配置 (SH-DA++ v4.0)"""
    # ========== Agent A (PP-OCRv5 TextRecognizer) 配置 ==========
    agent_a_model_dir: str = ""             # 模型目录路径
    agent_a_batch_size: int = 6             # 批处理大小
    rec_image_shape: str = "3, 48, 320"     # 输入图像形状
    rec_char_dict_path: str = "./ppocr/utils/ppocrv5_dict.txt"  # 字典路径
    rec_algorithm: str = "SVTR_LCNet"       # 识别算法
    use_space_char: bool = True             # 是否使用空格字符
    use_gpu: bool = True                    # 是否使用 GPU
    gpu_mem: int = 500                      # GPU 显存限制 (MB)
    
    # ========== SH-DA++ v4.0: RuleOnlyScorer 配置 ==========
    v_min: float = 0.0              # v_edge 归一化下界
    v_max: float = 5.0              # v_edge 归一化上界
    lambda_threshold: float = 0.5   # 初始分诊阈值 λ
    eta: float = 0.5                # 综合优先级中 r_d 权重 η
    
    # ========== SH-DA++ v4.0: OnlineBudgetController 配置 ==========
    budget_window_size: int = 200       # W: 滑动窗口大小
    budget_k: float = 0.05              # 比例系数 k
    budget_lambda_min: float = 0.0      # λ_min
    budget_lambda_max: float = 2.0      # λ_max
    budget_target: float = 0.2          # B: 目标调用率 (20%)
    
    # ========== 异步与超时配置 ==========
    agent_b_timeout_ms: int = 500       # Agent B 超时阈值 (ms)
    agent_b_max_workers: int = 2        # ThreadPool 最大工作线程数
    agent_b_queue_size: int = 10        # 队列最大长度
    
    # Agent B 配置
    agent_b_model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    agent_b_use_4bit: bool = True
    agent_b_use_flash_attention: bool = True
    skip_agent_b: bool = False      # 是否完全跳过 Agent B（Stage 1 数据采集模式）
    
    # 流水线配置
    verbose: bool = False           # 是否打印详细日志
    
    # ========== 日志配置 ==========
    router_features_log: str = "results/router_features.jsonl"  # 路由特征日志路径


@dataclass
class PipelineResult:
    """
    流水线结果
    
    符合 L2W1 Master Data Protocol v2.0 (data_protocol_v2.json)
    
    输出结构:
    {
        "id": str,
        "image": str,
        "gt_text": str,
        "agent_a": { "text", "confidence", "suspicious_index", "suspicious_char", "raw_logits_shape" },
        "router": { "is_hard", "visual_entropy", "semantic_ppl", "risk_level", "decision" },
        "agent_b": { "text", "is_corrected", "refinement_strategy" },
        "metadata": { "source", "split", "difficulty", "error_type", "gt_char_len", "processing_time_ms", "environment" }
    }
    """
    # ========== 顶层字段 ==========
    id: str = ""                     # 样本唯一 ID
    image_path: str = ""             # 图像路径
    gt_text: str = ""                # 真值文本 (Ground Truth)
    
    # ========== Agent A 输出 ==========
    agent_a_text: str = ""
    agent_a_confidence: float = 0.0
    agent_a_logits: Optional[np.ndarray] = None
    agent_a_logits_shape: Tuple[int, int] = (0, 0)  # raw_logits_shape
    
    # ========== Router 输出 ==========
    is_hard: bool = False
    suspicious_index: int = -1       # 0-indexed (系统内部统一使用 0-indexed)
    suspicious_char: str = ""
    risk_level: str = "low"
    visual_entropy: float = 0.0
    semantic_ppl: float = 0.0
    router_decision: str = "pass"    # "pass" | "route_to_agent_b"
    
    # ========== SH-DA++ v4.0: RuleOnlyScorer 输出 ==========
    s_b: float = 0.0                 # 边界风险评分
    s_a: float = 0.0                 # 识别歧义评分
    q: float = 0.0                   # 综合优先级
    route_type: str = "none"         # 分诊类型: none/boundary/ambiguity/both
    
    # ========== SH-DA++ v4.0: OnlineBudgetController 输出 ==========
    lambda_t: float = 0.5            # 当前阈值 λ_t
    upgrade: bool = False            # 是否升级 (q ≥ λ)
    
    # ========== SH-DA++ v4.0: 超时与回退 ==========
    b_timeout: bool = False          # Agent B 是否超时
    b_fallback: bool = False         # 是否使用 fallback (T_final = T_A)
    
    # ========== Agent B 输出 ==========
    agent_b_text: str = ""
    agent_b_is_corrected: bool = False
    agent_b_refinement_strategy: str = "explicit_indexing_prompt"  # EIP 策略
    
    # ========== 最终输出 ==========
    final_text: str = ""
    
    # ========== 元信息 ==========
    routed_to_agent_b: bool = False
    processing_time_ms: int = 0      # 毫秒
    source: str = ""                 # 数据来源 (viscgec, scut, casia 等)
    split: str = ""                  # 数据集划分 (train, val, test) [v2.0 新增]
    difficulty: str = "normal"       # "easy" | "normal" | "hard"
    error_type: str = ""             # 错误类型 (grammar_omission, similar_char 等)
    environment: str = ""            # 运行环境描述
    
    def to_dict(self) -> Dict:
        """
        转换为符合 Data Protocol v2.0 的嵌套结构
        
        Returns:
            符合 data_protocol_v2.json 规范的字典
        """
        # 计算 logits shape
        logits_shape = list(self.agent_a_logits_shape)
        if self.agent_a_logits is not None and len(self.agent_a_logits.shape) >= 2:
            logits_shape = list(self.agent_a_logits.shape[-2:])
        
        # Router 决策
        decision = "route_to_agent_b" if self.is_hard else "pass"
        
        # 难度评估
        difficulty = self.difficulty
        if not difficulty:
            if self.visual_entropy > 4.0 or self.semantic_ppl > 150:
                difficulty = "hard"
            elif self.visual_entropy > 2.5 or self.semantic_ppl > 80:
                difficulty = "normal"
            else:
                difficulty = "easy"
        
        return {
            # 顶层字段
            "id": self.id,
            "image": self.image_path,
            "gt_text": self.gt_text,
            
            # Agent A 嵌套结构
            "agent_a": {
                "text": self.agent_a_text,
                "confidence": round(self.agent_a_confidence, 4),
                "suspicious_index": self.suspicious_index,  # 0-indexed
                "suspicious_char": self.suspicious_char,
                "raw_logits_shape": logits_shape,
            },
            
            # Router 嵌套结构
            "router": {
                "is_hard": self.is_hard,
                "visual_entropy": round(self.visual_entropy, 4),
                "semantic_ppl": round(self.semantic_ppl, 2),
                "risk_level": self.risk_level,
                "decision": decision,
                # SH-DA++ v4.0 字段
                "s_b": round(self.s_b, 6),
                "s_a": round(self.s_a, 6),
                "q": round(self.q, 6),
                "route_type": self.route_type,
                "lambda_t": round(self.lambda_t, 6),
                "upgrade": self.upgrade,
                "b_timeout": self.b_timeout,
                "b_fallback": self.b_fallback,
            },
            
            # Agent B 嵌套结构
            "agent_b": {
                "text": self.agent_b_text,
                "is_corrected": self.agent_b_is_corrected,
                "refinement_strategy": self.agent_b_refinement_strategy,
            },
            
            # Metadata 嵌套结构 (v2.0)
            "metadata": {
                "source": self.source,
                "split": self.split,  # [v2.0 新增]
                "difficulty": difficulty,
                "error_type": self.error_type,
                "gt_char_len": len(self.gt_text),
                "processing_time_ms": self.processing_time_ms,
                "environment": self.environment,
            },
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PipelineResult':
        """
        从 Data Protocol v2.0 格式的字典构建 PipelineResult
        
        Args:
            data: 符合 data_protocol_v2.json 规范的字典
            
        Returns:
            PipelineResult 实例
        """
        agent_a = data.get('agent_a', {})
        router = data.get('router', {})
        agent_b = data.get('agent_b', {})
        metadata = data.get('metadata', {})
        
        return cls(
            # 顶层字段
            id=data.get('id', ''),
            image_path=data.get('image', ''),
            gt_text=data.get('gt_text', ''),
            
            # Agent A
            agent_a_text=agent_a.get('text', ''),
            agent_a_confidence=agent_a.get('confidence', 0.0),
            agent_a_logits_shape=tuple(agent_a.get('raw_logits_shape', [0, 0])),
            suspicious_index=agent_a.get('suspicious_index', -1),
            suspicious_char=agent_a.get('suspicious_char', ''),
            
            # Router
            is_hard=router.get('is_hard', False),
            visual_entropy=router.get('visual_entropy', 0.0),
            semantic_ppl=router.get('semantic_ppl', 0.0),
            risk_level=router.get('risk_level', 'low'),
            router_decision=router.get('decision', 'pass'),
            # SH-DA++ v4.0
            s_b=router.get('s_b', 0.0),
            s_a=router.get('s_a', 0.0),
            q=router.get('q', 0.0),
            route_type=router.get('route_type', 'none'),
            lambda_t=router.get('lambda_t', 0.5),
            upgrade=router.get('upgrade', False),
            b_timeout=router.get('b_timeout', False),
            b_fallback=router.get('b_fallback', False),
            
            # Agent B
            agent_b_text=agent_b.get('text', ''),
            agent_b_is_corrected=agent_b.get('is_corrected', False),
            agent_b_refinement_strategy=agent_b.get('refinement_strategy', 'explicit_indexing_prompt'),
            
            # Metadata (v2.0)
            source=metadata.get('source', ''),
            split=metadata.get('split', ''),  # [v2.0 新增]
            difficulty=metadata.get('difficulty', 'normal'),
            error_type=metadata.get('error_type', ''),
            processing_time_ms=metadata.get('processing_time_ms', 0),
            environment=metadata.get('environment', ''),
            
            # 计算字段
            routed_to_agent_b=router.get('decision') == 'route_to_agent_b',
            final_text=agent_b.get('text', '') if router.get('is_hard') else agent_a.get('text', ''),
        )


class L2W1Pipeline:
    """
    L2W1 完整推理流水线 (SH-DA++ v4.0)
    
    实现分层多智能体架构：
    - Agent A (System 1): 快速扫描 + Emission 导出
    - RuleOnlyScorer: 计算 s_b, s_a, q 评分
    - OnlineBudgetController: 动态阈值 λ 控制
    - Agent B (System 2): 异步精细重写（支持超时回退）
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
        
        # SH-DA++ v4.0: 新增组件
        self._rule_scorer = None
        self._budget_controller = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._router_features_file = None
        
        # 统计
        self.stats = {
            "total_processed": 0,
            "routed_to_agent_b": 0,
            "corrected_by_agent_b": 0,
            # SH-DA++ v4.0 新增统计
            "b_timeout_count": 0,
            "b_fallback_count": 0,
            "upgrade_count": 0,
        }
    
    @property
    def rule_scorer(self):
        """获取 RuleOnlyScorer"""
        if self._rule_scorer is None:
            self._init_rule_scorer()
        return self._rule_scorer
    
    @property
    def budget_controller(self):
        """获取 OnlineBudgetController"""
        if self._budget_controller is None:
            self._init_budget_controller()
        return self._budget_controller
    
    @property
    def executor(self):
        """获取 ThreadPoolExecutor"""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.config.agent_b_max_workers
            )
        return self._executor
    
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
        """初始化 Agent A (PP-OCRv5 TextRecognizerWithLogits) - 严格模式"""
        # 检查模型目录是否配置
        if not self.config.agent_a_model_dir:
            raise ValueError(
                "[FATAL] agent_a_model_dir 未配置\n"
                "请在 PipelineConfig 中指定 agent_a_model_dir 参数"
            )
        
        # 检查模型目录是否存在
        from pathlib import Path
        model_path = Path(self.config.agent_a_model_dir)
        if not model_path.exists():
            raise FileNotFoundError(
                f"[FATAL] 模型目录不存在: {self.config.agent_a_model_dir}\n"
                "请下载 PP-OCRv5 识别模型:\n"
                "  wget https://paddle-model-ecology.bj.bcebos.com/model/ocr/PP-OCRv5/ch_PP-OCRv5_rec_infer.tar\n"
                "  tar -xf ch_PP-OCRv5_rec_infer.tar\n"
                "  mv ch_PP-OCRv5_rec_infer ./models/ppocrv5_rec"
            )
        
        # 检查模型文件是否完整
        required_files = ["inference.pdiparams", "inference.pdmodel"]
        missing_files = [f for f in required_files if not (model_path / f).exists()]
        if missing_files:
            raise FileNotFoundError(
                f"[FATAL] 模型文件缺失: {self.config.agent_a_model_dir}\n"
                f"缺失文件: {', '.join(missing_files)}"
            )
        
        try:
            from modules.paddle_engine.predict_rec_modified import TextRecognizerWithLogits
            import tools.infer.utility as utility
            
            # 构建 args 对象
            parser = utility.init_args()
            args = parser.parse_args([])  # 使用默认参数
            
            # 覆盖配置参数
            args.rec_model_dir = self.config.agent_a_model_dir
            args.rec_batch_num = self.config.agent_a_batch_size
            args.rec_image_shape = self.config.rec_image_shape
            args.rec_char_dict_path = self.config.rec_char_dict_path
            args.rec_algorithm = self.config.rec_algorithm
            args.use_space_char = self.config.use_space_char
            args.use_gpu = self.config.use_gpu
            args.gpu_mem = self.config.gpu_mem
            args.warmup = False
            args.benchmark = False
            args.use_onnx = False
            args.return_word_box = False
            
            # 使用适配器包装 TextRecognizerWithLogits
            self._agent_a = AgentAAdapter(args, verbose=self.config.verbose)
            
            if self.config.verbose:
                print(f"[Pipeline] Agent A 已初始化: {self.config.agent_a_model_dir}")
            
        except ImportError as e:
            raise ImportError(
                f"[FATAL] 无法导入 Agent A 模块: {e}\n"
                "请确保 modules.paddle_engine.predict_rec_modified 模块可用"
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(
                f"[FATAL] Agent A 初始化失败: {e}\n"
                "请检查:\n"
                "  1. 模型文件是否完整\n"
                "  2. PaddlePaddle 是否正确安装\n"
                "  3. GPU/CUDA 是否可用"
            )
    
    def _init_router(self):
        """初始化 Router (旧版，保留兼容)"""
        from modules.router import UncertaintyRouter, RouterConfig
        
        router_config = RouterConfig()
        self._router = UncertaintyRouter(config=router_config)
    
    def _init_rule_scorer(self):
        """初始化 RuleOnlyScorer (SH-DA++ v4.0)"""
        from modules.router.uncertainty_router import RuleOnlyScorer, RuleScorerConfig
        
        scorer_config = RuleScorerConfig(
            v_min=self.config.v_min,
            v_max=self.config.v_max,
            lambda_threshold=self.config.lambda_threshold,
            eta=self.config.eta,
        )
        self._rule_scorer = RuleOnlyScorer(config=scorer_config)
    
    def _init_budget_controller(self):
        """初始化 OnlineBudgetController (SH-DA++ v4.0)"""
        from modules.router.uncertainty_router import OnlineBudgetController, BudgetControllerConfig
        
        budget_config = BudgetControllerConfig(
            window_size=self.config.budget_window_size,
            k=self.config.budget_k,
            lambda_min=self.config.budget_lambda_min,
            lambda_max=self.config.budget_lambda_max,
            lambda_init=self.config.lambda_threshold,
            target_budget=self.config.budget_target,
        )
        self._budget_controller = OnlineBudgetController(config=budget_config)
    
    def _init_agent_b(self):
        """初始化 Agent B - 严格模式"""
        try:
            from modules.vlm_expert import AgentBExpert, AgentBConfig
            
            agent_b_config = AgentBConfig(
                model_path=self.config.agent_b_model_path,
                use_4bit=self.config.agent_b_use_4bit,
                use_flash_attention=self.config.agent_b_use_flash_attention,
            )
            self._agent_b = AgentBExpert(config=agent_b_config)
            
            if self.config.verbose:
                print(f"[Pipeline] Agent B 已初始化: {self.config.agent_b_model_path}")
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise RuntimeError(
                f"[FATAL] Agent B 初始化失败: {e}\n"
                "请检查:\n"
                "  1. Qwen2.5-VL 模型是否已下载\n"
                "  2. transformers 版本是否兼容\n"
                "  3. GPU 显存是否足够"
            )
    
    def process(
        self,
        image: Union[str, np.ndarray],
        image_path: str = "",
        sample_id: str = "",
        gt_text: str = "",
        source: str = "",
        error_type: str = "",
        boundary_stats: Dict = None,
        top2_info: Dict = None,
        r_d: float = 0.0,
        expected_char_count: int = 0,
    ) -> PipelineResult:
        """
        处理单张图像 (SH-DA++ v4.0)
        
        Args:
            image: 图像路径或 numpy 数组
            image_path: 图像路径（用于记录）
            sample_id: 样本唯一 ID
            gt_text: 真值文本 (Ground Truth)
            source: 数据来源 (viscgec, scut 等)
            error_type: 错误类型分类
            boundary_stats: Stage 0 边界统计量 (来自 predict_rec_modified)
            top2_info: Stage 0 Top-2 信息 (来自 predict_rec_modified)
            r_d: 额外风险因子
            expected_char_count: 预期字符数 (用于计算 drop)
            
        Returns:
            PipelineResult: 处理结果 (符合 Data Protocol v2.0)
        """
        start_time = time_module.time()
        
        # 初始化结果
        result = PipelineResult(
            id=sample_id or f"sample_{self.stats['total_processed']:06d}",
            image_path=image_path or str(image),
            gt_text=gt_text,
            source=source,
            error_type=error_type,
            environment=self._get_environment_info(),
        )
        
        # Step 1: Agent A 推理
        if self.config.verbose:
            print("[Pipeline] Step 1: Agent A 推理...")
        
        agent_a_output = self.agent_a.recognize(image)
        result.agent_a_text = agent_a_output.get('text', '')
        result.agent_a_confidence = agent_a_output.get('confidence', 0.0)
        result.agent_a_logits = agent_a_output.get('logits', None)
        
        # 获取 Stage 0 信号 (如果未提供则尝试从 agent_a_output 获取)
        if boundary_stats is None:
            boundary_stats = agent_a_output.get('boundary_stats', {})
        if top2_info is None:
            top2_info = agent_a_output.get('top2_info', {})
        
        # 获取路由器耗时
        lat_router_ms = agent_a_output.get('lat_router_ms', 0.0)
        
        # 记录 logits 形状
        if result.agent_a_logits is not None:
            result.agent_a_logits_shape = result.agent_a_logits.shape[-2:] if len(result.agent_a_logits.shape) >= 2 else (0, 0)
        
        # Step 2: SH-DA++ v4.0 RuleOnlyScorer 评分
        if self.config.verbose:
            print("[Pipeline] Step 2: RuleOnlyScorer 评分...")
        
        char_count = len(result.agent_a_text)
        if expected_char_count <= 0:
            # 默认预期字符数（可根据图像宽度估计）
            expected_char_count = char_count + 2  # 假设可能丢失 2 个字符
        
        scoring_result = self.rule_scorer.score(
            boundary_stats=boundary_stats or {},
            top2_info=top2_info or {},
            r_d=r_d,
            char_count=char_count,
            expected_char_count=expected_char_count,
        )
        
        result.s_b = scoring_result.s_b
        result.s_a = scoring_result.s_a
        result.q = scoring_result.q
        result.route_type = scoring_result.route_type.value
        
        # Step 3: OnlineBudgetController 决策
        if self.config.verbose:
            print("[Pipeline] Step 3: OnlineBudgetController 决策...")
        
        result.lambda_t = self.budget_controller.current_lambda
        upgrade, budget_details = self.budget_controller.step(result.q)
        result.upgrade = upgrade
        
        # 兼容旧版字段
        result.is_hard = upgrade
        result.router_decision = "route_to_agent_b" if upgrade else "pass"
        
        # Step 4: 条件调用 Agent B (异步 + 超时回退)
        if upgrade:
            result.routed_to_agent_b = True
            self.stats["routed_to_agent_b"] += 1
            self.stats["upgrade_count"] += 1
            
            # 检查是否跳过 Agent B（Stage 1 数据采集模式）
            if self.config.skip_agent_b:
                if self.config.verbose:
                    print("[Pipeline] Step 4: 跳过 Agent B (skip_agent_b=True)")
                
                # Stage 1 模式：只记录路由决策，不实际调用 Agent B
                result.agent_b_text = ""
                result.b_timeout = False
                result.b_fallback = True  # 标记为回退
                result.final_text = result.agent_a_text
            else:
                if self.config.verbose:
                    print("[Pipeline] Step 4: 异步调用 Agent B...")
                
                # 构建 manifest
                manifest = {
                    'ocr_text': result.agent_a_text,
                    'suspicious_index': result.suspicious_index,
                    'suspicious_char': result.suspicious_char,
                    'risk_level': result.risk_level,
                    'route_type': result.route_type,
                    's_b': result.s_b,
                    's_a': result.s_a,
                }
                
                # 异步调用 Agent B，支持超时回退
                agent_b_text, b_timeout, b_fallback = self._call_agent_b_async(
                    image, manifest, result.agent_a_text
                )
                
                result.agent_b_text = agent_b_text
                result.b_timeout = b_timeout
                result.b_fallback = b_fallback
                
                if b_timeout:
                    self.stats["b_timeout_count"] += 1
                if b_fallback:
                    self.stats["b_fallback_count"] += 1
                
                # 判断是否有修正
                result.agent_b_is_corrected = (
                    not b_fallback and agent_b_text != result.agent_a_text
                )
                
                if result.agent_b_is_corrected:
                    self.stats["corrected_by_agent_b"] += 1
                
                # 最终文本：fallback 时使用 Agent A，否则使用 Agent B
                result.final_text = result.agent_a_text if b_fallback else result.agent_b_text
        else:
            if self.config.verbose:
                print("[Pipeline] Step 4: 跳过 Agent B (upgrade=False)")
            
            result.final_text = result.agent_a_text
        
        # 计算处理时间 (毫秒)
        result.processing_time_ms = int((time_module.time() - start_time) * 1000)
        self.stats["total_processed"] += 1
        
        # Step 5: 记录路由特征日志
        self._log_router_features(
            result, 
            scoring_result, 
            budget_details,
            boundary_stats=boundary_stats,
            top2_info=top2_info,
            lat_router_ms=lat_router_ms,
        )
        
        return result
    
    def _call_agent_b_async(
        self,
        image: Union[str, np.ndarray],
        manifest: Dict,
        fallback_text: str,
    ) -> Tuple[str, bool, bool]:
        """
        异步调用 Agent B，支持超时回退
        
        Args:
            image: 图像
            manifest: 任务描述
            fallback_text: 超时或失败时的回退文本 (T_A)
            
        Returns:
            Tuple[corrected_text, b_timeout, b_fallback]:
                - corrected_text: Agent B 输出或回退文本
                - b_timeout: 是否超时
                - b_fallback: 是否使用回退
        """
        timeout_sec = self.config.agent_b_timeout_ms / 1000.0
        
        try:
            # 提交异步任务
            future = self.executor.submit(
                self.agent_b.process_hard_sample, image, manifest
            )
            
            # 等待结果，支持超时
            agent_b_output = future.result(timeout=timeout_sec)
            corrected_text = agent_b_output.get('corrected_text', fallback_text)
            
            return corrected_text, False, False
            
        except FuturesTimeoutError:
            # 超时：使用回退
            if self.config.verbose:
                print(f"[Pipeline] Agent B 超时 (>{self.config.agent_b_timeout_ms}ms)，使用回退")
            return fallback_text, True, True
            
        except Exception as e:
            # 其他错误：使用回退
            if self.config.verbose:
                print(f"[Pipeline] Agent B 调用失败: {e}，使用回退")
            return fallback_text, False, True
    
    def _log_router_features(
        self,
        result: PipelineResult,
        scoring_result,
        budget_details: Dict,
        boundary_stats: Dict = None,
        top2_info: Dict = None,
        lat_router_ms: float = 0.0,
    ):
        """
        记录路由特征到 router_features.jsonl
        
        Stage 0 要求的完整字段集:
        - id, img_path, blank_id, rho, T, N
        - char_conf[], blank_mean_L/R, blank_peak_L/R
        - top2_status, lat_router_ms
        - s_b, s_a, q, lambda_t, route_type, b_timeout
        
        Args:
            result: 流水线结果
            scoring_result: RuleOnlyScorer 评分结果
            budget_details: OnlineBudgetController 更新详情
            boundary_stats: 边界统计量 (来自 Agent A)
            top2_info: Top-2 信息 (来自 Agent A)
            lat_router_ms: 路由器耗时 (ms)
        """
        log_path = Path(self.config.router_features_log)
        
        # 确保目录存在
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 提取边界统计量
        bs = boundary_stats or {}
        t2 = top2_info or {}
        
        # 构建日志记录 (Stage 0 完整字段集)
        log_entry = {
            # ===== 基础信息 =====
            "id": result.id,
            "img_path": result.image_path,
            "timestamp": time_module.time(),
            
            # ===== Stage 0: 边界统计量 =====
            "blank_id": bs.get("blank_id", 0),
            "rho": bs.get("rho", 0.1),
            "T": bs.get("T", 0),                     # 序列长度
            "N": len(result.agent_a_text),           # 识别字符数
            "blank_mean_L": round(bs.get("blank_mean_L", 0.0), 6),
            "blank_mean_R": round(bs.get("blank_mean_R", 0.0), 6),
            "blank_peak_L": round(bs.get("blank_peak_L", 0.0), 6),
            "blank_peak_R": round(bs.get("blank_peak_R", 0.0), 6),
            "L_range": list(bs.get("L_range", (0, 0))),
            "R_range": list(bs.get("R_range", (0, 0))),
            "boundary_valid": bs.get("valid", False),
            
            # ===== Stage 0: Top-2 信息 =====
            "top2_status": t2.get("top2_status", "missing"),
            "top1_conf_mean": round(t2.get("top1_conf_mean", 0.0), 6),
            "top2_conf_mean": round(t2.get("top2_conf_mean", 0.0), 6),
            "conf_gap_mean": round(t2.get("conf_gap_mean", 0.0), 6),
            
            # ===== Stage 0: 耗时 =====
            "lat_router_ms": round(lat_router_ms, 3),
            
            # ===== RuleOnlyScorer 输出 =====
            "s_b": round(result.s_b, 6),
            "s_a": round(result.s_a, 6),
            "q": round(result.q, 6),
            "route_type": result.route_type,
            
            # ===== OnlineBudgetController 输出 =====
            "lambda_t": round(result.lambda_t, 6),
            "eta": self.config.eta,
            "upgrade": result.upgrade,
            
            # ===== 预算控制器详情 =====
            "budget_actual": round(budget_details.get("actual_budget", 0.0), 4),
            "budget_target": budget_details.get("target_budget", self.config.budget_target),
            "is_warmup": budget_details.get("is_warmup", True),
            
            # ===== 超时与回退 =====
            "b_timeout": result.b_timeout,
            "b_fallback": result.b_fallback,
            
            # ===== 评分详情 (可选) =====
            "scoring_details": scoring_result.details if hasattr(scoring_result, 'details') else {},
        }
        
        # 追加写入 JSONL
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        except Exception as e:
            if self.config.verbose:
                print(f"[Pipeline] 日志写入失败: {e}")
    
    def _get_environment_info(self) -> str:
        """获取运行环境信息"""
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                # 简化 GPU 名称
                if "2080" in gpu_name:
                    return "RTX2080Ti_4bit_quant" if self.config.agent_b_use_4bit else "RTX2080Ti"
                elif "3090" in gpu_name:
                    return "RTX3090_4bit_quant" if self.config.agent_b_use_4bit else "RTX3090"
                elif "4090" in gpu_name:
                    return "RTX4090_4bit_quant" if self.config.agent_b_use_4bit else "RTX4090"
                else:
                    return f"{gpu_name}_4bit_quant" if self.config.agent_b_use_4bit else gpu_name
            return "CPU"
        except:
            return "unknown"
    
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
        """获取统计信息 (SH-DA++ v4.0)"""
        stats = self.stats.copy()
        
        if stats["total_processed"] > 0:
            stats["agent_b_routing_rate"] = stats["routed_to_agent_b"] / stats["total_processed"]
            stats["agent_b_correction_rate"] = (
                stats["corrected_by_agent_b"] / stats["routed_to_agent_b"]
                if stats["routed_to_agent_b"] > 0 else 0.0
            )
            # SH-DA++ v4.0 统计
            stats["upgrade_rate"] = stats["upgrade_count"] / stats["total_processed"]
            stats["timeout_rate"] = (
                stats["b_timeout_count"] / stats["upgrade_count"]
                if stats["upgrade_count"] > 0 else 0.0
            )
            stats["fallback_rate"] = (
                stats["b_fallback_count"] / stats["upgrade_count"]
                if stats["upgrade_count"] > 0 else 0.0
            )
        
        # 添加 OnlineBudgetController 状态
        if self._budget_controller is not None:
            stats["budget_controller"] = self.budget_controller.get_stats()
        
        return stats
    
    def shutdown(self):
        """关闭流水线，释放资源"""
        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None


class AgentAAdapter:
    """
    Agent A 适配器 - 封装 TextRecognizerWithLogits
    
    将 TextRecognizerWithLogits 的输出格式适配为流水线期望的格式
    """
    
    def __init__(self, args, verbose: bool = False):
        """
        Args:
            args: PaddleOCR 参数对象
            verbose: 是否打印详细日志
        """
        from modules.paddle_engine.predict_rec_modified import TextRecognizerWithLogits
        
        self.verbose = verbose
        self._recognizer = TextRecognizerWithLogits(args)
        
        if verbose:
            print(f"[AgentAAdapter] 初始化完成, blank_id={self._recognizer.blank_id}, rho={self._recognizer.rho}")
    
    def recognize(self, image: Union[str, np.ndarray]) -> Dict:
        """
        识别图像
        
        Args:
            image: 图像路径或 numpy 数组
            
        Returns:
            Dict: {
                'text': 识别文本,
                'confidence': 置信度,
                'logits': 原始 logits (可选),
                'boundary_stats': 边界统计量,
                'top2_info': Top-2 信息,
                'lat_router_ms': 路由器耗时,
            }
        """
        import cv2
        
        # 加载图像
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                return {
                    'text': '',
                    'confidence': 0.0,
                    'logits': None,
                    'boundary_stats': {},
                    'top2_info': {},
                    'lat_router_ms': 0.0,
                    'error': f'无法读取图像: {image}',
                }
        else:
            img = image
        
        # 调用 TextRecognizerWithLogits
        output = self._recognizer([img])
        
        # 解析输出
        results = output.get('results', [])
        boundary_stats_list = output.get('boundary_stats', [])
        top2_info_list = output.get('top2_info', [])
        lat_router_ms_list = output.get('lat_router_ms', [])
        
        if not results:
            return {
                'text': '',
                'confidence': 0.0,
                'logits': None,
                'boundary_stats': {},
                'top2_info': {},
                'lat_router_ms': 0.0,
            }
        
        # 第一个结果
        text, confidence = results[0] if results else ('', 0.0)
        
        # 处理元组格式的结果
        if isinstance(results[0], (list, tuple)) and len(results[0]) >= 2:
            text, confidence = results[0][0], results[0][1]
        elif isinstance(results[0], dict):
            text = results[0].get('text', '')
            confidence = results[0].get('confidence', 0.0)
        
        return {
            'text': text,
            'confidence': float(confidence) if confidence else 0.0,
            'logits': None,  # 已在 TextRecognizerWithLogits 内部处理
            'boundary_stats': boundary_stats_list[0] if boundary_stats_list else {},
            'top2_info': top2_info_list[0] if top2_info_list else {},
            'lat_router_ms': lat_router_ms_list[0] if lat_router_ms_list else 0.0,
        }
    
    def recognize_batch(self, images: List[Union[str, np.ndarray]]) -> List[Dict]:
        """
        批量识别图像
        
        Args:
            images: 图像路径或 numpy 数组列表
            
        Returns:
            List[Dict]: 识别结果列表
        """
        import cv2
        
        # 加载图像
        img_list = []
        for image in images:
            if isinstance(image, str):
                img = cv2.imread(image)
                if img is None:
                    img = np.zeros((48, 320, 3), dtype=np.uint8)
            else:
                img = image
            img_list.append(img)
        
        if not img_list:
            return []
        
        # 调用 TextRecognizerWithLogits
        output = self._recognizer(img_list)
        
        # 解析输出
        results = output.get('results', [])
        boundary_stats_list = output.get('boundary_stats', [])
        top2_info_list = output.get('top2_info', [])
        lat_router_ms_list = output.get('lat_router_ms', [])
        
        # 构建结果列表
        output_list = []
        for i in range(len(img_list)):
            if i < len(results):
                text, confidence = results[i] if isinstance(results[i], (list, tuple)) else ('', 0.0)
            else:
                text, confidence = '', 0.0
            
            output_list.append({
                'text': text,
                'confidence': float(confidence) if confidence else 0.0,
                'logits': None,
                'boundary_stats': boundary_stats_list[i] if i < len(boundary_stats_list) else {},
                'top2_info': top2_info_list[i] if i < len(top2_info_list) else {},
                'lat_router_ms': lat_router_ms_list[i] if i < len(lat_router_ms_list) else 0.0,
            })
        
        return output_list


class MockAgentA:
    """Agent A 模拟版本 (SH-DA++ v4.0 兼容)"""
    
    def recognize(self, image: Union[str, np.ndarray]) -> Dict:
        """模拟识别，返回 SH-DA++ v4.0 格式的输出"""
        import random
        
        # 模拟识别结果
        sample_texts = [
            "中国科学院计算技术研究所",
            "在时间的未尾",
            "深度学习模型训练",
            "自然语言处理技术",
        ]
        
        text = random.choice(sample_texts)
        seq_len = 80
        
        # 模拟边界统计量
        boundary_stats = {
            'blank_mean_L': random.uniform(0.01, 0.1),
            'blank_mean_R': random.uniform(0.01, 0.1),
            'blank_peak_L': random.uniform(0.05, 0.2),
            'blank_peak_R': random.uniform(0.05, 0.2),
            'T': seq_len,
            'L_range': (0, 8),
            'R_range': (72, 80),
            'valid': True,
        }
        
        # 模拟 Top-2 信息
        top2_info = {
            'top2_status': 'available',
            'T': seq_len,
            'C': 6625,
            'top1_conf_mean': random.uniform(0.7, 0.95),
            'top2_conf_mean': random.uniform(0.01, 0.1),
            'conf_gap_mean': random.uniform(0.6, 0.9),
            'top2_indices': None,
            'top2_probs': None,
        }
        
        return {
            'text': text,
            'confidence': random.uniform(0.7, 0.95),
            'logits': None,
            'boundary_stats': boundary_stats,
            'top2_info': top2_info,
            'lat_router_ms': random.uniform(0.5, 3.0),
        }


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("L2W1 完整推理流水线测试")
    print("=" * 60)
    print("\n[注意] 此模块需要真实的模型文件才能运行")
    print("请确保已下载 PP-OCRv5 模型:")
    print("  wget https://paddle-model-ecology.bj.bcebos.com/model/ocr/PP-OCRv5/ch_PP-OCRv5_rec_infer.tar")
    print("  tar -xf ch_PP-OCRv5_rec_infer.tar")
    print("  mv ch_PP-OCRv5_rec_infer ./models/ppocrv5_rec")
    print("\n使用示例:")
    print("""
    from modules.pipeline import L2W1Pipeline, PipelineConfig
    
    config = PipelineConfig(
        agent_a_model_dir="./models/ppocrv5_rec",
        skip_agent_b=True,  # Stage 1 跳过 Agent B
        verbose=True,
    )
    
    pipeline = L2W1Pipeline(config)
    result = pipeline.process("./test_image.jpg")
    print(f"识别结果: {result.agent_a_text}")
    """)
    
    # 检查模型是否存在
    import os
    model_dir = "./models/ppocrv5_rec"
    if not os.path.exists(model_dir):
        print(f"\n[FATAL] 模型目录不存在: {model_dir}")
        print("请先下载模型后再运行测试")
        exit(1)
    
    # 配置（严格模式：必须提供真实模型）
    config = PipelineConfig(
        agent_a_model_dir=model_dir,
        skip_agent_b=True,  # Stage 1 跳过 Agent B
        verbose=True,
    )
    
    # 创建流水线
    pipeline = L2W1Pipeline(config)
    
    # 测试单张图像
    test_image = "./test_image.jpg"
    if os.path.exists(test_image):
        print(f"\n[测试] 处理图像: {test_image}")
        result = pipeline.process(test_image)
        
        print(f"\n[结果]")
        print(f"  Agent A 输出: '{result.agent_a_text}' (置信度: {result.agent_a_confidence:.2%})")
        print(f"  Router 决策: is_hard={result.is_hard}, risk_level={result.risk_level}")
        
        if result.routed_to_agent_b:
            print(f"  Agent B 输出: '{result.agent_b_text}' (已修正: {result.agent_b_is_corrected})")
        
        print(f"  最终文本: '{result.final_text}'")
        print(f"  处理时间: {result.processing_time_ms} ms")
        
        # SH-DA++ v4.0 新增字段
        print(f"  s_b={result.s_b:.4f}, s_a={result.s_a:.4f}, q={result.q:.4f}")
        print(f"  route_type={result.route_type}, lambda_t={result.lambda_t:.4f}, upgrade={result.upgrade}")
    else:
        print(f"\n[跳过] 测试图像不存在: {test_image}")
    
    # 关闭流水线
    pipeline.shutdown()
    print("\n[完成] 流水线测试结束")
    
    # 以下代码仅作为示例，不实际执行
    """
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

