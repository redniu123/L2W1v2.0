"""
L2W1 Agent B - 视觉专家模块 (Visual Expert)

核心功能:
1. 加载 Qwen2.5-VL-3B-Instruct（4-bit 量化）
2. 动态分辨率处理：保留原始长宽比，避免几何失真
3. 显式索引提示 (EIP)：将开放式 OCR 降维为验证式纠错
4. 精准视觉重写：结合行级视觉上下文进行定点修正

技术栈:
- Model: Qwen2.5-VL-3B-Instruct
- Quantization: 4-bit (bitsandbytes)
- Acceleration: Flash Attention 2
- Target VRAM: < 11GB (RTX 2080Ti)

动态分辨率配置:
- min_pixels: 256 × 28 × 28 = 200,704
- max_pixels: 1280 × 28 × 28 = 1,003,520
- 支持极端长宽比 (最高 20:1)
"""

import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np

# 延迟导入，避免在未安装依赖时报错
torch = None
Image = None


@dataclass
class AgentBConfig:
    """Agent B 配置"""
    # 模型配置
    model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    # 量化配置
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    
    # 动态分辨率配置 (关键参数)
    min_pixels: int = 256 * 28 * 28      # 200,704
    max_pixels: int = 1280 * 28 * 28     # 1,003,520
    
    # 推理配置
    max_new_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9
    do_sample: bool = False
    
    # 硬件配置
    device: str = "cuda"
    use_flash_attention: bool = True
    
    # 资源限制
    max_vram_gb: float = 11.0  # RTX 2080Ti


class EIPPromptTemplate:
    """
    显式索引提示模板 (Explicit Index Prompting)
    
    将任务从"开放式 OCR"降维为"验证式纠错"
    """
    
    # 标准 EIP 模板
    STANDARD_TEMPLATE = """这是一张中文手写文档的行图片。
OCR识别结果："{ocr_text}"
系统检测到第 {suspicious_index} 个字符（当前识别为"{suspicious_char}"）存在视觉不确定性。

任务：
1. 请结合整行的视觉上下文，重新审视该位置的字迹。
2. 如果原识别正确，请保持不变；如果错误，请修正。
3. 输出修正后的整行文本。

修正后的文本："""

    # 多位置不确定模板
    MULTI_POSITION_TEMPLATE = """这是一张中文手写文档的行图片。
OCR识别结果："{ocr_text}"
系统检测到以下位置存在视觉不确定性：
{position_list}

任务：请结合整行的视觉上下文，审视存疑位置的字迹，输出修正后的整行文本。

修正后的文本："""

    # 高风险模板
    HIGH_RISK_TEMPLATE = """这是一张中文手写文档的行图片。
OCR识别结果："{ocr_text}"
该识别结果的置信度较低（风险等级：{risk_level}）。

系统提示第 {suspicious_index} 个字符"{suspicious_char}"最可能存在错误。

请仔细观察图片中的手写内容，输出正确的整行文本："""

    # 回退模板（无具体位置信息）
    FALLBACK_TEMPLATE = """这是一张中文手写文档的行图片。
OCR识别结果："{ocr_text}"
系统检测到该行文本可能存在识别错误。

请仔细观察图片中的手写内容，输出正确的整行文本："""

    @classmethod
    def build_prompt(
        cls,
        ocr_text: str,
        suspicious_index: int = -1,
        suspicious_char: str = "",
        risk_level: str = "medium",
        entropy_sequence: List[float] = None
    ) -> str:
        """
        构建 EIP 提示
        
        Args:
            ocr_text: OCR 识别文本
            suspicious_index: 存疑字符位置 (0-indexed，内部约定)
            suspicious_char: 存疑字符
            risk_level: 风险等级
            entropy_sequence: 熵序列（可选，用于多位置提示）
            
        Returns:
            格式化后的提示文本
            
        Note:
            索引转换规则: 
            - 内部逻辑 (Router, Evaluator) 使用 0-indexed
            - Prompt 显示使用 1-indexed (人类可读)
            - 使用 modules.utils.indexing 统一管理
        """
        # 导入统一索引管理工具
        try:
            from modules.utils.indexing import to_display_index
        except ImportError:
            # 回退：手动转换
            def to_display_index(idx): return idx + 1
        
        if suspicious_index >= 0 and suspicious_char:
            # 转换为 1-indexed (人类可读)
            display_index = to_display_index(suspicious_index)
            
            if risk_level == "high" or risk_level == "critical":
                return cls.HIGH_RISK_TEMPLATE.format(
                    ocr_text=ocr_text,
                    suspicious_index=display_index,
                    suspicious_char=suspicious_char,
                    risk_level=risk_level
                )
            else:
                return cls.STANDARD_TEMPLATE.format(
                    ocr_text=ocr_text,
                    suspicious_index=display_index,
                    suspicious_char=suspicious_char
                )
        else:
            return cls.FALLBACK_TEMPLATE.format(ocr_text=ocr_text)


class AgentBExpert:
    """
    L2W1 Agent B - 视觉专家
    
    基于 Qwen2.5-VL-3B 的视觉语言模型，用于处理困难样本的精准重写
    
    核心特性:
    1. 4-bit 量化：适配 11GB 显存
    2. 动态分辨率：保留原始长宽比，避免几何失真
    3. Flash Attention 2：加速长序列推理
    4. EIP 策略：显式索引引导的定点纠错
    """
    
    def __init__(self, config: AgentBConfig = None, lazy_init: bool = True):
        """
        Args:
            config: Agent B 配置
            lazy_init: 是否延迟初始化模型
        """
        self.config = config or AgentBConfig()
        self.model = None
        self.processor = None
        self._initialized = False
        
        if not lazy_init:
            self._init_model()
    
    def _import_dependencies(self):
        """导入依赖库"""
        global torch, Image
        
        if torch is None:
            import torch as _torch
            torch = _torch
        
        if Image is None:
            from PIL import Image as _Image
            Image = _Image
    
    def _init_model(self):
        """初始化模型和处理器"""
        if self._initialized:
            return
        
        self._import_dependencies()
        
        print("[Agent B] 正在初始化 Qwen2.5-VL-3B...")
        
        try:
            from transformers import (
                Qwen2_5_VLForConditionalGeneration,
                AutoProcessor,
                BitsAndBytesConfig
            )
        except ImportError:
            raise ImportError(
                "请安装 transformers>=4.40.0: pip install transformers>=4.40.0"
            )
        
        # 配置 4-bit 量化
        quantization_config = None
        if self.config.use_4bit:
            try:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                    bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
                )
                print("[Agent B] 4-bit 量化配置完成 (bitsandbytes)")
            except Exception as e:
                print(f"[Warning] 4-bit 量化配置失败: {e}")
                print("[Agent B] 将使用 float16 模式")
                quantization_config = None
        
        # 配置 Flash Attention
        attn_implementation = None
        if self.config.use_flash_attention:
            try:
                # 检测 Flash Attention 2 可用性
                import subprocess
                result = subprocess.run(
                    ["python", "-c", "import flash_attn"],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    attn_implementation = "flash_attention_2"
                    print("[Agent B] Flash Attention 2 已启用")
                else:
                    print("[Warning] Flash Attention 2 不可用，使用默认注意力机制")
            except Exception:
                print("[Warning] Flash Attention 2 检测失败，使用默认注意力机制")
        
        # 加载模型
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config
        
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        
        try:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.config.model_path,
                **model_kwargs
            )
            print(f"[Agent B] 模型加载完成: {self.config.model_path}")
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}")
        
        # 加载处理器（配置动态分辨率）
        try:
            self.processor = AutoProcessor.from_pretrained(
                self.config.model_path,
                trust_remote_code=True,
                min_pixels=self.config.min_pixels,
                max_pixels=self.config.max_pixels,
            )
            print(f"[Agent B] 处理器加载完成")
            print(f"  - min_pixels: {self.config.min_pixels:,}")
            print(f"  - max_pixels: {self.config.max_pixels:,}")
        except Exception as e:
            raise RuntimeError(f"处理器加载失败: {e}")
        
        # 显存检查
        if torch.cuda.is_available():
            vram_used = torch.cuda.max_memory_allocated() / (1024 ** 3)
            print(f"[Agent B] 显存占用: {vram_used:.2f} GB")
            if vram_used > self.config.max_vram_gb:
                warnings.warn(
                    f"显存占用 ({vram_used:.2f}GB) 超过目标限制 ({self.config.max_vram_gb}GB)"
                )
        
        self._initialized = True
        print("[Agent B] 初始化完成!")
    
    def _load_image(self, image: Union[str, np.ndarray, "Image.Image"]) -> "Image.Image":
        """
        加载图像
        
        Args:
            image: 图像路径、numpy 数组或 PIL Image
            
        Returns:
            PIL Image
        """
        self._import_dependencies()
        
        if isinstance(image, str):
            return Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).convert("RGB")
        elif hasattr(image, 'convert'):
            return image.convert("RGB")
        else:
            raise ValueError(f"不支持的图像类型: {type(image)}")
    
    def _build_messages(
        self,
        image: "Image.Image",
        prompt: str
    ) -> List[Dict]:
        """
        构建 Qwen2.5-VL 消息格式
        
        Args:
            image: PIL Image
            prompt: 提示文本
            
        Returns:
            消息列表
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        return messages
    
    def _parse_output(self, generated_text: str) -> str:
        """
        解析模型输出，提取修正后的文本
        
        Args:
            generated_text: 模型生成的原始文本
            
        Returns:
            清理后的修正文本
        """
        # 移除可能的前缀
        text = generated_text.strip()
        
        # 尝试提取"修正后的文本："之后的内容
        patterns = [
            r"修正后的文本[：:]\s*(.+)",
            r"正确的文本[：:]\s*(.+)",
            r"输出[：:]\s*(.+)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                text = match.group(1).strip()
                break
        
        # 移除引号
        text = text.strip('"\'""''')
        
        # 移除多余的空白和换行
        text = re.sub(r'\s+', '', text)
        
        return text
    
    def process_hard_sample(
        self,
        image: Union[str, np.ndarray, "Image.Image"],
        manifest: Dict
    ) -> Dict:
        """
        处理困难样本
        
        核心接口：接收 Router 输出的 manifest，进行视觉重写
        
        Args:
            image: 行级图像
            manifest: Router 输出的 JSON，包含:
                - is_hard: bool
                - suspicious_index: int
                - suspicious_char: str
                - risk_level: str
                - metrics: dict (可选)
                
        Returns:
            dict: {
                'original_text': str,      # 原始 OCR 文本
                'corrected_text': str,     # 修正后的文本
                'suspicious_index': int,   # 存疑位置
                'suspicious_char': str,    # 存疑字符
                'is_corrected': bool,      # 是否进行了修正
            }
        """
        # 确保模型已初始化
        if not self._initialized:
            self._init_model()
        
        # 提取 manifest 信息
        ocr_text = manifest.get('ocr_text', manifest.get('text', ''))
        suspicious_index = manifest.get('suspicious_index', -1)
        suspicious_char = manifest.get('suspicious_char', '')
        risk_level = manifest.get('risk_level', 'medium')
        entropy_sequence = manifest.get('entropy_sequence', None)
        
        # 加载图像
        pil_image = self._load_image(image)
        
        # 记录图像尺寸（调试用）
        w, h = pil_image.size
        aspect_ratio = w / h if h > 0 else 0
        
        # 构建 EIP 提示
        prompt = EIPPromptTemplate.build_prompt(
            ocr_text=ocr_text,
            suspicious_index=suspicious_index,
            suspicious_char=suspicious_char,
            risk_level=risk_level,
            entropy_sequence=entropy_sequence
        )
        
        # 构建消息
        messages = self._build_messages(pil_image, prompt)
        
        # 应用处理器
        text = self.processor.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[pil_image],
            padding=True,
            return_tensors="pt"
        )
        
        # 移动到设备
        inputs = inputs.to(self.model.device)
        
        # 生成
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample,
            )
        
        # 解码输出
        generated_ids_trimmed = [
            out_ids[len(in_ids):] 
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # 解析输出
        corrected_text = self._parse_output(output_text)
        
        # 判断是否进行了修正
        is_corrected = corrected_text != ocr_text
        
        return {
            'original_text': ocr_text,
            'corrected_text': corrected_text,
            'suspicious_index': suspicious_index,
            'suspicious_char': suspicious_char,
            'is_corrected': is_corrected,
            'image_size': (w, h),
            'aspect_ratio': aspect_ratio,
        }
    
    def correct_line(
        self,
        image: Union[str, np.ndarray, "Image.Image"],
        ocr_text: str,
        suspicious_idx: int,
        suspicious_char: str
    ) -> str:
        """
        便捷接口：行级文本纠错
        
        Args:
            image: 行级图像
            ocr_text: OCR 识别文本
            suspicious_idx: 存疑字符位置 (0-indexed)
            suspicious_char: 存疑字符
            
        Returns:
            修正后的文本
        """
        manifest = {
            'ocr_text': ocr_text,
            'suspicious_index': suspicious_idx,
            'suspicious_char': suspicious_char,
            'risk_level': 'medium',
        }
        
        result = self.process_hard_sample(image, manifest)
        return result['corrected_text']
    
    def batch_process(
        self,
        samples: List[Tuple[Union[str, np.ndarray], Dict]]
    ) -> List[Dict]:
        """
        批量处理
        
        注意：由于动态分辨率特性，目前实现为逐个处理
        
        Args:
            samples: [(image, manifest), ...]
            
        Returns:
            结果列表
        """
        results = []
        for image, manifest in samples:
            result = self.process_hard_sample(image, manifest)
            results.append(result)
        return results
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        info = {
            "model_path": self.config.model_path,
            "initialized": self._initialized,
            "use_4bit": self.config.use_4bit,
            "use_flash_attention": self.config.use_flash_attention,
            "min_pixels": self.config.min_pixels,
            "max_pixels": self.config.max_pixels,
        }
        
        if self._initialized and torch is not None and torch.cuda.is_available():
            info["vram_used_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)
        
        return info


# =============================================================================
# 模拟模式（用于测试，无需加载真实模型）
# =============================================================================

class AgentBExpertMock:
    """
    Agent B 模拟版本
    
    用于在无 GPU 或无模型权重时测试流水线逻辑
    """
    
    def __init__(self, config: AgentBConfig = None):
        self.config = config or AgentBConfig()
        self._initialized = True
    
    def process_hard_sample(
        self,
        image: Union[str, np.ndarray],
        manifest: Dict
    ) -> Dict:
        """模拟处理"""
        ocr_text = manifest.get('ocr_text', manifest.get('text', ''))
        suspicious_index = manifest.get('suspicious_index', -1)
        suspicious_char = manifest.get('suspicious_char', '')
        
        # 模拟修正：简单地返回原文（或模拟修正）
        corrected_text = ocr_text
        
        # 模拟一些常见的形近字修正
        corrections = {
            '未': '末', '末': '未',
            '己': '已', '已': '己',
            '土': '士', '士': '土',
            '日': '曰', '曰': '日',
        }
        
        if 0 <= suspicious_index < len(ocr_text):
            char = ocr_text[suspicious_index]
            if char in corrections:
                corrected_list = list(ocr_text)
                corrected_list[suspicious_index] = corrections[char]
                corrected_text = ''.join(corrected_list)
        
        return {
            'original_text': ocr_text,
            'corrected_text': corrected_text,
            'suspicious_index': suspicious_index,
            'suspicious_char': suspicious_char,
            'is_corrected': corrected_text != ocr_text,
            'image_size': (100, 50),
            'aspect_ratio': 2.0,
        }
    
    def correct_line(
        self,
        image,
        ocr_text: str,
        suspicious_idx: int,
        suspicious_char: str
    ) -> str:
        """模拟行级纠错"""
        manifest = {
            'ocr_text': ocr_text,
            'suspicious_index': suspicious_idx,
            'suspicious_char': suspicious_char,
        }
        result = self.process_hard_sample(image, manifest)
        return result['corrected_text']
    
    def get_model_info(self) -> Dict:
        return {
            "model_path": "MOCK",
            "initialized": True,
            "use_4bit": False,
            "mode": "simulation",
        }


# =============================================================================
# 便捷函数
# =============================================================================

def create_agent_b(
    model_path: str = None,
    use_mock: bool = False,
    **kwargs
) -> Union[AgentBExpert, AgentBExpertMock]:
    """
    创建 Agent B 实例
    
    Args:
        model_path: 模型路径
        use_mock: 是否使用模拟版本
        **kwargs: 其他配置参数
        
    Returns:
        Agent B 实例
    """
    if use_mock:
        return AgentBExpertMock()
    
    config = AgentBConfig()
    if model_path:
        config.model_path = model_path
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    return AgentBExpert(config=config)


def process_hard_sample(
    image: Union[str, np.ndarray],
    manifest: Dict,
    agent: AgentBExpert = None,
    use_mock: bool = False
) -> Dict:
    """
    便捷函数：处理困难样本
    
    Args:
        image: 图像
        manifest: Router 输出的 manifest
        agent: Agent B 实例（可选）
        use_mock: 是否使用模拟模式
        
    Returns:
        处理结果
    """
    if agent is None:
        agent = create_agent_b(use_mock=use_mock)
    
    return agent.process_hard_sample(image, manifest)


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("L2W1 Agent B 视觉专家测试")
    print("=" * 60)
    
    # 使用模拟模式测试
    print("\n[1] 创建 Agent B (模拟模式)...")
    agent = create_agent_b(use_mock=True)
    print(f"    模型信息: {agent.get_model_info()}")
    
    # 测试 EIP 提示模板
    print("\n[2] 测试 EIP 提示模板...")
    
    prompt = EIPPromptTemplate.build_prompt(
        ocr_text="在时间的未尾",
        suspicious_index=4,
        suspicious_char="未",
        risk_level="medium"
    )
    print(f"    提示内容:\n{prompt}")
    
    # 测试处理流程
    print("\n[3] 测试处理流程...")
    
    manifest = {
        'ocr_text': "在时间的未尾",
        'suspicious_index': 4,
        'suspicious_char': '未',
        'risk_level': 'medium',
        'metrics': {
            'visual_entropy': 3.5,
            'max_char_entropy': 5.2,
            'semantic_ppl': 150.0,
        }
    }
    
    # 使用模拟图像路径
    result = agent.process_hard_sample("./test_image.jpg", manifest)
    
    print(f"    原始文本: '{result['original_text']}'")
    print(f"    修正文本: '{result['corrected_text']}'")
    print(f"    是否修正: {result['is_corrected']}")
    print(f"    存疑位置: 第 {result['suspicious_index'] + 1} 个字符 '{result['suspicious_char']}'")
    
    # 测试便捷接口
    print("\n[4] 测试便捷接口...")
    
    corrected = agent.correct_line(
        image="./test_image.jpg",
        ocr_text="在时间的未尾",
        suspicious_idx=4,
        suspicious_char="未"
    )
    print(f"    correct_line() 输出: '{corrected}'")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
    
    # 提示真实模式使用方法
    print("""
真实模式使用示例:

    from modules.vlm_expert import AgentBExpert, AgentBConfig
    
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
            'ocr_text': "识别文本",
            'suspicious_index': 2,
            'suspicious_char': '字',
            'risk_level': 'high',
        }
    )
    
    print(f"修正结果: {result['corrected_text']}")
""")

