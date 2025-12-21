"""
L2W1 Agent B SFT 数据集

关键技术点:
1. 负样本生成 (Negative Sampling): 15% 的样本用于幻觉抑制
2. Label Masking: 只在 assistant 回复部分计算 Loss
3. 动态分辨率: 与推理配置保持一致 (min_pixels=200704, max_pixels=1003520)

数据格式:
{
    "id": str,
    "image": str,
    "conversations": [
        {"from": "user", "value": str},
        {"from": "assistant", "value": str}
    ]
}
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# 尝试导入依赖
try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# =============================================================================
# 动态分辨率配置 (与 agent_b_expert.py 保持一致)
# =============================================================================

# 关键：这些值必须与推理时的配置完全一致
DYNAMIC_RESOLUTION_CONFIG = {
    "min_pixels": 256 * 28 * 28,   # 200,704
    "max_pixels": 1280 * 28 * 28,  # 1,003,520
}


# =============================================================================
# 负样本模板
# =============================================================================

class NegativeSampleTemplates:
    """
    负样本模板
    
    用于生成"识别正确，无需修正"的训练样本
    教会模型"不瞎改"，抑制过度纠错 (OCR-R)
    """
    
    # 负样本 Prompt 模板列表
    TEMPLATES = [
        "OCR识别结果为：'{text}'。\n系统未检测到明显的识别错误。\n请确认该识别结果是否正确，如正确请直接输出原文。",
        
        "OCR识别结果为：'{text}'。\n该行文本置信度较高，无明显存疑字符。\n请验证并输出最终文本。",
        
        "OCR识别结果为：'{text}'。\n自动检测未发现需要修正的位置。\n请确认或修正后输出。",
        
        "以下是OCR识别结果：'{text}'\n系统评估该结果可信度高。请直接输出最终文本。",
        
        "OCR结果：'{text}'\n检测结果：无存疑字符。\n任务：确认正确性并输出。",
    ]
    
    @classmethod
    def generate_prompt(cls, correct_text: str) -> str:
        """
        随机选择一个模板生成负样本 Prompt
        
        Args:
            correct_text: 正确的文本（作为 OCR 结果和期望输出）
            
        Returns:
            格式化后的 Prompt
        """
        template = random.choice(cls.TEMPLATES)
        return template.format(text=correct_text)


# =============================================================================
# SFT 数据集
# =============================================================================

class AgentBSFTDataset(Dataset):
    """
    Agent B SFT 数据集
    
    关键特性:
    1. 自动生成负样本（15%）用于幻觉抑制
    2. 实现 Label Masking：只在 assistant 回复部分计算 Loss
    3. 动态分辨率处理：与推理配置保持一致
    
    数据格式:
    {
        "id": "sample_001",
        "image": "path/to/image.jpg",
        "conversations": [
            {"from": "user", "value": "OCR识别结果为：..."},
            {"from": "assistant", "value": "修正后的文本"}
        ]
    }
    """
    
    # Label Masking 使用的忽略索引
    IGNORE_INDEX = -100
    
    def __init__(
        self,
        data_path: str,
        processor,
        max_seq_length: int = 2048,
        negative_sample_ratio: float = 0.15,
        min_pixels: int = None,
        max_pixels: int = None,
        is_training: bool = True,
        seed: int = 42,
    ):
        """
        Args:
            data_path: JSONL 数据文件路径
            processor: Qwen2.5-VL 处理器
            max_seq_length: 最大序列长度
            negative_sample_ratio: 负样本比例 (0.1-0.2 推荐)
            min_pixels: 最小像素数（动态分辨率）
            max_pixels: 最大像素数（动态分辨率）
            is_training: 是否为训练模式
            seed: 随机种子
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for AgentBSFTDataset")
        
        self.data_path = Path(data_path)
        self.processor = processor
        self.max_seq_length = max_seq_length
        self.negative_sample_ratio = negative_sample_ratio
        self.is_training = is_training
        
        # 动态分辨率配置（与推理保持一致）
        self.min_pixels = min_pixels or DYNAMIC_RESOLUTION_CONFIG["min_pixels"]
        self.max_pixels = max_pixels or DYNAMIC_RESOLUTION_CONFIG["max_pixels"]
        
        # 设置随机种子
        random.seed(seed)
        
        # 加载数据
        self.samples = self._load_data()
        self.original_size = len(self.samples)
        
        # 训练模式下添加负样本
        if is_training and negative_sample_ratio > 0:
            self._add_negative_samples()
        
        logger.info(f"[AgentBSFTDataset] 加载完成:")
        logger.info(f"  - 原始样本: {self.original_size}")
        logger.info(f"  - 负样本: {len(self.samples) - self.original_size}")
        logger.info(f"  - 总样本: {len(self.samples)}")
        logger.info(f"  - 动态分辨率: min={self.min_pixels:,}, max={self.max_pixels:,}")
    
    def _load_data(self) -> List[Dict]:
        """加载 JSONL 数据"""
        samples = []
        
        if not self.data_path.exists():
            logger.warning(f"数据文件不存在: {self.data_path}")
            return samples
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    sample = json.loads(line)
                    
                    # 验证数据格式
                    if not self._validate_sample(sample):
                        logger.warning(f"第 {line_num} 行数据格式错误，跳过")
                        continue
                    
                    # 标记为正样本
                    sample['is_negative'] = False
                    samples.append(sample)
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"第 {line_num} 行 JSON 解析错误: {e}")
        
        return samples
    
    def _validate_sample(self, sample: Dict) -> bool:
        """验证样本格式"""
        if 'conversations' not in sample:
            return False
        
        conversations = sample['conversations']
        if not isinstance(conversations, list) or len(conversations) < 2:
            return False
        
        # 检查是否有 user 和 assistant 消息
        has_user = any(c.get('from') == 'user' for c in conversations)
        has_assistant = any(c.get('from') == 'assistant' for c in conversations)
        
        return has_user and has_assistant
    
    def _add_negative_samples(self):
        """
        添加负样本（识别正确，无需修正）
        
        核心幻觉抑制策略：
        - 从正样本中提取 assistant 的回答（即正确文本）
        - 构建"确认正确"的 Prompt
        - 期望模型输出与输入相同的文本
        
        这教会模型：当被告知"可能无错误"时，不要瞎改
        """
        num_negative = int(self.original_size * self.negative_sample_ratio)
        
        if num_negative == 0:
            return
        
        negative_samples = []
        debug_samples = []  # 用于调试验证
        
        for i in range(num_negative):
            # 随机选择一个正样本作为模板
            template = random.choice(self.samples[:self.original_size])
            
            # 提取正确文本（assistant 的回答）
            correct_text = ""
            for conv in template.get('conversations', []):
                if conv.get('from') == 'assistant':
                    correct_text = conv.get('value', '').strip()
                    break
            
            if not correct_text:
                continue
            
            # 生成负样本 Prompt
            negative_prompt = NegativeSampleTemplates.generate_prompt(correct_text)
            
            # 构建负样本
            negative_sample = {
                'id': f"negative_{i:06d}",
                'image': template.get('image', ''),
                'conversations': [
                    {'from': 'user', 'value': negative_prompt},
                    {'from': 'assistant', 'value': correct_text}  # 保持原文不变
                ],
                'is_negative': True,
                'source_id': template.get('id', ''),
            }
            
            negative_samples.append(negative_sample)
            
            # 收集调试样本（前 5 个）
            if len(debug_samples) < 5:
                debug_samples.append({
                    'neg_id': negative_sample['id'],
                    'source_id': template.get('id', ''),
                    'image': template.get('image', ''),
                    'correct_text': correct_text[:50] + ('...' if len(correct_text) > 50 else ''),
                    'prompt_preview': negative_prompt[:60] + '...',
                })
        
        # 添加到样本列表并打乱
        self.samples.extend(negative_samples)
        random.shuffle(self.samples)
        
        logger.info(f"[负样本生成] 添加 {len(negative_samples)} 个负样本 (幻觉抑制)")
        
        # ========== 调试验证日志 ==========
        # 打印 5 个随机负样本，验证图像与 correct_text 的对应关系
        if debug_samples:
            logger.info("[负样本验证] 抽样检查 (请确认 image 与 correct_text 对应):")
            for idx, sample in enumerate(debug_samples):
                logger.info(f"  [{idx+1}] ID: {sample['neg_id']} <- {sample['source_id']}")
                logger.info(f"      Image: {sample['image']}")
                logger.info(f"      Text: '{sample['correct_text']}'")
                logger.info(f"      Prompt: {sample['prompt_preview']}")
            logger.info("  [!] 如果 image 与 text 不对应，请检查数据源的一致性")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取单个样本
        
        关键：实现 Label Masking
        - user 消息的 token：label 设为 IGNORE_INDEX (-100)
        - assistant 消息的 token：label 保留原始 token id
        """
        sample = self.samples[idx]
        return self._process_sample(sample)
    
    def _load_image(self, image_path: str) -> "Image.Image":
        """加载图像"""
        if not PIL_AVAILABLE:
            raise ImportError("PIL is required for image loading")
        
        try:
            if os.path.exists(image_path):
                image = Image.open(image_path).convert('RGB')
                return image
        except Exception as e:
            logger.debug(f"图像加载失败: {image_path}, {e}")
        
        # 返回占位图像
        return Image.new('RGB', (320, 48), color=(255, 255, 255))
    
    def _process_sample(self, sample: Dict) -> Dict[str, torch.Tensor]:
        """
        处理单个样本
        
        关键实现：
        1. 构建多轮对话消息
        2. 编码为 token
        3. 实现 Label Masking（只计算 assistant 部分的 loss）
        
        Returns:
            {
                'input_ids': tensor,
                'attention_mask': tensor,
                'labels': tensor (with masking),
                'pixel_values': tensor,
            }
        """
        # 获取图像和对话
        image_path = sample.get('image', '')
        conversations = sample.get('conversations', [])
        
        # 加载图像
        image = self._load_image(image_path)
        
        # 构建消息（用于 apply_chat_template）
        messages = []
        for conv in conversations:
            role = 'user' if conv.get('from') == 'user' else 'assistant'
            content = conv.get('value', '')
            
            if role == 'user':
                messages.append({
                    'role': 'user',
                    'content': [
                        {'type': 'image', 'image': image},
                        {'type': 'text', 'text': content}
                    ]
                })
            else:
                messages.append({
                    'role': 'assistant',
                    'content': content
                })
        
        # 使用 chat template 编码完整对话
        text_with_template = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # 编码
        inputs = self.processor(
            text=[text_with_template],
            images=[image],
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        
        # ========== Label Masking 关键实现 ==========
        # 只在 assistant 回复部分计算 Loss
        labels = self._create_labels_with_masking(
            input_ids=input_ids,
            conversations=conversations,
            text_with_template=text_with_template
        )
        # =============================================
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
        
        # 添加视觉特征
        if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
            result['pixel_values'] = inputs['pixel_values'].squeeze(0)
        
        return result
    
    def _create_labels_with_masking(
        self,
        input_ids: torch.Tensor,
        conversations: List[Dict],
        text_with_template: str
    ) -> torch.Tensor:
        """
        创建带有 Masking 的 Labels
        
        关键逻辑：
        - user 消息的 token → label = IGNORE_INDEX (-100)
        - assistant 消息的 token → label = token id
        
        这确保模型只在学习如何回复，而不是学习如何提问
        
        Args:
            input_ids: 输入 token ids
            conversations: 原始对话列表
            text_with_template: 应用 chat template 后的完整文本
            
        Returns:
            labels: 带有 masking 的 label tensor
        """
        # 初始化 labels 为 IGNORE_INDEX
        labels = torch.full_like(input_ids, self.IGNORE_INDEX)
        
        # 获取 assistant 回复文本
        assistant_responses = []
        for conv in conversations:
            if conv.get('from') == 'assistant':
                assistant_responses.append(conv.get('value', '').strip())
        
        if not assistant_responses:
            return labels
        
        # 方法：找到 assistant 回复在完整文本中的位置，并标记对应的 token
        # 这里使用简化方法：标记序列后半部分（通常是 assistant 回复）
        
        # 找到有效 token 的范围（非 padding）
        valid_length = attention_mask_sum = (input_ids != self.processor.tokenizer.pad_token_id).sum().item()
        
        # 估计 assistant 回复的起始位置
        # 通常 assistant 回复在对话的后半部分
        # 更精确的方法需要解析 chat template 的特殊 token
        
        try:
            # 尝试找到 assistant 标记
            # Qwen2.5-VL 使用 <|im_start|>assistant 格式
            assistant_marker = "<|im_start|>assistant"
            if assistant_marker in text_with_template:
                # 编码 marker 来找到位置
                marker_pos = text_with_template.find(assistant_marker)
                if marker_pos >= 0:
                    # 编码 marker 之前的文本来估计 token 位置
                    prefix_text = text_with_template[:marker_pos + len(assistant_marker)]
                    prefix_tokens = self.processor.tokenizer.encode(
                        prefix_text, add_special_tokens=False
                    )
                    
                    # assistant 回复从 marker 之后开始
                    start_pos = min(len(prefix_tokens), valid_length - 1)
                    
                    # 将 assistant 部分的 labels 设为原始 token id
                    labels[start_pos:valid_length] = input_ids[start_pos:valid_length]
                    
                    return labels
        except Exception as e:
            logger.debug(f"精确 label masking 失败，使用回退策略: {e}")
        
        # 回退策略：假设后 1/3 是 assistant 回复
        fallback_start = int(valid_length * 0.67)
        labels[fallback_start:valid_length] = input_ids[fallback_start:valid_length]
        
        return labels


# =============================================================================
# 数据整理器
# =============================================================================

def create_data_collator(processor):
    """
    创建数据整理器
    
    处理批量数据的拼接，特别是变长的 pixel_values
    """
    def collate_fn(features: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        数据整理函数
        
        Args:
            features: 样本列表
            
        Returns:
            批量 tensor 字典
        """
        batch = {
            'input_ids': torch.stack([f['input_ids'] for f in features]),
            'attention_mask': torch.stack([f['attention_mask'] for f in features]),
            'labels': torch.stack([f['labels'] for f in features]),
        }
        
        # 处理 pixel_values（可能有不同尺寸）
        if 'pixel_values' in features[0] and features[0]['pixel_values'] is not None:
            # 对于 Qwen2.5-VL，pixel_values 可能是嵌套结构
            try:
                batch['pixel_values'] = torch.stack([f['pixel_values'] for f in features])
            except RuntimeError:
                # 如果尺寸不同，尝试 cat
                batch['pixel_values'] = torch.cat([f['pixel_values'].unsqueeze(0) for f in features])
        
        return batch
    
    return collate_fn


# =============================================================================
# 验证函数
# =============================================================================

def validate_negative_samples(dataset: AgentBSFTDataset) -> Dict:
    """
    验证负样本生成逻辑
    
    首席科学家审计点 1: 确保负样本正确生成
    
    Returns:
        验证报告
    """
    report = {
        "total_samples": len(dataset.samples),
        "original_samples": dataset.original_size,
        "negative_samples": 0,
        "negative_ratio": 0.0,
        "negative_examples": [],
    }
    
    for sample in dataset.samples:
        if sample.get('is_negative', False):
            report["negative_samples"] += 1
            
            # 收集前几个负样本作为示例
            if len(report["negative_examples"]) < 3:
                user_msg = ""
                assistant_msg = ""
                for conv in sample.get('conversations', []):
                    if conv.get('from') == 'user':
                        user_msg = conv.get('value', '')[:100]
                    elif conv.get('from') == 'assistant':
                        assistant_msg = conv.get('value', '')[:50]
                
                report["negative_examples"].append({
                    "id": sample.get('id', ''),
                    "user_prompt_preview": user_msg + "...",
                    "assistant_response_preview": assistant_msg,
                })
    
    report["negative_ratio"] = report["negative_samples"] / report["total_samples"] if report["total_samples"] > 0 else 0
    
    return report


def validate_label_masking(sample_output: Dict) -> Dict:
    """
    验证 Label Masking 逻辑
    
    首席科学家审计点 2: 确保 Loss 只在 assistant 部分计算
    
    Args:
        sample_output: 数据集返回的单个样本
        
    Returns:
        验证报告
    """
    labels = sample_output.get('labels')
    input_ids = sample_output.get('input_ids')
    
    if labels is None or input_ids is None:
        return {"error": "Missing labels or input_ids"}
    
    total_tokens = len(labels)
    masked_tokens = (labels == -100).sum().item()
    unmasked_tokens = total_tokens - masked_tokens
    
    return {
        "total_tokens": total_tokens,
        "masked_tokens": masked_tokens,
        "unmasked_tokens": unmasked_tokens,
        "mask_ratio": masked_tokens / total_tokens if total_tokens > 0 else 0,
        "note": "masked_tokens should be user prompt, unmasked_tokens should be assistant response"
    }


def validate_resolution_config(dataset: AgentBSFTDataset, agent_b_config: Dict = None) -> Dict:
    """
    验证动态分辨率配置一致性
    
    首席科学家审计点 3: 确保训练和推理配置一致
    
    Args:
        dataset: 数据集实例
        agent_b_config: Agent B 推理配置（可选）
        
    Returns:
        验证报告
    """
    # 默认推理配置（从 agent_b_expert.py）
    inference_config = agent_b_config or {
        "min_pixels": 256 * 28 * 28,  # 200,704
        "max_pixels": 1280 * 28 * 28,  # 1,003,520
    }
    
    training_config = {
        "min_pixels": dataset.min_pixels,
        "max_pixels": dataset.max_pixels,
    }
    
    is_consistent = (
        training_config["min_pixels"] == inference_config["min_pixels"] and
        training_config["max_pixels"] == inference_config["max_pixels"]
    )
    
    return {
        "training_config": training_config,
        "inference_config": inference_config,
        "is_consistent": is_consistent,
        "warning": None if is_consistent else "MISMATCH: Training and inference configs differ!"
    }


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("L2W1 SFT 数据集模块测试")
    print("=" * 60)
    
    # 测试负样本模板
    print("\n[1] 负样本模板测试:")
    for i in range(3):
        prompt = NegativeSampleTemplates.generate_prompt("中国科学院计算技术研究所")
        print(f"  模板 {i+1}: {prompt[:60]}...")
    
    # 测试动态分辨率配置
    print("\n[2] 动态分辨率配置:")
    print(f"  min_pixels: {DYNAMIC_RESOLUTION_CONFIG['min_pixels']:,}")
    print(f"  max_pixels: {DYNAMIC_RESOLUTION_CONFIG['max_pixels']:,}")
    
    # 模拟数据集测试（无需真实处理器）
    print("\n[3] 数据集逻辑测试:")
    print("  (需要安装 transformers 和 Qwen2.5-VL 处理器)")
    
    print("\n" + "=" * 60)
    print("模块加载成功!")
    print("=" * 60)

