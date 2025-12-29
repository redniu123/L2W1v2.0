#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
L2W1 V-CoT (Visual Chain-of-Thought) 提示词生成器

专门为 Agent B (VLM) 设计的边界补全提示词，
引导模型关注图像边缘区域，修复因压缩导致的边界字符丢失问题。

核心策略:
1. 边界聚焦: 引导 VLM 放大观察图像左右边缘
2. 对比推理: 提供 OCR 识别结果作为参考，进行差异分析
3. 完整转录: 输出包含边界字符的完整文本

Prompt 设计原则:
- 明确告知 OCR 可能的失误类型（边界截断）
- 提供结构化的 V-CoT 推理步骤
- 支持多种场景（仅左边丢失、仅右边丢失、两边都丢失）
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
from enum import Enum


class BoundaryRiskType(Enum):
    """边界风险类型"""
    NONE = "none"              # 无边界风险
    LEFT_ONLY = "left_only"    # 仅左边界风险
    RIGHT_ONLY = "right_only"  # 仅右边界风险
    BOTH = "both"              # 两边都有风险
    UNKNOWN = "unknown"        # 未知（几何检测触发但无字符级信息）


@dataclass
class VCoTPromptConfig:
    """V-CoT 提示词配置"""
    # 是否启用分步推理
    enable_step_by_step: bool = True
    
    # 是否显示置信度信息
    show_confidence_hint: bool = True
    
    # 输出语言
    language: str = "zh"  # "zh" 或 "en"
    
    # 最大补充字符数限制（防止幻觉）
    max_supplement_chars: int = 5
    
    # 是否要求 VLM 解释修改理由
    require_explanation: bool = False


class VCoTPrompter:
    """
    V-CoT 提示词生成器
    
    为 Agent B 生成针对边界补全的结构化提示词
    """
    
    def __init__(self, config: VCoTPromptConfig = None):
        self.config = config or VCoTPromptConfig()
    
    def detect_boundary_risk_type(
        self,
        left_confidence: float,
        right_confidence: float,
        threshold: float = 0.8
    ) -> BoundaryRiskType:
        """
        检测边界风险类型
        
        Args:
            left_confidence: 左边界置信度
            right_confidence: 右边界置信度
            threshold: 置信度阈值
            
        Returns:
            BoundaryRiskType: 风险类型
        """
        left_risk = left_confidence < threshold
        right_risk = right_confidence < threshold
        
        if left_risk and right_risk:
            return BoundaryRiskType.BOTH
        elif left_risk:
            return BoundaryRiskType.LEFT_ONLY
        elif right_risk:
            return BoundaryRiskType.RIGHT_ONLY
        else:
            return BoundaryRiskType.NONE
    
    def build_boundary_completion_prompt(
        self,
        pred_text: str,
        risk_type: BoundaryRiskType = BoundaryRiskType.BOTH,
        left_confidence: float = None,
        right_confidence: float = None,
        aspect_ratio: float = None,
    ) -> str:
        """
        构建边界补全提示词 (中文版)
        
        Args:
            pred_text: OCR 识别结果
            risk_type: 边界风险类型
            left_confidence: 左边界置信度（可选，用于提示）
            right_confidence: 右边界置信度（可选，用于提示）
            aspect_ratio: 图像长宽比（可选，用于提示）
            
        Returns:
            str: 完整的 V-CoT 提示词
        """
        if self.config.language == "en":
            return self._build_prompt_en(pred_text, risk_type, left_confidence, right_confidence, aspect_ratio)
        else:
            return self._build_prompt_zh(pred_text, risk_type, left_confidence, right_confidence, aspect_ratio)
    
    def _build_prompt_zh(
        self,
        pred_text: str,
        risk_type: BoundaryRiskType,
        left_confidence: float,
        right_confidence: float,
        aspect_ratio: float,
    ) -> str:
        """构建中文版 V-CoT 提示词"""
        
        # 系统角色设定
        system_context = (
            "你是一位专业的手写文本识别专家。"
            "你的任务是根据图像内容，补全或修正 OCR 模型可能遗漏的边界文字。"
        )
        
        # 问题陈述
        ocr_result_section = f"""
## OCR 识别结果
```
{pred_text}
```
"""
        
        # 根据风险类型生成观察指令
        if risk_type == BoundaryRiskType.LEFT_ONLY:
            observation_instruction = """
## 观察任务
⚠️ **系统检测到图像左侧边缘可能存在被截断的文字。**

请按照以下步骤进行观察和分析：

### 步骤 1：观察图像最左边
仔细查看图像的**最左侧边缘**，看是否有：
- 被截断的笔画或偏旁
- 完整但未被 OCR 识别的字符
- 模糊但可辨认的文字痕迹

### 步骤 2：对比 OCR 结果
将你观察到的内容与 OCR 识别结果对比：
- OCR 是否遗漏了左侧的字符？
- 首字是否被截断？

### 步骤 3：输出完整转录
如果发现遗漏，请在 OCR 结果的**开头**添加缺失的文字。
"""
        elif risk_type == BoundaryRiskType.RIGHT_ONLY:
            observation_instruction = """
## 观察任务
⚠️ **系统检测到图像右侧边缘可能存在被截断的文字。**

请按照以下步骤进行观察和分析：

### 步骤 1：观察图像最右边
仔细查看图像的**最右侧边缘**，看是否有：
- 被截断的笔画或偏旁
- 完整但未被 OCR 识别的字符
- 模糊但可辨认的文字痕迹

### 步骤 2：对比 OCR 结果
将你观察到的内容与 OCR 识别结果对比：
- OCR 是否遗漏了右侧的字符？
- 末字是否被截断？

### 步骤 3：输出完整转录
如果发现遗漏，请在 OCR 结果的**末尾**添加缺失的文字。
"""
        else:  # BOTH 或 UNKNOWN
            observation_instruction = """
## 观察任务
⚠️ **系统检测到图像左右两侧边缘可能存在被截断的文字。**

请按照以下步骤进行观察和分析：

### 步骤 1：观察图像最左边
仔细查看图像的**最左侧边缘**，看是否有：
- 被截断的笔画或偏旁
- 完整但未被 OCR 识别的字符
- 模糊但可辨认的文字痕迹

### 步骤 2：观察图像最右边
仔细查看图像的**最右侧边缘**，看是否有：
- 被截断的笔画或偏旁
- 完整但未被 OCR 识别的字符
- 模糊但可辨认的文字痕迹

### 步骤 3：对比 OCR 结果
将你观察到的内容与 OCR 识别结果对比：
- OCR 是否遗漏了左右两侧的字符？
- 首字或末字是否被截断？

### 步骤 4：输出完整转录
如果发现遗漏，请补全缺失的文字，确保转录内容完整。
"""
        
        # 置信度提示（可选）
        confidence_hint = ""
        if self.config.show_confidence_hint and (left_confidence is not None or right_confidence is not None):
            confidence_hint = "\n## 置信度提示\n"
            if left_confidence is not None:
                confidence_hint += f"- 左边界置信度: {left_confidence:.2%}\n"
            if right_confidence is not None:
                confidence_hint += f"- 右边界置信度: {right_confidence:.2%}\n"
            if aspect_ratio is not None:
                confidence_hint += f"- 图像长宽比: {aspect_ratio:.1f}:1\n"
        
        # 输出格式要求
        output_format = f"""
## 输出要求
请直接输出**完整的文本转录**，格式如下：

```
[完整文本]
```

注意事项：
1. 如果确认 OCR 结果无误，直接输出原文即可
2. 如果发现遗漏，补全后输出完整文本
3. 最多补充 {self.config.max_supplement_chars} 个字符（防止过度推测）
4. 不要添加任何解释或说明，只输出最终文本
"""
        
        # 组装完整提示词
        full_prompt = f"""{system_context}

{ocr_result_section}
{observation_instruction}
{confidence_hint}
{output_format}
"""
        
        return full_prompt.strip()
    
    def _build_prompt_en(
        self,
        pred_text: str,
        risk_type: BoundaryRiskType,
        left_confidence: float,
        right_confidence: float,
        aspect_ratio: float,
    ) -> str:
        """构建英文版 V-CoT 提示词"""
        
        system_context = (
            "You are a professional handwritten text recognition expert. "
            "Your task is to complete or correct characters that the OCR model may have missed at the image boundaries."
        )
        
        ocr_result_section = f"""
## OCR Recognition Result
```
{pred_text}
```
"""
        
        if risk_type == BoundaryRiskType.LEFT_ONLY:
            observation_instruction = """
## Observation Task
⚠️ **System detected potential truncated text at the LEFT edge of the image.**

Please follow these steps:

### Step 1: Examine the leftmost edge
Look carefully at the **left edge** of the image for:
- Truncated strokes or radicals
- Complete but unrecognized characters
- Blurry but discernible text traces

### Step 2: Compare with OCR result
Compare your observations with the OCR result:
- Did OCR miss characters on the left?
- Is the first character truncated?

### Step 3: Output complete transcription
If you find missing text, add it to the **beginning** of the OCR result.
"""
        elif risk_type == BoundaryRiskType.RIGHT_ONLY:
            observation_instruction = """
## Observation Task
⚠️ **System detected potential truncated text at the RIGHT edge of the image.**

Please follow these steps:

### Step 1: Examine the rightmost edge
Look carefully at the **right edge** of the image for:
- Truncated strokes or radicals
- Complete but unrecognized characters
- Blurry but discernible text traces

### Step 2: Compare with OCR result
Compare your observations with the OCR result:
- Did OCR miss characters on the right?
- Is the last character truncated?

### Step 3: Output complete transcription
If you find missing text, add it to the **end** of the OCR result.
"""
        else:
            observation_instruction = """
## Observation Task
⚠️ **System detected potential truncated text at BOTH edges of the image.**

Please follow these steps:

### Step 1: Examine the leftmost edge
Look for truncated or missed characters at the left boundary.

### Step 2: Examine the rightmost edge
Look for truncated or missed characters at the right boundary.

### Step 3: Compare with OCR result
Identify any discrepancies between your observations and the OCR result.

### Step 4: Output complete transcription
Provide the full text including any missing boundary characters.
"""
        
        output_format = f"""
## Output Requirements
Output the **complete text transcription** directly:

```
[Complete Text]
```

Notes:
1. If OCR result is correct, output it as-is
2. If you find missing text, include it in your output
3. Maximum {self.config.max_supplement_chars} supplementary characters allowed
4. No explanations needed, just the final text
"""
        
        full_prompt = f"""{system_context}

{ocr_result_section}
{observation_instruction}
{output_format}
"""
        
        return full_prompt.strip()
    
    def build_general_correction_prompt(
        self,
        pred_text: str,
        suspicious_char: str = None,
        suspicious_index: int = None,
    ) -> str:
        """
        构建通用纠错提示词（非边界场景）
        
        用于处理中间字符的识别错误
        """
        if self.config.language == "en":
            prompt = f"""You are a professional OCR correction expert.

## OCR Recognition Result
```
{pred_text}
```
"""
            if suspicious_char and suspicious_index is not None:
                prompt += f"""
## Suspicious Character
The character at position {suspicious_index + 1} ('{suspicious_char}') has low confidence.

Please carefully examine this character in the image and provide the correct transcription.
"""
            prompt += """
## Output Requirements
Output the corrected text directly, without explanation.
"""
        else:
            prompt = f"""你是一位专业的 OCR 纠错专家。

## OCR 识别结果
```
{pred_text}
```
"""
            if suspicious_char and suspicious_index is not None:
                prompt += f"""
## 存疑字符
第 {suspicious_index + 1} 个字符「{suspicious_char}」的置信度较低。

请仔细观察图像中该位置的字符，给出正确的转录。
"""
            prompt += """
## 输出要求
请直接输出修正后的文本，无需解释。
"""
        
        return prompt.strip()


# ==================== 便捷函数 ====================

def create_boundary_prompt(
    pred_text: str,
    left_confidence: float = None,
    right_confidence: float = None,
    aspect_ratio: float = None,
    threshold: float = 0.8,
    language: str = "zh",
) -> str:
    """
    快速创建边界补全提示词
    
    Args:
        pred_text: OCR 识别结果
        left_confidence: 左边界置信度
        right_confidence: 右边界置信度
        aspect_ratio: 图像长宽比
        threshold: 置信度阈值
        language: 输出语言 ("zh" 或 "en")
        
    Returns:
        str: V-CoT 提示词
    """
    config = VCoTPromptConfig(language=language)
    prompter = VCoTPrompter(config)
    
    # 检测风险类型
    if left_confidence is not None and right_confidence is not None:
        risk_type = prompter.detect_boundary_risk_type(left_confidence, right_confidence, threshold)
    else:
        risk_type = BoundaryRiskType.UNKNOWN
    
    return prompter.build_boundary_completion_prompt(
        pred_text=pred_text,
        risk_type=risk_type,
        left_confidence=left_confidence,
        right_confidence=right_confidence,
        aspect_ratio=aspect_ratio,
    )


# ==================== 测试代码 ====================

if __name__ == "__main__":
    # 测试示例
    print("=" * 60)
    print("V-CoT 提示词生成器测试")
    print("=" * 60)
    
    # 创建提示词生成器
    prompter = VCoTPrompter()
    
    # 测试场景 1: 左边界风险
    print("\n【场景 1: 左边界风险】")
    prompt1 = prompter.build_boundary_completion_prompt(
        pred_text="锦涛强调做好农业标准化和食品安",
        risk_type=BoundaryRiskType.LEFT_ONLY,
        left_confidence=0.65,
        right_confidence=0.92,
    )
    print(prompt1[:500] + "...")
    
    # 测试场景 2: 两边都有风险
    print("\n【场景 2: 两边边界风险】")
    prompt2 = create_boundary_prompt(
        pred_text="京4月24日讯中共中央政治局期27日下午进行第四",
        left_confidence=0.55,
        right_confidence=0.60,
        aspect_ratio=15.5,
    )
    print(prompt2[:500] + "...")
    
    # 测试场景 3: 英文版
    print("\n【场景 3: 英文版提示词】")
    prompt3 = create_boundary_prompt(
        pred_text="Hello Worl",
        left_confidence=0.95,
        right_confidence=0.45,
        language="en",
    )
    print(prompt3[:500] + "...")
    
    print("\n" + "=" * 60)
    print("测试完成!")

