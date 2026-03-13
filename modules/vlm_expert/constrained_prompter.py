#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v4.0 Constrained Prompt Templates

实现 BOUNDARY 和 AMBIGUITY 路径的受限提示词模板。

核心约束：
1. BOUNDARY: 只允许首尾修改
2. AMBIGUITY: 只允许单点 Top-2 内替换
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class PromptType(Enum):
    """提示词类型"""

    # v5.1 新增
    TARGETED_CORRECTION = "targeted_correction"

    BOUNDARY = "boundary"
    AMBIGUITY = "ambiguity"
    BOTH = "both"
    NONE = "none"


@dataclass
class PromptTemplate:
    """提示词模板"""

    system_prompt: str
    user_prompt: str
    constraints: List[str]


class ConstrainedPrompter:
    """
    SH-DA++ v4.0 受限提示词生成器

    根据路由类型生成结构化的约束提示词。
    """

    # BOUNDARY 路径模板
    BOUNDARY_TEMPLATE = PromptTemplate(
        system_prompt=(
            "你是一个严格的 OCR 纠错助手。你的任务是修正文本边界的漏字错误。"
            "你必须严格遵守以下约束：\n"
            "1. 只能在文本的【最前端】或【最后端】添加或修改字符\n"
            "2. 绝对禁止修改文本中间的任何字符\n"
            "3. 直接输出修正后的完整文本，不要任何解释"
        ),
        user_prompt=(
            "以下是初步识别结果：\n"
            "{T_A}\n\n"
            "图像左右边缘可能存在漏字。请结合提供的图像，仅在文本的【最前端】或【最后端】补充遗漏的字符。\n"
            "**绝对禁止**修改文本中间的任何字符。\n\n"
            "直接输出修正后的完整文本："
        ),
        constraints=[
            "只能修改首尾字符",
            "禁止修改中间字符",
            "不要添加解释",
        ],
    )

    # AMBIGUITY 路径模板
    AMBIGUITY_TEMPLATE = PromptTemplate(
        system_prompt=(
            "你是一个严格的 OCR 纠错助手。你的任务是修正单个字符的识别歧义。"
            "你必须严格遵守以下约束：\n"
            "1. 只能修改指定位置的字符\n"
            "2. 新字符必须从给定的两个候选字符中选择\n"
            "3. 绝对禁止修改其他位置的字符\n"
            "4. 直接输出修正后的完整文本，不要任何解释"
        ),
        user_prompt=(
            "以下是初步识别结果：\n"
            "{T_A}\n\n"
            "经检测，第 {idx_susp} 个字符（即 '{suspicious_char}'）置信度极低，存在歧义。\n"
            "它极有可能是以下两个字符之一：['{top1_char}', '{top2_char}']。\n\n"
            "请结合图像，在这两个候选字符中做出选择，并替换原字符。\n"
            "**绝对禁止**修改其他位置的字符。\n\n"
            "直接输出修正后的完整文本："
        ),
        constraints=[
            "只能修改指定位置",
            "必须从 Top-2 中选择",
            "禁止修改其他字符",
            "不要添加解释",
        ],
    )

    # BOTH 路径模板（两阶段）
    BOTH_BOUNDARY_TEMPLATE = PromptTemplate(
        system_prompt=(
            "你是一个严格的 OCR 纠错助手。你的任务是修正文本边界的漏字错误。"
            "这是第一阶段修正（边界修正）。\n"
            "你必须严格遵守以下约束：\n"
            "1. 只能在文本的【最前端】或【最后端】添加或修改字符\n"
            "2. 绝对禁止修改文本中间的任何字符\n"
            "3. 直接输出修正后的完整文本，不要任何解释"
        ),
        user_prompt=(
            "以下是初步识别结果：\n"
            "{T_A}\n\n"
            "【第一阶段：边界修正】\n"
            "图像左右边缘可能存在漏字。请仅在文本的【最前端】或【最后端】补充遗漏的字符。\n"
            "**绝对禁止**修改文本中间的任何字符。\n\n"
            "直接输出修正后的完整文本："
        ),
        constraints=[
            "只能修改首尾字符",
            "禁止修改中间字符",
            "不要添加解释",
        ],
    )

    BOTH_AMBIGUITY_TEMPLATE = PromptTemplate(
        system_prompt=(
            "你是一个严格的 OCR 纠错助手。你的任务是修正单个字符的识别歧义。"
            "这是第二阶段修正（歧义修正）。\n"
            "你必须严格遵守以下约束：\n"
            "1. 只能修改指定位置的字符\n"
            "2. 新字符必须从给定的两个候选字符中选择\n"
            "3. 绝对禁止修改其他位置的字符\n"
            "4. 直接输出修正后的完整文本，不要任何解释"
        ),
        user_prompt=(
            "以下是第一阶段修正后的文本：\n"
            "{T_A}\n\n"
            "【第二阶段：歧义修正】\n"
            "经检测，第 {idx_susp} 个字符（即 '{suspicious_char}'）置信度极低，存在歧义。\n"
            "它极有可能是以下两个字符之一：['{top1_char}', '{top2_char}']。\n\n"
            "请结合图像，在这两个候选字符中做出选择，并替换原字符。\n"
            "**绝对禁止**修改其他位置的字符。\n\n"
            "直接输出修正后的完整文本："
        ),
        constraints=[
            "只能修改指定位置",
            "必须从 Top-2 中选择",
            "禁止修改其他字符",
            "不要添加解释",
        ],
    )

    def __init__(self):
        """初始化提示词生成器"""
        pass



    # v5.1: Targeted Correction (Soft-Hint)
    def generate_targeted_correction_prompt(
        self, T_A, min_conf_idx=None, domain=None, image_path=None
    ):
        if domain is None:
            domain = "地质勘探"
        system_prompt = (
            "你是一个严格的 OCR 纠错专家，修正单行文本中的识别错误，保留原文原貌。"
        )
        hint_lines = []
        if min_conf_idx is not None and 0 <= min_conf_idx < len(T_A):
            hint_lines.append(
                "其中第 " + str(min_conf_idx + 1) + " 个字符的机器置信度极低，请重点关注。"
            )
        if domain:
            hint_lines.append(
                "本文本属于【" + domain + "】领域，请留意专业术语的准确性。"
            )
        sep = "\n"
        prefix = "系统检测到该文本可能存在识别错误。"
        if hint_lines:
            hint_block = prefix + "\n" + "\n".join(hint_lines) + "\n\n"
        else:
            hint_block = prefix + "\n\n"
        user_prompt = (
            "你是一个严格的 OCR 纠错专家，以下是初步的单行文本识别结果：\n"
            + "【 " + T_A + " 】\n\n"
            + hint_block
            + "请结合提供的图像，修正上述文本中的错别字或漏字。\n"
            + "**最高约束红线：**\n"
            + "1. 尽可能保持原句原貌，绝对禁止对句子进行润色、改写或大幅度增删。\n"
            + "2. 如果认为没有错误，请直接原样输出。\n\n"
            + "请直接输出修正后的完整文本，不要任何解释或多余字符："
        )
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "image_path": image_path,
            "prompt_type": "targeted_correction",
            "min_conf_idx": min_conf_idx,
            "domain": domain,
        }

    def generate_boundary_prompt(self, T_A: str, image_path: str = None) -> Dict:
        """
        生成 BOUNDARY 路径提示词

        Args:
            T_A: Agent A 输出文本
            image_path: 图像路径（可选）

        Returns:
            Dict: 包含 system_prompt, user_prompt, image_path
        """
        template = self.BOUNDARY_TEMPLATE

        user_prompt = template.user_prompt.format(T_A=T_A)

        return {
            "system_prompt": template.system_prompt,
            "user_prompt": user_prompt,
            "image_path": image_path,
            "constraints": template.constraints,
            "prompt_type": PromptType.BOUNDARY.value,
        }

    def generate_ambiguity_prompt(
        self,
        T_A: str,
        idx_susp: int,
        top2_chars: List[str],
        image_path: str = None,
    ) -> Dict:
        """
        生成 AMBIGUITY 路径提示词

        Args:
            T_A: Agent A 输出文本
            idx_susp: 存疑字符位置（0-indexed）
            top2_chars: Top-2 候选字符 [top1, top2]
            image_path: 图像路径（可选）

        Returns:
            Dict: 包含 system_prompt, user_prompt, image_path
        """
        if idx_susp < 0 or idx_susp >= len(T_A):
            raise ValueError(f"idx_susp={idx_susp} 超出文本范围 [0, {len(T_A)})")

        if not top2_chars or len(top2_chars) < 2:
            raise ValueError(f"top2_chars 必须包含 2 个字符，当前: {top2_chars}")

        template = self.AMBIGUITY_TEMPLATE

        suspicious_char = T_A[idx_susp]
        top1_char = top2_chars[0]
        top2_char = top2_chars[1]

        user_prompt = template.user_prompt.format(
            T_A=T_A,
            idx_susp=idx_susp + 1,  # 转换为 1-indexed（用户友好）
            suspicious_char=suspicious_char,
            top1_char=top1_char,
            top2_char=top2_char,
        )

        return {
            "system_prompt": template.system_prompt,
            "user_prompt": user_prompt,
            "image_path": image_path,
            "constraints": template.constraints,
            "prompt_type": PromptType.AMBIGUITY.value,
            "idx_susp": idx_susp,
            "top2_chars": top2_chars,
        }

    def generate_both_prompts(
        self,
        T_A: str,
        idx_susp: int,
        top2_chars: List[str],
        image_path: str = None,
    ) -> Dict:
        """
        生成 BOTH 路径提示词（两阶段）

        Args:
            T_A: Agent A 输出文本
            idx_susp: 存疑字符位置
            top2_chars: Top-2 候选字符
            image_path: 图像路径

        Returns:
            Dict: 包含 stage1 和 stage2 的提示词
        """
        # Stage 1: BOUNDARY
        stage1_template = self.BOTH_BOUNDARY_TEMPLATE
        stage1_user_prompt = stage1_template.user_prompt.format(T_A=T_A)

        stage1 = {
            "system_prompt": stage1_template.system_prompt,
            "user_prompt": stage1_user_prompt,
            "image_path": image_path,
            "constraints": stage1_template.constraints,
            "stage": 1,
        }

        # Stage 2: AMBIGUITY（需要在 Stage 1 完成后动态生成）
        # 这里提供模板，实际使用时需要用 Stage 1 的输出替换 T_A
        stage2_template = self.BOTH_AMBIGUITY_TEMPLATE

        stage2 = {
            "system_prompt": stage2_template.system_prompt,
            "user_prompt_template": stage2_template.user_prompt,
            "image_path": image_path,
            "constraints": stage2_template.constraints,
            "stage": 2,
            "idx_susp": idx_susp,
            "top2_chars": top2_chars,
        }

        return {
            "prompt_type": PromptType.BOTH.value,
            "stage1": stage1,
            "stage2": stage2,
        }

    def generate_stage2_prompt(
        self,
        T_A_stage1: str,
        idx_susp: int,
        top2_chars: List[str],
        image_path: str = None,
    ) -> Dict:
        """
        生成 BOTH 路径的 Stage 2 提示词

        Args:
            T_A_stage1: Stage 1 修正后的文本
            idx_susp: 存疑字符位置（需要重新计算）
            top2_chars: Top-2 候选字符
            image_path: 图像路径

        Returns:
            Dict: Stage 2 提示词
        """
        if idx_susp < 0 or idx_susp >= len(T_A_stage1):
            raise ValueError(
                f"idx_susp={idx_susp} 超出 Stage 1 文本范围 [0, {len(T_A_stage1)})"
            )

        template = self.BOTH_AMBIGUITY_TEMPLATE

        suspicious_char = T_A_stage1[idx_susp]
        top1_char = top2_chars[0]
        top2_char = top2_chars[1]

        user_prompt = template.user_prompt.format(
            T_A=T_A_stage1,
            idx_susp=idx_susp + 1,
            suspicious_char=suspicious_char,
            top1_char=top1_char,
            top2_char=top2_char,
        )

        return {
            "system_prompt": template.system_prompt,
            "user_prompt": user_prompt,
            "image_path": image_path,
            "constraints": template.constraints,
            "stage": 2,
            "idx_susp": idx_susp,
            "top2_chars": top2_chars,
        }

    def generate_prompt(
        self,
        prompt_type: str,
        T_A: str,
        idx_susp: Optional[int] = None,
        top2_chars: Optional[List[str]] = None,
        image_path: Optional[str] = None,
    ) -> Dict:
        """
        统一入口：根据类型生成提示词

        Args:
            prompt_type: 提示词类型（"boundary" / "ambiguity" / "both"）
            T_A: Agent A 输出文本
            idx_susp: 存疑字符位置（AMBIGUITY/BOTH 必需）
            top2_chars: Top-2 候选字符（AMBIGUITY/BOTH 必需）
            image_path: 图像路径

        Returns:
            Dict: 提示词字典
        """
        if prompt_type == "boundary":
            return self.generate_boundary_prompt(T_A, image_path)
        elif prompt_type == "ambiguity":
            if idx_susp is None or top2_chars is None:
                raise ValueError("AMBIGUITY 路径需要 idx_susp 和 top2_chars")
            return self.generate_ambiguity_prompt(T_A, idx_susp, top2_chars, image_path)
        elif prompt_type == "both":
            if idx_susp is None or top2_chars is None:
                raise ValueError("BOTH 路径需要 idx_susp 和 top2_chars")
            return self.generate_both_prompts(T_A, idx_susp, top2_chars, image_path)
        else:
            raise ValueError(f"未知的 prompt_type: {prompt_type}")


# 便捷函数
def create_boundary_prompt(T_A: str, image_path: str = None) -> Dict:
    """便捷函数：创建 BOUNDARY 提示词"""
    prompter = ConstrainedPrompter()
    return prompter.generate_boundary_prompt(T_A, image_path)


def create_ambiguity_prompt(
    T_A: str, idx_susp: int, top2_chars: List[str], image_path: str = None
) -> Dict:
    """便捷函数：创建 AMBIGUITY 提示词"""
    prompter = ConstrainedPrompter()
    return prompter.generate_ambiguity_prompt(T_A, idx_susp, top2_chars, image_path)


def create_both_prompts(
    T_A: str, idx_susp: int, top2_chars: List[str], image_path: str = None
) -> Dict:
    """便捷函数：创建 BOTH 提示词"""
    prompter = ConstrainedPrompter()
    return prompter.generate_both_prompts(T_A, idx_susp, top2_chars, image_path)


def create_targeted_correction_prompt(
    T_A: str,
    min_conf_idx=None,
    domain: str = '地质勘探',
    image_path=None,
):
    return ConstrainedPrompter().generate_targeted_correction_prompt(
        T_A, min_conf_idx, domain, image_path
    )
