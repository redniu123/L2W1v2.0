"""
L2W1 索引管理工具

核心原则:
- 所有内部逻辑 (Router, Evaluator) 使用 0-indexed
- 只在生成 Agent B Prompt 时转换为 1-indexed (人类可读)

使用方法:
    from modules.utils import format_eip_index, validate_char_index
    
    # 在 Prompt 中使用
    prompt = f"第 {format_eip_index(suspicious_idx)} 个字符..."
    
    # 验证索引有效性
    is_valid, safe_idx = validate_char_index(idx, text)
"""

from typing import Tuple, Optional


def to_display_index(zero_indexed: int) -> int:
    """
    将 0-indexed 转换为人类可读的 1-indexed
    
    内部约定: 所有内部逻辑使用 0-indexed
    显示约定: 给用户/Agent B 看的使用 1-indexed
    
    Args:
        zero_indexed: 0-based 索引
        
    Returns:
        1-based 索引
    """
    return zero_indexed + 1


def from_display_index(one_indexed: int) -> int:
    """
    将 1-indexed 转换为程序使用的 0-indexed
    
    Args:
        one_indexed: 1-based 索引 (人类可读)
        
    Returns:
        0-based 索引 (程序使用)
    """
    return one_indexed - 1


def format_eip_index(
    zero_indexed: int, 
    to_1_indexed: bool = True
) -> str:
    """
    格式化 EIP (显式索引提示) 中的字符位置
    
    核心用途: 在生成 Agent B Prompt 时，确保索引显示一致
    
    Args:
        zero_indexed: 0-based 索引 (内部逻辑使用)
        to_1_indexed: 是否转换为 1-indexed (默认 True)
        
    Returns:
        格式化后的索引字符串
        
    Example:
        >>> format_eip_index(4)  # 内部索引 4
        "5"                      # 显示为第 5 个字符
        
        >>> format_eip_index(4, to_1_indexed=False)
        "4"                      # 保持 0-indexed (调试用)
    """
    if zero_indexed < 0:
        return "?"  # 无效索引
    
    if to_1_indexed:
        return str(zero_indexed + 1)
    else:
        return str(zero_indexed)


def validate_char_index(
    idx: int, 
    text: str,
    allow_negative: bool = False
) -> Tuple[bool, int]:
    """
    验证字符索引的有效性，并返回安全的索引值
    
    用于边界保护，防止索引越界
    
    Args:
        idx: 待验证的 0-based 索引
        text: 目标文本
        allow_negative: 是否允许负数索引 (表示无效/未知)
        
    Returns:
        Tuple[is_valid, safe_idx]:
            - is_valid: 索引是否有效
            - safe_idx: 安全的索引值 (越界时返回边界值)
            
    Example:
        >>> validate_char_index(10, "Hello")  # 越界
        (False, 4)  # 返回最后一个有效索引
        
        >>> validate_char_index(-1, "Hello", allow_negative=True)
        (True, -1)  # -1 表示未知/无效
    """
    if len(text) == 0:
        return (False, -1)
    
    if idx < 0:
        if allow_negative:
            return (True, idx)  # -1 是合法的"无效"标记
        else:
            return (False, 0)  # 钳制到第一个字符
    
    if idx >= len(text):
        return (False, len(text) - 1)  # 钳制到最后一个字符
    
    return (True, idx)


def get_char_at_index(text: str, idx: int, default: str = "?") -> str:
    """
    安全获取指定索引处的字符
    
    Args:
        text: 文本
        idx: 0-based 索引
        default: 索引无效时的默认值
        
    Returns:
        指定位置的字符，或默认值
    """
    if idx < 0 or idx >= len(text):
        return default
    return text[idx]


# =============================================================================
# 调试工具
# =============================================================================

def debug_index_info(
    idx: int, 
    text: str, 
    context: str = ""
) -> str:
    """
    生成索引调试信息
    
    Args:
        idx: 0-based 索引
        text: 目标文本
        context: 上下文说明
        
    Returns:
        格式化的调试信息
    """
    is_valid, safe_idx = validate_char_index(idx, text, allow_negative=True)
    
    char = get_char_at_index(text, idx)
    display_idx = format_eip_index(idx)
    
    lines = [
        f"[Index Debug] {context}",
        f"  Text: '{text}' (len={len(text)})",
        f"  Index: {idx} (0-indexed) → {display_idx} (1-indexed)",
        f"  Char: '{char}'",
        f"  Valid: {is_valid}",
    ]
    
    if not is_valid and safe_idx != idx:
        lines.append(f"  Safe Index: {safe_idx}")
    
    return "\n".join(lines)


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("L2W1 索引管理工具测试")
    print("=" * 60)
    
    # 测试基本转换
    print("\n[1] 基本转换测试:")
    for i in range(-1, 6):
        display = format_eip_index(i)
        print(f"  0-indexed {i} → 1-indexed {display}")
    
    # 测试验证
    print("\n[2] 索引验证测试:")
    text = "Hello"
    test_cases = [-1, 0, 2, 4, 5, 10]
    for idx in test_cases:
        is_valid, safe = validate_char_index(idx, text)
        char = get_char_at_index(text, idx)
        print(f"  idx={idx}: valid={is_valid}, safe={safe}, char='{char}'")
    
    # 测试调试信息
    print("\n[3] 调试信息测试:")
    print(debug_index_info(4, "在时间的未尾", "Router 输出"))
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)

