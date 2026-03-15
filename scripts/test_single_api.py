#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test single Gemini API call"""
import sys
sys.path.insert(0, '.')

from modules.vlm_expert.gemini_expert import GeminiAgentB, GeminiConfig

# 创建 Agent
print("Creating Gemini Agent...")
agent = GeminiAgentB(config=GeminiConfig())
print(f"Agent info: {agent.get_model_info()}")

# 测试单个样本
manifest = {
    "ocr_text": "将来再次升成泻湖",
    "suspicious_index": 4,
    "suspicious_char": "升",
    "risk_level": "medium",
}

print("\n测试 API 调用...")
print(f"Original text: {manifest['ocr_text']}")
print(f"Suspicious char at index {manifest['suspicious_index']}: '{manifest['suspicious_char']}'")

try:
    result = agent.process_hard_sample(
        "data/geo/geotext/GeoP0153_L006.jpg",
        manifest
    )
    print(f"\nResult:")
    print(f"  Original:     {result['original_text']}")
    print(f"  Corrected:    {result['corrected_text']}")
    print(f"  Is corrected: {result['is_corrected']}")
    
    if result['is_corrected']:
        print(f"\n[OK] API 工作正常！修正了文本")
    else:
        print(f"\n[--] API 返回了原文，可能是：")
        print(f"    1. Gemini 认为没有错误")
        print(f"    2. API 调用失败被静默降级")
except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()
