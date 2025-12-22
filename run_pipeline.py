#!/usr/bin/env python3
"""
L2W1 v5.0 端到端运行脚本

使用方法:
    python run_pipeline.py <image_path>
    python run_pipeline.py data/raw/images/line_001.jpg
"""

import sys
from pathlib import Path
from modules import L2W1Pipeline, PipelineConfig

def main():
    # 配置
    config = PipelineConfig(
        # Agent A 配置
        agent_a_model_dir="./models/agent_a_ppocr",
        
        # Agent B 配置（会自动从 HuggingFace 下载）
        agent_b_model_path="Qwen/Qwen2.5-VL-3B-Instruct",
        agent_b_use_4bit=True,
        
        # Router 配置
        entropy_threshold_low=2.0,
        entropy_threshold_high=4.0,
        ppl_threshold_low=50.0,
        ppl_threshold_high=200.0,
        
        # 其他
        verbose=True
    )
    
    # 获取图像路径
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        print("使用方法: python run_pipeline.py <image_path>")
        print("示例: python run_pipeline.py data/raw/images/line_001.jpg")
        sys.exit(1)
    
    if not Path(image_path).exists():
        print(f"错误: 图像文件不存在: {image_path}")
        sys.exit(1)
    
    # 创建 Pipeline
    print("=" * 60)
    print("L2W1 v5.0 Pipeline 初始化")
    print("=" * 60)
    print("正在加载模型...")
    print("  - Agent A: PP-OCRv5")
    print("  - Agent B: Qwen2.5-VL-3B (首次运行会自动下载)")
    print("  - Router: Uncertainty Router")
    print()
    
    try:
        pipeline = L2W1Pipeline(config)
        print("✓ Pipeline 初始化完成!\n")
    except Exception as e:
        print(f"✗ Pipeline 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 处理图像
    print("=" * 60)
    print(f"处理图像: {image_path}")
    print("=" * 60)
    
    try:
        result = pipeline.process(image_path)
    except Exception as e:
        print(f"✗ 处理失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 输出结果
    print("\n" + "=" * 60)
    print("推理结果")
    print("=" * 60)
    print(f"Agent A 识别: {result.agent_a_text}")
    print(f"置信度: {result.agent_a_confidence:.4f}")
    print()
    print(f"Router 决策:")
    print(f"  - 是否困难样本: {result.is_hard}")
    print(f"  - 风险等级: {result.risk_level}")
    print(f"  - 视觉熵: {result.visual_entropy:.4f}")
    print(f"  - 语义 PPL: {result.semantic_ppl:.2f}")
    if result.is_hard:
        print(f"  - 存疑字符索引: {result.suspicious_index} (字符: '{result.suspicious_char}')")
    print()
    if result.routed_to_agent_b:
        print(f"Agent B 修正: {result.agent_b_text}")
        print(f"是否修正: {result.agent_b_is_corrected}")
        print()
    print(f"最终输出: {result.final_text}")
    print("=" * 60)

if __name__ == "__main__":
    main()

