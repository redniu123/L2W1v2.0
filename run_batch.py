#!/usr/bin/env python3
"""
L2W1 v5.0 批量处理脚本

使用方法:
    python run_batch.py
    需要先准备 data/raw/labels.txt 文件
"""

import json
from pathlib import Path
from modules import L2W1Pipeline, PipelineConfig

def main():
    # 配置
    config = PipelineConfig(
        agent_a_model_dir="./models/agent_a_ppocr",
        agent_b_model_path="Qwen/Qwen2.5-VL-3B-Instruct",
        agent_b_use_4bit=True,
        verbose=False  # 批量处理时关闭详细日志
    )
    
    # 读取标注文件
    labels_file = Path("data/raw/labels.txt")
    if not labels_file.exists():
        print(f"错误: 标注文件不存在: {labels_file}")
        print("请创建 data/raw/labels.txt，格式: image_path\\tground_truth")
        sys.exit(1)
    
    # 创建 Pipeline
    print("正在初始化 Pipeline...")
    try:
        pipeline = L2W1Pipeline(config)
        print("✓ Pipeline 初始化完成\n")
    except Exception as e:
        print(f"✗ Pipeline 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 读取数据
    print(f"读取标注文件: {labels_file}")
    samples = []
    with open(labels_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or '\t' not in line:
                continue
            
            try:
                image_path, gt_text = line.split('\t', 1)
                full_path = Path("data/raw") / image_path
                
                if not full_path.exists():
                    print(f"警告 [行 {line_num}]: 图像不存在: {full_path}")
                    continue
                
                samples.append((str(full_path), gt_text))
            except Exception as e:
                print(f"警告 [行 {line_num}]: 格式错误: {e}")
                continue
    
    if not samples:
        print("错误: 没有找到有效样本")
        sys.exit(1)
    
    print(f"找到 {len(samples)} 个有效样本\n")
    
    # 处理样本
    results = []
    for idx, (image_path, gt_text) in enumerate(samples, 1):
        print(f"[{idx}/{len(samples)}] 处理: {Path(image_path).name}")
        
        try:
            result = pipeline.process(image_path)
            
            results.append({
                'image': str(Path(image_path).relative_to(Path("data/raw"))),
                'agent_a_text': result.agent_a_text,
                'final_text': result.final_text,
                'gt_text': gt_text,
                'is_hard': result.is_hard,
                'routed_to_agent_b': result.routed_to_agent_b,
                'risk_level': result.risk_level,
                'visual_entropy': result.visual_entropy,
                'semantic_ppl': result.semantic_ppl
            })
        except Exception as e:
            print(f"  ✗ 处理失败: {e}")
            continue
    
    # 保存结果
    output_file = Path("data/test/inference_results.jsonl")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    print(f"\n处理完成!")
    print(f"  - 成功: {len(results)}/{len(samples)}")
    print(f"  - 结果保存到: {output_file}")
    
    # 统计信息
    hard_samples = sum(1 for r in results if r['is_hard'])
    routed_samples = sum(1 for r in results if r['routed_to_agent_b'])
    
    print(f"\n统计信息:")
    print(f"  - 困难样本: {hard_samples} ({hard_samples/len(results)*100:.1f}%)")
    print(f"  - 路由到 Agent B: {routed_samples} ({routed_samples/len(results)*100:.1f}%)")

if __name__ == "__main__":
    import sys
    main()

