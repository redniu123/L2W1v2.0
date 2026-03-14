#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v5.1: Multi-VLM Smoke Test

对选定的本地 VLM 跑一张图，验证能正常返回修正后的字符串。

用法:
  # 使用 router_config.yaml 中的默认配置
  python scripts/smoke_test_multi_vlm.py

  # 覆盖 model_type
  python scripts/smoke_test_multi_vlm.py --model_type qwen2.5_vl
  python scripts/smoke_test_multi_vlm.py --model_type internvl2
  python scripts/smoke_test_multi_vlm.py --model_type minicpm_v

  # 指定模型路径
  python scripts/smoke_test_multi_vlm.py \\
    --model_type qwen2.5_vl \\
    --model_path ./models/agent_b_vlm/Qwen2.5-VL-7B-Instruct

  # 使用自定义图像
  python scripts/smoke_test_multi_vlm.py --image data/geo/geotext/GeoP0001_L001.jpg
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    parser = argparse.ArgumentParser(description="SH-DA++ v5.1: Multi-VLM Smoke Test")
    parser.add_argument("--config", default="configs/router_config.yaml")
    parser.add_argument(
        "--model_type",
        choices=["qwen2.5_vl", "internvl2", "minicpm_v"],
        default=None,
        help="覆盖 YAML 中的 model_type",
    )
    parser.add_argument("--model_path", default=None, help="覆盖 YAML 中的 model_path")
    parser.add_argument(
        "--image",
        default="data/geo/geotext/GeoP0001_L001.jpg",
        help="测试图像路径",
    )
    parser.add_argument("--ocr_text", default="固体物理", help="OCR 识别文本")
    parser.add_argument("--dtype", default=None, help="覆盖 dtype")
    args = parser.parse_args()

    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 命令行参数覆盖
    if args.model_type:
        config["agent_b"]["model_type"] = args.model_type
        config["agent_b"]["backend"] = "local_vlm"
    if args.model_path:
        config["agent_b"]["model_path"] = args.model_path
    if args.dtype:
        config["agent_b"]["dtype"] = args.dtype

    agent_b_cfg = config["agent_b"]
    print("=" * 60)
    print("SH-DA++ v5.1: Multi-VLM Smoke Test")
    print("=" * 60)
    print(f"Backend   : {agent_b_cfg.get('backend')}")
    print(f"Model type: {agent_b_cfg.get('model_type')}")
    print(f"Model path: {agent_b_cfg.get('model_path')}")
    print(f"Dtype     : {agent_b_cfg.get('dtype')}")
    print(f"Image     : {args.image}")
    print(f"OCR text  : {args.ocr_text}")
    print("=" * 60)

    # 检查图像存在
    img_path = Path(args.image)
    if not img_path.exists():
        print(f"[ERROR] Image not found: {img_path}")
        sys.exit(1)

    # 检查模型路径存在（local_vlm）
    if agent_b_cfg.get("backend") == "local_vlm":
        model_path = Path(agent_b_cfg.get("model_path", ""))
        if not model_path.exists():
            print(f"[ERROR] Model path not found: {model_path}")
            print(f"  请先运行: python scripts/download_vlms.py --model {agent_b_cfg.get('model_type')}")
            sys.exit(1)

    # 创建专家实例
    print("\n[1/3] Loading model...")
    from modules.vlm_expert import AgentBFactory

    try:
        expert = AgentBFactory.create(config)
        print(f"  Model info: {expert.get_model_info()}")
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 推理
    print("\n[2/3] Running inference...")
    manifest = {
        "ocr_text": args.ocr_text,
        "suspicious_index": 1,
        "suspicious_char": args.ocr_text[1] if len(args.ocr_text) > 1 else "",
        "risk_level": "medium",
    }

    try:
        result = expert.process_hard_sample(str(img_path), manifest)
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # 输出结果
    print("\n[3/3] Results:")
    print("-" * 40)
    print(f"Original : {result['original_text']}")
    print(f"Corrected: {result['corrected_text']}")
    print(f"Changed  : {result['is_corrected']}")
    print("-" * 40)

    if result["corrected_text"]:
        print("\n[PASS] Smoke test passed!")
    else:
        print("\n[WARN] Model returned empty string, check parse logic.")
        sys.exit(1)


if __name__ == "__main__":
    main()
