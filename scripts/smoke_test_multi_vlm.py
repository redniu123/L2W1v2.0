#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v5.1: Multi-VLM Smoke Test

跑通一张验证集图片，打印峰值显存占用和 T_cand。

用法:
  python scripts/smoke_test_multi_vlm.py
  python scripts/smoke_test_multi_vlm.py --model_type smolvlm --model_path ./models/agent_b_vlm/SmolVLM-500M-Instruct
  python scripts/smoke_test_multi_vlm.py --model_type qwen2.5_vl --model_path ./models/agent_b_vlm/Qwen2.5-VL-7B-Instruct
  python scripts/smoke_test_multi_vlm.py --model_type qwen2.5_vl --model_path ./models/agent_b_vlm/Qwen3-VL-8B
  python scripts/smoke_test_multi_vlm.py --model_type internvl2 --model_path ./models/agent_b_vlm/InternVL2-8B-AWQ
"""

import argparse
import os
import sys
from pathlib import Path

# 默认使用第三张显卡 (index=2)
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def get_peak_vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def reset_vram_stats():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="SH-DA++ v5.1: Multi-VLM Smoke Test")
    parser.add_argument("--config", default="configs/router_config.yaml")
    parser.add_argument(
        "--model_type",
        choices=["qwen2.5_vl", "qwen3.5", "internvl2", "internvl2_5",
                 "minicpm_v", "paddleocr_vl", "llava", "smolvlm"],
        default=None,
    )
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--torch_dtype", default=None, choices=["float16", "bfloat16"])
    parser.add_argument("--image", default="data/raw/hctr_riskbench/val.jsonl",
                        help="图像路径或 val.jsonl（自动取第一张）")
    parser.add_argument("--ocr_text", default=None, help="指定 OCR 文本（不指定则从 jsonl 读取）")
    args = parser.parse_args()

    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # CLI 覆盖
    if args.model_type:
        config["agent_b"]["model_type"] = args.model_type
        config["agent_b"]["backend"] = "local_vlm"
    if args.model_path:
        config["agent_b"]["model_path"] = args.model_path
    if args.torch_dtype:
        config["agent_b"]["torch_dtype"] = args.torch_dtype

    cfg = config["agent_b"]

    print("=" * 60)
    print("SH-DA++ v5.1: Multi-VLM Smoke Test")
    print("=" * 60)
    print(f"  backend    : {cfg.get('backend')}")
    print(f"  model_type : {cfg.get('model_type')}")
    print(f"  model_path : {cfg.get('model_path')}")
    print(f"  torch_dtype: {cfg.get('torch_dtype')}")

    # 确定测试图像和 OCR 文本
    image_path = args.image
    ocr_text = args.ocr_text

    if image_path.endswith(".jsonl"):
        import json
        jsonl_path = Path(image_path)
        if not jsonl_path.exists():
            print(f"[ERROR] JSONL not found: {jsonl_path}")
            sys.exit(1)
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                sample = json.loads(line.strip())
                if sample:
                    break
        raw_img = sample.get("image_path") or sample.get("image", "")
        img_path = Path("data/geo") / raw_img
        ocr_text = ocr_text or sample.get("gt_text", "测试文本")
    else:
        img_path = Path(image_path)
        ocr_text = ocr_text or "固体物理"

    if not img_path.exists():
        print(f"[ERROR] Image not found: {img_path}")
        sys.exit(1)

    # 检查模型路径
    if cfg.get("backend") == "local_vlm":
        mp = Path(cfg.get("model_path", ""))
        if not mp.exists():
            print(f"[ERROR] Model not found: {mp}")
            print(f"  Run: python scripts/download_vlms.py --model {cfg.get('model_type')}")
            sys.exit(1)

    print(f"  image      : {img_path}")
    print(f"  ocr_text   : {ocr_text}")
    print("=" * 60)

    # 重置显存统计
    reset_vram_stats()

    # 加载模型
    print("\n[1/3] Loading model...")
    from modules.vlm_expert import AgentBFactory
    try:
        expert = AgentBFactory.create(config)
        print(f"  Info: {expert.get_model_info()}")
    except Exception as e:
        import traceback
        print(f"[ERROR] Load failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    vram_after_load = get_peak_vram_gb()
    print(f"  Peak VRAM after load: {vram_after_load:.2f} GB")

    # 推理
    print("\n[2/3] Running inference...")
    reset_vram_stats()
    manifest = {
        "ocr_text": ocr_text,
        "suspicious_index": 1,
        "suspicious_char": ocr_text[1] if len(ocr_text) > 1 else "",
        "risk_level": "medium",
    }
    try:
        result = expert.process_hard_sample(str(img_path), manifest)
    except Exception as e:
        import traceback
        print(f"[ERROR] Inference failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    vram_peak_infer = get_peak_vram_gb()

    # 结果输出
    print("\n[3/3] Results:")
    print("=" * 60)
    print(f"  Original  : {result['original_text']}")
    print(f"  T_cand    : {result['corrected_text']}")
    print(f"  Changed   : {result['is_corrected']}")
    print(f"  Peak VRAM (inference): {vram_peak_infer:.2f} GB")
    print("=" * 60)

    if result["corrected_text"]:
        print("[PASS] Smoke test passed!")
    else:
        print("[WARN] Empty output, check parse logic.")
        sys.exit(1)


if __name__ == "__main__":
    main()
