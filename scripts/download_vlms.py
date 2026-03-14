#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v5.1: VLM Weight Downloader (9-model edition)

支持通过 ModelScope 镜像下载 9 款 VLM 到 models/agent_b_vlm/
云服务器推荐使用 --source modelscope（无需代理）。

用法:
  python scripts/download_vlms.py --all
  python scripts/download_vlms.py --model qwen2.5_vl
  python scripts/download_vlms.py --model internvl2_awq
  python scripts/download_vlms.py --source huggingface --model llava
"""

import argparse
import os
import sys
from pathlib import Path

# ============================================================
# 模型注册表
# ============================================================
MODEL_REGISTRY = {
    "paddleocr_vl_15": {
        "hf_id":   "PaddlePaddle/PaddleOCR-VL-1.5",
        "ms_id":   "PaddlePaddle/PaddleOCR-VL-1.5",
        "local_dir": "PaddleOCR-VL-1.5",
        "desc": "PaddleOCR-VL-1.5",
    },
    "paddleocr_vl": {
        "hf_id":   "PaddlePaddle/PaddleOCR-VL",
        "ms_id":   "PaddlePaddle/PaddleOCR-VL",
        "local_dir": "PaddleOCR-VL",
        "desc": "PaddleOCR-VL",
    },
    "minicpm_v_int4": {
        "hf_id":   "openbmb/MiniCPM-V-2_6-int4",
        "ms_id":   "OpenBMB/MiniCPM-V-2_6-int4",
        "local_dir": "MiniCPM-V-2_6-int4",
        "desc": "MiniCPM-V-2.6-int4 (native int4, ~8GB)",
    },
    "qwen2.5_vl": {
        "hf_id":   "Qwen/Qwen2.5-VL-7B-Instruct",
        "ms_id":   "Qwen/Qwen2.5-VL-7B-Instruct",
        "local_dir": "Qwen2.5-VL-7B-Instruct",
        "desc": "Qwen2.5-VL-7B-Instruct (~15GB fp16)",
    },
    "internvl2_5_awq": {
        "hf_id":   "OpenGVLab/InternVL2_5-8B-AWQ",
        "ms_id":   "OpenGVLab/InternVL2_5-8B-AWQ",
        "local_dir": "InternVL2_5-8B-AWQ",
        "desc": "InternVL2.5-8B-AWQ (~8GB AWQ int4)",
    },
    "internvl2_awq": {
        "hf_id":   "OpenGVLab/InternVL2-8B-AWQ",
        "ms_id":   "OpenGVLab/InternVL2-8B-AWQ",
        "local_dir": "InternVL2-8B-AWQ",
        "desc": "InternVL2-8B-AWQ (~8GB AWQ int4)",
    },
    "qwen3.5": {
        "hf_id":   "Qwen/Qwen3.5-9B",
        "ms_id":   "Qwen/Qwen3.5-9B",
        "local_dir": "Qwen3.5-9B",
        "desc": "Qwen3.5-9B (~18GB fp16)",
    },
    "llava": {
        "hf_id":   "llava-hf/llava-1.5-7b-hf",
        "ms_id":   "llava-hf/llava-1.5-7b-hf",
        "local_dir": "llava-1.5-7b-hf",
        "desc": "LLaVA-1.5-7B (~14GB fp16)",
    },
    "smolvlm": {
        "hf_id":   "HuggingFaceTB/SmolVLM-500M-Instruct",
        "ms_id":   "HuggingFaceTB/SmolVLM-500M-Instruct",
        "local_dir": "SmolVLM-500M-Instruct",
        "desc": "SmolVLM-500M-Instruct (~1GB fp16, 极速冒烟)",
    },
}

OUTPUT_BASE = Path("models/agent_b_vlm")


def download_modelscope(model_key: str, cfg: dict) -> bool:
    """ModelScope 下载（国内云服务器首选）"""
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("[ERROR] 请先安装: pip install modelscope")
        return False

    local_dir = OUTPUT_BASE / cfg["local_dir"]
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ModelScope] {cfg['desc']} -> {local_dir}")
    try:
        snapshot_download(model_id=cfg["ms_id"], local_dir=str(local_dir),
                          ignore_patterns=["*.bin"])
        print(f"[OK] {model_key}")
        return True
    except Exception as e:
        print(f"[WARN] safetensors failed ({e}), retrying with .bin...")
        try:
            snapshot_download(model_id=cfg["ms_id"], local_dir=str(local_dir))
            print(f"[OK] {model_key}")
            return True
        except Exception as e2:
            print(f"[ERROR] {model_key}: {e2}")
            return False


def download_huggingface(model_key: str, cfg: dict) -> bool:
    """HuggingFace Hub 下载（支持 hf-mirror.com 镜像）"""
    if "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[ERROR] 请先安装: pip install huggingface_hub")
        return False

    local_dir = OUTPUT_BASE / cfg["local_dir"]
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"[HuggingFace/{os.environ['HF_ENDPOINT']}] {cfg['desc']} -> {local_dir}")
    try:
        snapshot_download(repo_id=cfg["hf_id"], local_dir=str(local_dir),
                          ignore_patterns=["*.bin"])
        print(f"[OK] {model_key}")
        return True
    except Exception as e:
        print(f"[WARN] safetensors failed ({e}), retrying with .bin...")
        try:
            snapshot_download(repo_id=cfg["hf_id"], local_dir=str(local_dir))
            print(f"[OK] {model_key}")
            return True
        except Exception as e2:
            print(f"[ERROR] {model_key}: {e2}")
            return False


def main():
    parser = argparse.ArgumentParser(description="SH-DA++ VLM Downloader (9-model)")
    parser.add_argument("--model", choices=list(MODEL_REGISTRY.keys()),
                        action="append", dest="models",
                        help="下载指定模型（可多次指定，如 --model qwen2.5_vl --model internvl2_awq）")
    parser.add_argument("--all", action="store_true", help="下载全部 9 款模型")
    parser.add_argument("--source", choices=["modelscope", "huggingface"],
                        default="modelscope", help="下载源（默认 modelscope）")
    args = parser.parse_args()

    if not args.models and not args.all:
        parser.print_help()
        print("\n可用模型:")
        for k, v in MODEL_REGISTRY.items():
            print(f"  {k:20s} {v['desc']}")
        sys.exit(0)

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)
    targets = list(MODEL_REGISTRY.keys()) if args.all else args.models

    print("=" * 60)
    print(f"SH-DA++ VLM Downloader | source={args.source} | models={len(targets)}")
    print("=" * 60)

    results = {}
    for key in targets:
        cfg = MODEL_REGISTRY[key]
        if args.source == "modelscope":
            results[key] = download_modelscope(key, cfg)
        else:
            results[key] = download_huggingface(key, cfg)

    print("\n" + "=" * 60)
    print("Summary:")
    for k, ok in results.items():
        print(f"  [{'OK    ' if ok else 'FAILED'}] {k}")
    print("=" * 60)
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
