#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v5.1: VLM Weight Downloader

支持通过 ModelScope 镜像下载以下模型到 models/agent_b_vlm/:
  - Qwen/Qwen2.5-VL-7B-Instruct
  - OpenGVLab/InternVL2-8B
  - openbmb/MiniCPM-V-2_6

使用方法:
  # 下载全部
  python scripts/download_vlms.py --all

  # 下载单个
  python scripts/download_vlms.py --model qwen2.5_vl
  python scripts/download_vlms.py --model internvl2
  python scripts/download_vlms.py --model minicpm_v

  # 指定下载源 (默认 modelscope)
  python scripts/download_vlms.py --all --source modelscope
  python scripts/download_vlms.py --all --source huggingface

注意:
  云服务器请使用 --source modelscope (默认)，无需配置代理。
  HuggingFace 镜像: 设置环境变量 HF_ENDPOINT=https://hf-mirror.com
"""

import argparse
import os
import sys
from pathlib import Path

# 模型配置
MODEL_REGISTRY = {
    "qwen2.5_vl": {
        "hf_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "ms_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "local_dir": "Qwen2.5-VL-7B-Instruct",
        "desc": "Qwen2.5-VL-7B-Instruct (~16GB bf16)",
    },
    "internvl2": {
        "hf_id": "OpenGVLab/InternVL2-8B",
        "ms_id": "OpenGVLab/InternVL2-8B",
        "local_dir": "InternVL2-8B",
        "desc": "InternVL2-8B (~16GB bf16)",
    },
    "minicpm_v": {
        "hf_id": "openbmb/MiniCPM-V-2_6",
        "ms_id": "OpenBMB/MiniCPM-V-2_6",
        "local_dir": "MiniCPM-V-2_6",
        "desc": "MiniCPM-V-2.6 (~16GB bf16)",
    },
}

OUTPUT_BASE = Path("models/agent_b_vlm")


def download_modelscope(model_key: str, cfg: dict) -> bool:
    """使用 ModelScope 下载（国内云服务器推荐）"""
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("[ERROR] modelscope 未安装，请运行: pip install modelscope")
        return False

    local_dir = OUTPUT_BASE / cfg["local_dir"]
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ModelScope] Downloading {cfg['desc']}")
    print(f"  Model ID : {cfg['ms_id']}")
    print(f"  Local dir: {local_dir}")

    try:
        snapshot_download(
            model_id=cfg["ms_id"],
            local_dir=str(local_dir),
            ignore_patterns=["*.bin"],  # 优先下载 safetensors
        )
        print(f"[OK] {model_key} downloaded to {local_dir}")
        return True
    except Exception as e:
        # 重试时包含 .bin 文件
        print(f"[WARN] safetensors 下载失败: {e}，尝试下载 .bin 格式...")
        try:
            snapshot_download(
                model_id=cfg["ms_id"],
                local_dir=str(local_dir),
            )
            print(f"[OK] {model_key} downloaded to {local_dir}")
            return True
        except Exception as e2:
            print(f"[ERROR] {model_key} download failed: {e2}")
            return False


def download_huggingface(model_key: str, cfg: dict) -> bool:
    """使用 HuggingFace Hub 下载（支持 hf-mirror.com 镜像）"""
    # 设置镜像
    if "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print(f"[HF] 使用镜像: {os.environ['HF_ENDPOINT']}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("[ERROR] huggingface_hub 未安装，请运行: pip install huggingface_hub")
        return False

    local_dir = OUTPUT_BASE / cfg["local_dir"]
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"[HuggingFace] Downloading {cfg['desc']}")
    print(f"  Repo ID  : {cfg['hf_id']}")
    print(f"  Local dir: {local_dir}")
    print(f"  Endpoint : {os.environ.get('HF_ENDPOINT', 'default')}")

    try:
        snapshot_download(
            repo_id=cfg["hf_id"],
            local_dir=str(local_dir),
            ignore_patterns=["*.bin"],  # 优先 safetensors
        )
        print(f"[OK] {model_key} downloaded to {local_dir}")
        return True
    except Exception as e:
        print(f"[WARN] safetensors 下载失败: {e}，尝试下载 .bin 格式...")
        try:
            snapshot_download(
                repo_id=cfg["hf_id"],
                local_dir=str(local_dir),
            )
            print(f"[OK] {model_key} downloaded to {local_dir}")
            return True
        except Exception as e2:
            print(f"[ERROR] {model_key} download failed: {e2}")
            return False


def main():
    parser = argparse.ArgumentParser(description="SH-DA++ VLM Weight Downloader")
    parser.add_argument(
        "--model",
        choices=list(MODEL_REGISTRY.keys()),
        help="下载指定模型",
    )
    parser.add_argument("--all", action="store_true", help="下载全部模型")
    parser.add_argument(
        "--source",
        choices=["modelscope", "huggingface"],
        default="modelscope",
        help="下载源 (默认 modelscope，云服务器推荐)",
    )
    args = parser.parse_args()

    if not args.model and not args.all:
        parser.print_help()
        sys.exit(1)

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    targets = list(MODEL_REGISTRY.keys()) if args.all else [args.model]

    print("=" * 60)
    print("SH-DA++ v5.1: VLM Weight Downloader")
    print(f"Source : {args.source}")
    print(f"Output : {OUTPUT_BASE.resolve()}")
    print(f"Models : {', '.join(targets)}")
    print("=" * 60)

    results = {}
    for model_key in targets:
        cfg = MODEL_REGISTRY[model_key]
        print(f"\n[{model_key}] {cfg['desc']}")
        if args.source == "modelscope":
            ok = download_modelscope(model_key, cfg)
        else:
            ok = download_huggingface(model_key, cfg)
        results[model_key] = ok

    print("\n" + "=" * 60)
    print("Download Summary:")
    for k, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  [{status:6s}] {k}: {MODEL_REGISTRY[k]['local_dir']}")
    print("=" * 60)

    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
