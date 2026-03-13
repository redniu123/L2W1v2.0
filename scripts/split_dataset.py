#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v5.1 Phase 0: 数据绝对隔离

将原始 JSONL 数据集按 Train(60%) / Val(20%) / Test(20%) 切分，
强制随机种子 42，确保可复现。

输出：data/raw/hctr_riskbench/train.jsonl
       data/raw/hctr_riskbench/val.jsonl
       data/raw/hctr_riskbench/test.jsonl
"""

import argparse
import json
from pathlib import Path

import numpy as np


def split_dataset(
    input_jsonl: str,
    output_dir: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "三个比例之和必须为 1.0"

    np.random.seed(seed)

    # 读取原始数据
    samples = []
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    n = len(samples)
    print(f"总样本数: {n}")

    # 随机打乱
    indices = np.random.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # test 取剩余，确保总和 = n
    n_test = n - n_train - n_val

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    splits = {
        "train": [samples[i] for i in train_idx],
        "val":   [samples[i] for i in val_idx],
        "test":  [samples[i] for i in test_idx],
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, split_samples in splits.items():
        out_path = output_dir / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for s in split_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")
        print(f"  {split_name:5s}: {len(split_samples):5d} 条 → {out_path}")

    # 写入 manifest
    manifest = {
        "dataset_version": "snapshot_v1",
        "source": str(input_jsonl),
        "num_samples": n,
        "split_seed": seed,
        "splits": {
            "train": {"num_samples": len(splits["train"]), "path": str(output_dir / "train.jsonl")},
            "val":   {"num_samples": len(splits["val"]),   "path": str(output_dir / "val.jsonl")},
            "test":  {"num_samples": len(splits["test"]),  "path": str(output_dir / "test.jsonl")},
        },
        "split_manifest_path": str(output_dir / "manifest.json"),
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n✓ 切分完成，manifest 已写入: {output_dir / 'manifest.json'}")
    print(f"  Train: {len(splits['train'])} / Val: {len(splits['val'])} / Test: {len(splits['test'])}")


def main():
    parser = argparse.ArgumentParser(description="SH-DA++ v5.1 Phase 0: 数据切分")
    parser.add_argument("--input_jsonl", type=str, default="data/geo/geotext.jsonl")
    parser.add_argument("--output_dir", type=str, default="data/raw/hctr_riskbench")
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_dataset(
        input_jsonl=args.input_jsonl,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
