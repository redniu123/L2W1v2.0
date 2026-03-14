#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v5.1: Quick Test on 100 Samples

快速测试脚本：
- 只跑前 100 个样本
- 预算点：B=0.05, 0.20
- 3 个策略并行：Random / ConfOnly / SH-DA++
- 控制并发避免 API 封禁
"""

import argparse
import csv
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 复用 run_efficiency_frontier.py 的函数
from scripts.run_efficiency_frontier import (
    build_agent_b_callable,
    compute_cer,
    infer_all_samples,
    run_pipeline,
)


def main():
    parser = argparse.ArgumentParser(description="SH-DA++ v5.1: Quick Test (100 samples)")
    parser.add_argument("--config", default="configs/router_config.yaml")
    parser.add_argument("--test_jsonl", default="data/raw/hctr_riskbench/test.jsonl")
    parser.add_argument("--image_root", default="data/geo")
    parser.add_argument("--rec_model_dir", default="./models/agent_a_ppocr/PP-OCRv5_server_rec_infer")
    parser.add_argument("--rec_char_dict_path", default="ppocr/utils/ppocrv5_dict.txt")
    parser.add_argument("--geo_dict", default="data/dicts/Geology.txt")
    parser.add_argument("--output_dir", default="results/stage2_v51")
    parser.add_argument("--n_samples", type=int, default=100, help="Number of samples to test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("=" * 60)
    print(f"SH-DA++ v5.1: Quick Test ({args.n_samples} samples)")
    print("=" * 60)

    # 初始化 Agent A
    print("\n[1/4] Init Agent A...")
    import argparse as _ap
    rec_args = _ap.Namespace(
        rec_model_dir=args.rec_model_dir,
        rec_char_dict_path=args.rec_char_dict_path,
        rec_image_shape="3, 48, 320",
        rec_batch_num=6,
        rec_algorithm="SVTR_LCNet",
        use_space_char=True,
        use_gpu=True,  # GPU 模式
        use_xpu=False,
        use_npu=False,
        use_mlu=False,
        use_metax_gpu=False,
        use_gcu=False,
        ir_optim=True,
        use_tensorrt=False,
        min_subgraph_size=15,
        precision="fp32",
        gpu_mem=500,
        gpu_id=0,
        enable_mkldnn=None,
        cpu_threads=10,
        warmup=False,
        benchmark=False,
        save_log_path="./log_output/",
        show_log=False,
        use_onnx=False,
        max_batch_size=10,
        return_word_box=False,
        drop_score=0.5,
        max_text_length=25,
        rec_image_inverse=True,
        use_det=False,
        det_model_dir="",
    )
    from modules.paddle_engine.predict_rec_modified import TextRecognizerWithLogits
    recognizer = TextRecognizerWithLogits(rec_args)

    # 初始化组件
    print("[2/4] Init Router & components...")
    from modules.router.sh_da_router import SHDARouter
    from modules.router.backfill import BackfillConfig, StrictBackfillController
    from modules.vlm_expert.constrained_prompter import ConstrainedPrompter
    from modules.router.domain_knowledge import DomainKnowledgeEngine

    router = SHDARouter.from_yaml(args.config)
    backfill_controller = StrictBackfillController(BackfillConfig())
    prompter = ConstrainedPrompter()
    domain_engine = DomainKnowledgeEngine(args.geo_dict)
    agent_b_callable = build_agent_b_callable(config)

    # 读取测试集
    print("[3/4] Load test set...")
    samples = []
    with open(args.test_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"  Total samples: {len(samples)}")

    # Agent A 推理
    print("[4/4] Agent A inference...")
    all_results = infer_all_samples(samples, recognizer, domain_engine, None, args.image_root)
    
    # 限制样本数
    if args.n_samples < len(all_results):
        all_results = all_results[:args.n_samples]
        print(f"  Limited to first {args.n_samples} samples")
    
    print(f"  Valid: {len(all_results)}")

    # 输出 CSV
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"quick_test_{args.n_samples}.csv"
    fieldnames = [
        "Strategy",
        "Target_Budget",
        "Actual_Call_Rate",
        "Overall_CER",
        "AER",
        "CVR",
        "N_valid",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Baseline: AgentA_Only
        print("\n" + "=" * 60)
        print("Baseline: AgentA_Only")
        print("=" * 60)
        row = run_pipeline(
            "AgentA_Only", 0.0, all_results,
            router, backfill_controller, prompter, agent_b_callable,
        )
        writer.writerow({k: row.get(k, "") for k in fieldnames})
        csvfile.flush()
        print(f"  CER={row['Overall_CER']:.4%}")

        # 测试预算点：B=0.05, 0.20
        budgets = [0.05, 0.20]
        strategies = ["Random", "ConfOnly", "SH-DA++"]

        for B in budgets:
            print("\n" + "=" * 60)
            print(f"Budget B={B:.2f}")
            print("=" * 60)
            
            # 顺序执行 3 个策略（避免并发问题）
            for strategy in strategies:
                print(f"\n[{strategy}] Running...")
                try:
                    row = run_pipeline(
                        strategy, B, all_results,
                        router, backfill_controller, prompter, agent_b_callable,
                    )
                    writer.writerow({k: row.get(k, "") for k in fieldnames})
                    csvfile.flush()
                    print(
                        f"  [{strategy:10s}] "
                        f"CER={row['Overall_CER']:.4%}  "
                        f"AER={row['AER']:.2%}  "
                        f"CVR={row['CVR']:.2%}  "
                        f"ActualRate={row['Actual_Call_Rate']:.2%}"
                    )
                except Exception as e:
                    print(f"  [{strategy:10s}] ERROR: {e}")

    print("\n" + "=" * 60)
    print(f"Test completed! Results saved to:")
    print(f"  {csv_path}")
    print("=" * 60)

    # 打印 CSV 内容
    print("\nCSV Content:")
    print("-" * 60)
    with open(csv_path, "r", encoding="utf-8") as f:
        print(f.read())


if __name__ == "__main__":
    main()
