#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v5.1 Phase 1: 特征工程 2.0

特征矩阵 X (5维):
  [Mean_Confidence, Min_Confidence, b_edge, drop, r_d]

变更说明：
  - 彻底删除 v_edge 及 v_edge * b_edge
  - 新增 Mean_Confidence、Min_Confidence
  - 激活 r_d（基于多领域词典匹配）
  - 按 train/val/test 三个 split 分别提取，严格隔离
  - 支持 --image_root 参数显式指定图像路径基准目录
"""

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import Levenshtein
from modules.paddle_engine.predict_rec_modified import TextRecognizerWithLogits
from modules.router.domain_knowledge import DomainKnowledgeEngine


def generate_deletion_label(T_A: str, T_GT: str, K: int = 2) -> int:
    """
    生成边界漏字标签（Levenshtein 对齐，唯一合法的标签来源）
    """
    if not T_A or not T_GT:
        return 0
    ops = Levenshtein.editops(T_A, T_GT)
    for op_type, pos_A, pos_GT in ops:
        if op_type == "delete":
            if pos_GT < K or pos_GT >= len(T_GT) - K:
                return 1
    return 0


def extract_features_from_output(
    output: dict,
    conf: float,
    T_A: str,
    domain_engine: DomainKnowledgeEngine,
    domain: str = 'geology',
) -> np.ndarray:
    """
    从 recognizer 输出提取 5 维特征向量
    特征顺序：[Mean_Confidence, Min_Confidence, b_edge, drop, r_d]
    """
    # --- Mean_Confidence & Min_Confidence ---
    top2_info_list = output.get("top2_info", [])
    top2_info = top2_info_list[0] if top2_info_list else None

    char_confs = []
    if top2_info and isinstance(top2_info, dict):
        char_confs = top2_info.get("char_confs", []) or top2_info.get("top1_probs", [])

    if char_confs and len(char_confs) > 0:
        mean_conf = float(np.mean(char_confs))
        min_conf = float(np.min(char_confs))
    else:
        mean_conf = float(conf)
        min_conf = float(conf)

    # --- b_edge & drop ---
    boundary_stats_list = output.get("boundary_stats", [])
    boundary_stats = boundary_stats_list[0] if boundary_stats_list else None

    if boundary_stats and boundary_stats.get("valid", False):
        blank_mean_L = float(boundary_stats.get("blank_mean_L", 0.0))
        blank_mean_R = float(boundary_stats.get("blank_mean_R", 0.0))
        blank_peak_L = float(boundary_stats.get("blank_peak_L", 0.0))
        blank_peak_R = float(boundary_stats.get("blank_peak_R", 0.0))
        b_edge = float(max(0.6 * blank_mean_L + 0.4 * blank_peak_L,
                           0.6 * blank_mean_R + 0.4 * blank_peak_R))
        drop = float(np.clip(abs(blank_mean_L - blank_mean_R), 0.0, 1.0))
    else:
        b_edge = float(np.clip(1.0 - conf, 0.0, 1.0))
        drop = 0.0

    # --- r_d: 多领域语义风险分 ---
    r_d = domain_engine.compute_r_d(T_A, domain=domain)

    return np.array([mean_conf, min_conf, b_edge, drop, r_d], dtype=np.float32)


def process_split(
    jsonl_path: str,
    output_prefix: str,
    recognizer,
    domain_engine: DomainKnowledgeEngine,
    K: int = 2,
    image_root: Optional[str] = None,
) -> None:
    """
    处理单个 split（train/val/test），提取特征并保存

    Args:
        jsonl_path:    split JSONL 文件路径
        output_prefix: 输出文件前缀
        recognizer:    TextRecognizerWithLogits 实例
        domain_engine: DomainKnowledgeEngine 实例
        K:             边界窗口大小
        image_root:    图像路径基准目录（若为 None 则使用 jsonl 所在目录）
    """
    import cv2

    jsonl_path = Path(jsonl_path)
    # 图像路径基准目录：优先使用外部传入的 image_root，否则退回 jsonl 所在目录
    if image_root:
        data_root = Path(image_root).resolve()
    else:
        data_root = jsonl_path.resolve().parent

    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    split_name = jsonl_path.stem
    print(f"\n[{split_name}] 共 {len(samples)} 条样本，图像基准目录: {data_root}")

    features_list = []
    labels_list = []
    metadata_list = []
    skip_count = 0

    for sample in tqdm(samples, desc=f"{split_name}"):
        image_path = sample.get("image") or sample.get("image_path")
        T_GT = sample.get("gt_text") or sample.get("gt") or sample.get("text") or sample.get("label", "")

        if not image_path or not T_GT:
            skip_count += 1
            continue

        img_path = Path(image_path)
        if not img_path.is_absolute():
            rel_path = Path(image_path)
            if rel_path.parts[:2] == ('dataset', 'images'):
                rel_path = Path(*rel_path.parts[2:])
            img_path = data_root / rel_path

        img = cv2.imread(str(img_path))
        if img is None:
            skip_count += 1
            continue

        try:
            output = recognizer([img])
            if not output or not output.get("results"):
                skip_count += 1
                continue
            T_A, conf = output["results"][0]
        except Exception:
            skip_count += 1
            continue

        features = extract_features_from_output(output, conf, T_A, domain_engine, domain=sample.get("domain", "geology"))
        y_deletion = generate_deletion_label(T_A, T_GT, K=K)

        features_list.append(features)
        labels_list.append(y_deletion)
        metadata_list.append({
            "sample_id": sample.get("sample_id", ""),
            "domain": sample.get("domain", "geology"),
            "split": sample.get("split", split_name),
            "image": str(image_path),
            "T_A": T_A,
            "T_GT": T_GT,
            "y_deletion": y_deletion,
            "Mean_Confidence": float(features[0]),
            "Min_Confidence": float(features[1]),
            "b_edge": float(features[2]),
            "drop": float(features[3]),
            "r_d": float(features[4]),
        })

    if not features_list:
        print(f"  [警告] 没有有效样本！跳过: {skip_count}")
        return

    X = np.array(features_list, dtype=np.float32)
    Y = np.array(labels_list, dtype=np.int32)

    np.save(f"{output_prefix}.npy", X)
    np.save(str(output_prefix).replace("features_", "labels_") + ".npy", Y)
    with open(str(output_prefix).replace("features_", "metadata_") + ".jsonl",
              "w", encoding="utf-8") as f:
        for m in metadata_list:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print(f"  有效: {len(features_list)} / {len(samples)} (跳过: {skip_count})")
    print(f"  正样本比例: {Y.mean():.2%}")
    print(f"  features_{split_name}.npy: {X.shape}")


def main():
    from tools.infer.utility import init_args

    parser = init_args()
    parser.add_argument("--data_dir", type=str, default="data/l2w1data",
                        help="包含 train/val/test.jsonl 的目录")
    parser.add_argument("--output_dir", type=str, default="results/stage2_v51",
                        help="特征输出目录")
    parser.add_argument("--image_root", type=str, default="data/l2w1data/images",
                        help="图像路径基准目录（JSONL 中图像路径相对于此目录）")
    parser.add_argument("--geo_dict", type=str, default="data/dicts/Geology.txt",
                        help="地质词典路径")
    parser.add_argument("--finance_dict", type=str, default="data/dicts/Finance.txt",
                        help="金融词典路径")
    parser.add_argument("--medicine_dict", type=str, default="data/dicts/Medicine.txt",
                        help="医学词典路径")
    parser.add_argument("--K", type=int, default=2, help="边界窗口大小")
    parser.add_argument("--use_det", action="store_true")
    parser.add_argument("--det_model_dir", type=str, default="")
    parser.add_argument("--splits", type=str, default="train,val,test",
                        help="要处理的 split，逗号分隔")

    args = parser.parse_args()

    if args.rec_model_dir is None:
        args.rec_model_dir = "./models/agent_a_ppocr/PP-OCRv5_server_rec_infer"

    print("[1/3] 初始化 Agent A...")
    recognizer = TextRecognizerWithLogits(args)

    print("[2/3] 初始化语义引擎...")
    domain_engine = DomainKnowledgeEngine({
        'geology': args.geo_dict,
        'finance': args.finance_dict,
        'medicine': args.medicine_dict,
    })

    print("[3/3] 按 split 提取特征...")
    splits = [s.strip() for s in args.splits.split(",")]
    for split_name in splits:
        jsonl_path = Path(args.data_dir) / f"{split_name}.jsonl"
        if not jsonl_path.exists():
            print(f"[跳过] {jsonl_path} 不存在")
            continue
        output_prefix = Path(args.output_dir) / f"features_{split_name}"
        process_split(
            jsonl_path=str(jsonl_path),
            output_prefix=str(output_prefix),
            recognizer=recognizer,
            domain_engine=domain_engine,
            K=args.K,
            image_root=args.image_root,
        )

    print(f"\n✓ 全部 split 特征提取完成，保存在: {args.output_dir}")


if __name__ == "__main__":
    main()
