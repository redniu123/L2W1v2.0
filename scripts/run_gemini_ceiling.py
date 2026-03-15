#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SH-DA++ v5.1: Gemini 100% 上限实验

实验设计：
  Mode 1 (correction): Agent A 先识别，Gemini 100% 纠错  -> 测纠错上限
  Mode 2 (ocr):        完全跳过 Agent A，Gemini 直接看图识字 -> 测纯识别上限

输出: results/stage2_v51/gemini_ceiling.csv
用法:
  python scripts/run_gemini_ceiling.py --mode correction
  python scripts/run_gemini_ceiling.py --mode ocr
  python scripts/run_gemini_ceiling.py --mode both   # 两个都跑
"""

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import Levenshtein
import numpy as np
from tqdm import tqdm
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# Gemini 纠错调用（Mode 1）
# ============================================================
def call_gemini_correction(agent, r: dict) -> str:
    """用 Gemini 对 Agent A 的结果做纠错"""
    T_A = r["T_A"]
    min_conf_idx = r.get("min_conf_idx", -1) or -1
    suspicious_char = T_A[min_conf_idx] if 0 <= min_conf_idx < len(T_A) else ""
    manifest = {
        "ocr_text": T_A,
        "suspicious_index": min_conf_idx,
        "suspicious_char": suspicious_char,
        "risk_level": "medium",
    }
    try:
        result = agent.process_hard_sample(r["img_path"], manifest)
        return result["corrected_text"]
    except Exception as e:
        print(f"  [Gemini correction] error: {e}")
        return T_A


# ============================================================
# Gemini 纯识别调用（Mode 2）
# ============================================================
OCR_PROMPT = (
    "你是一个专业的 OCR 识别专家，擅长识别历史档案、地质勘探文档中的手写与印刷文字。\n"
    "请仔细观察图像中的单行文字，直接输出你识别到的完整文本内容。\n"
    "要求：\n"
    "1. 只输出识别的文字内容，不要任何解释、标点说明或多余字符。\n"
    "2. 如果是地质专业术语，请尽量准确识别。\n"
    "3. 直接输出文字，不要引号或括号包裹。"
)


def call_gemini_ocr(agent, r: dict) -> str:
    """用 Gemini 直接看图识字（不依赖 Agent A）"""
    try:
        # 直接调用底层 _call_api，使用纯 OCR 提示词
        image_base64 = agent._encode_image(r["img_path"])
        for attempt in range(agent.config.max_retries):
            api_key = agent.config.key_manager.get_next_key()
            result = agent._call_api(OCR_PROMPT, image_base64, api_key)
            if result:
                # 只取第一行，去除多余字符
                text = result.strip().split("\n")[0].strip().strip('"\' \u201c\u201d')
                return text if text else r.get("T_A", "")
        return r.get("T_A", "")
    except Exception as e:
        print(f"  [Gemini OCR] error: {e}")
        return r.get("T_A", "")


# ============================================================
# 并发批量调用
# ============================================================
def batch_call_gemini(agent, all_results, call_fn, desc, max_workers=5, delay=0.2):
    """并发调用 Gemini，返回 {idx: output_text} 字典"""
    outputs = {}

    def worker(idx, r):
        time.sleep(delay)
        return idx, call_fn(agent, r)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(worker, i, r): i for i, r in enumerate(all_results)}
        for future in tqdm(as_completed(futures), total=len(all_results), desc=desc):
            try:
                idx, text = future.result()
                outputs[idx] = text
            except Exception as e:
                idx = futures[future]
                outputs[idx] = all_results[idx].get("T_A", "")
    return outputs


# ============================================================
# 指标计算
# ============================================================
def compute_metrics(all_results, outputs, mode):
    """计算 CER、字符级准确率等指标"""
    cer_num = 0
    gt_len_total = 0
    n_changed = 0
    n_improved = 0  # 改了且变好
    n_worsened = 0  # 改了且变差
    n_same = 0      # 没改

    for i, r in enumerate(all_results):
        T_GT = r["T_GT"]
        T_A = r["T_A"]
        T_out = outputs.get(i, T_A)

        if not T_GT:
            continue

        ed_out = Levenshtein.distance(T_out, T_GT)
        ed_a = Levenshtein.distance(T_A, T_GT)

        cer_num += ed_out
        gt_len_total += len(T_GT)

        if T_out != T_A:
            n_changed += 1
            if ed_out < ed_a:
                n_improved += 1
            elif ed_out > ed_a:
                n_worsened += 1
        else:
            n_same += 1

    N = len(all_results)
    overall_cer = cer_num / gt_len_total if gt_len_total else 0.0

    return {
        "Mode": mode,
        "N": N,
        "Overall_CER": round(overall_cer, 6),
        "Change_Rate": round(n_changed / N, 4) if N else 0,
        "Improve_Rate": round(n_improved / N, 4) if N else 0,
        "Worsen_Rate": round(n_worsened / N, 4) if N else 0,
        "Same_Rate": round(n_same / N, 4) if N else 0,
    }


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Gemini 100% 上限实验")
    parser.add_argument("--mode", choices=["correction", "ocr", "both"], default="both")
    parser.add_argument("--config", default="configs/router_config.yaml")
    parser.add_argument("--test_jsonl", default="data/raw/hctr_riskbench/test.jsonl")
    parser.add_argument("--image_root", default="data/geo")
    parser.add_argument("--rec_model_dir", default="./models/agent_a_ppocr/PP-OCRv5_server_rec_infer")
    parser.add_argument("--rec_char_dict_path", default="ppocr/utils/ppocrv5_dict.txt")
    parser.add_argument("--output_dir", default="results/stage2_v51")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--use_cache", action="store_true", default=False)
    parser.add_argument("--use_gpu", action="store_true", default=False)
    parser.add_argument("--max_workers", type=int, default=5)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "agent_a_cache.json"

    # ---- 初始化 Gemini ----
    print("[1] Init Gemini Agent B...")
    from modules.vlm_expert.gemini_expert import GeminiAgentB, GeminiConfig
    agent = GeminiAgentB(config=GeminiConfig())
    print(f"  Model: {agent.config.model_name}  Keys: {agent.config.key_manager.get_key_count()}")

    # ---- 加载 Test 集 ----
    print("[2] Load test set...")
    samples = []
    with open(args.test_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    print(f"  Test samples: {len(samples)}")

    # ---- Agent A 推理 / 加载缓存 ----
    need_agent_a = (args.mode in ("correction", "both"))

    if need_agent_a:
        if args.use_cache and cache_path.exists():
            print("[3] Agent A cache HIT, loading...")
            with open(cache_path, "r", encoding="utf-8") as f:
                all_results = json.load(f)
            print(f"  Loaded {len(all_results)} records")
        else:
            print("[3] Agent A full inference...")
            import argparse as _ap
            rec_args = _ap.Namespace(
                rec_model_dir=args.rec_model_dir,
                rec_char_dict_path=args.rec_char_dict_path,
                rec_image_shape="3, 48, 320", rec_batch_num=6,
                rec_algorithm="SVTR_LCNet", use_space_char=True,
                use_gpu=args.use_gpu, use_xpu=False, use_npu=False, use_mlu=False,
                use_metax_gpu=False, use_gcu=False, ir_optim=True,
                use_tensorrt=False, min_subgraph_size=15, precision="fp32",
                gpu_mem=500, gpu_id=0, enable_mkldnn=None, cpu_threads=10,
                warmup=False, benchmark=False, save_log_path="./log_output/",
                show_log=False, use_onnx=False, max_batch_size=10,
                return_word_box=False, drop_score=0.5, max_text_length=25,
                rec_image_inverse=True, use_det=False, det_model_dir="",
            )
            from modules.paddle_engine.predict_rec_modified import TextRecognizerWithLogits
            from modules.router.domain_knowledge import DomainKnowledgeEngine
            import cv2
            recognizer = TextRecognizerWithLogits(rec_args)
            domain_engine = DomainKnowledgeEngine("data/dicts/Geology.txt")
            all_results = []
            for sample in tqdm(samples, desc="Agent A"):
                image_path = sample.get("image") or sample.get("image_path", "")
                T_GT = sample.get("gt_text") or sample.get("text") or sample.get("label", "")
                if not image_path or not T_GT:
                    continue
                img_path = Path(image_path)
                if not img_path.is_absolute():
                    img_path = Path(args.image_root).resolve() / img_path
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                try:
                    output = recognizer([img])
                    if not output or not output.get("results"):
                        continue
                    T_A, conf = output["results"][0]
                except Exception:
                    continue
                top2_info = (output.get("top2_info") or [{}])[0]
                top1_probs = (top2_info or {}).get("top1_probs") or []
                min_conf_idx = int(np.argmin(top1_probs)) if top1_probs else None
                all_results.append({
                    "image_path": str(image_path),
                    "img_path": str(img_path),
                    "T_A": T_A, "T_GT": T_GT,
                    "min_conf_idx": min_conf_idx,
                })
            # 保存缓存
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False)
            print(f"  Cache saved: {cache_path}")
    else:
        # Mode ocr only：不需要 Agent A，直接从 JSONL 构建样本列表
        print("[3] OCR mode: skipping Agent A inference")
        all_results = []
        for sample in samples:
            image_path = sample.get("image") or sample.get("image_path", "")
            T_GT = sample.get("gt_text") or sample.get("text") or sample.get("label", "")
            if not image_path or not T_GT:
                continue
            img_path = Path(image_path)
            if not img_path.is_absolute():
                img_path = Path(args.image_root).resolve() / img_path
            all_results.append({
                "image_path": str(image_path),
                "img_path": str(img_path),
                "T_A": "", "T_GT": T_GT,
                "min_conf_idx": None,
            })

    if args.n_samples and args.n_samples < len(all_results):
        all_results = all_results[:args.n_samples]
        print(f"  Limited to {args.n_samples} samples")
    print(f"  Valid samples: {len(all_results)}")

    # ---- 结果收集 ----
    csv_path = output_dir / "gemini_ceiling.csv"
    fieldnames = ["Mode", "N", "Overall_CER", "Change_Rate",
                  "Improve_Rate", "Worsen_Rate", "Same_Rate"]

    rows = []

    # ---- Mode 1: Correction（100% 纠错）----
    if args.mode in ("correction", "both"):
        print("\n[4a] Mode 1: Gemini 100% Correction (AgentA + Gemini)...")
        # 先计算 AgentA_Only baseline
        cer_a_num = sum(Levenshtein.distance(r["T_A"], r["T_GT"]) for r in all_results)
        gt_len_total = sum(len(r["T_GT"]) for r in all_results)
        cer_a = cer_a_num / gt_len_total if gt_len_total else 0.0
        print(f"  AgentA_Only CER (baseline): {cer_a:.4%}")

        outputs_corr = batch_call_gemini(
            agent, all_results, call_gemini_correction,
            desc="Gemini Correction 100%",
            max_workers=args.max_workers,
        )
        metrics_corr = compute_metrics(all_results, outputs_corr, "Gemini_Correction_100pct")
        metrics_corr["AgentA_CER"] = round(cer_a, 6)
        rows.append(metrics_corr)
        print(f"  [Correction] CER={metrics_corr['Overall_CER']:.4%}  "
              f"Improve={metrics_corr['Improve_Rate']:.2%}  "
              f"Worsen={metrics_corr['Worsen_Rate']:.2%}")
        print(f"  vs AgentA_Only: {cer_a:.4%} -> {metrics_corr['Overall_CER']:.4%}  "
              f"({'better' if metrics_corr['Overall_CER'] < cer_a else 'worse'})")

    # ---- Mode 2: Pure OCR ----
    if args.mode in ("ocr", "both"):
        print("\n[4b] Mode 2: Gemini 100% Pure OCR (no AgentA)...")
        # OCR mode 需要原始样本列表（img_path + T_GT）
        if args.mode == "both":
            # 复用 all_results（已有 img_path 和 T_GT）
            ocr_results = all_results
        else:
            ocr_results = all_results

        outputs_ocr = batch_call_gemini(
            agent, ocr_results, call_gemini_ocr,
            desc="Gemini Pure OCR 100%",
            max_workers=args.max_workers,
        )
        metrics_ocr = compute_metrics(ocr_results, outputs_ocr, "Gemini_PureOCR_100pct")
        metrics_ocr["AgentA_CER"] = "N/A"
        rows.append(metrics_ocr)
        print(f"  [Pure OCR] CER={metrics_ocr['Overall_CER']:.4%}")

    # ---- 写入 CSV ----
    all_fieldnames = ["Mode", "N", "Overall_CER", "AgentA_CER",
                      "Change_Rate", "Improve_Rate", "Worsen_Rate", "Same_Rate"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in all_fieldnames})

    print(f"\n[Done] Results saved to {csv_path}")
    print("\n=== Gemini Ceiling Summary ===")
    for row in rows:
        print(f"  {row['Mode']:35s}  CER={row['Overall_CER']:.4%}")


if __name__ == "__main__":
    main()
 