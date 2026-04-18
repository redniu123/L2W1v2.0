#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert old Gemini cache into Main C `V3_full_call_cache.jsonl`."""
import argparse
import json
from pathlib import Path


DEFAULT_MODEL_NAME = "Gemini 3 Flash Preview"
DEFAULT_PROMPT_VERSION = "prompt_v1.1"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def convert_row(row: dict, run_id: str, model_name: str, prompt_version: str) -> dict:
    return {
        "sample_id": row.get("sample_id", ""),
        "image_path": row.get("image_path", ""),
        "source_image_id": row.get("source_image_id", ""),
        "domain": row.get("domain", "geology"),
        "split": row.get("split", "test"),
        "gt": row.get("gt", ""),
        "ocr_text": row.get("ocr_text", ""),
        "final_text_if_upgraded": row.get("final_text_if_upgraded") or row.get("final_text") or row.get("ocr_text", ""),
        "vlm_raw_output": row.get("vlm_raw_output") or row.get("final_text_if_upgraded") or row.get("final_text") or row.get("ocr_text", ""),
        "latency_ms": row.get("latency_ms"),
        "token_usage": row.get("token_usage"),
        "error_type": row.get("error_type", "none"),
        "has_professional_terms": row.get("has_professional_terms", False),
        "professional_terms": row.get("professional_terms", []),
        "is_correct_ocr": row.get("is_correct_ocr", False),
        "edit_distance_ocr": row.get("edit_distance_ocr", 0),
        "vlm_model": model_name,
        "prompt_version": row.get("prompt_version") or prompt_version,
        "run_id": run_id,
    }


def main():
    p = argparse.ArgumentParser(description="Convert old Gemini results to Main C V3 cache format")
    p.add_argument("--input_jsonl", required=True, help="Old Gemini jsonl, e.g. full_budget_results_M5.jsonl")
    p.add_argument("--output_jsonl", required=True, help="Target Main C V3_full_call_cache.jsonl path")
    p.add_argument("--run_id", required=True, help="Main C run_id to stamp into converted rows")
    p.add_argument("--model_name", default=DEFAULT_MODEL_NAME)
    p.add_argument("--prompt_version", default=DEFAULT_PROMPT_VERSION)
    args = p.parse_args()

    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [convert_row(r, args.run_id, args.model_name, args.prompt_version) for r in read_jsonl(input_path)]

    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"converted_rows={len(rows)}")
    print(output_path)


if __name__ == "__main__":
    main()
