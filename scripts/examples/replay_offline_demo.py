#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

PROMPT_VERSION = "stage9_synthetic_demo"
RUN_ID = "stage9_replay_offline_demo"


def build_synthetic_full_call_cache() -> list[dict[str, Any]]:
    return [
        {
            "sample_id": "demo-boundary",
            "image_path": "synthetic/demo-boundary.png",
            "gt": "地层边界",
            "T_A": "层边界",
            "final_text_if_upgraded": "地层边界",
            "vlm_raw_output": "地层边界",
            "latency_ms": 120.0,
            "token_usage": 16,
            "error_type": "boundary",
            "mean_conf": 0.42,
            "min_conf": 0.20,
            "drop": 0.32,
            "conf": 0.41,
            "r_d": 0.35,
        },
        {
            "sample_id": "demo-substitution",
            "image_path": "synthetic/demo-substitution.png",
            "gt": "ABCD",
            "T_A": "ABXD",
            "final_text_if_upgraded": "ABCD",
            "vlm_raw_output": "ABCD",
            "latency_ms": 98.5,
            "token_usage": 12,
            "error_type": "substitution",
            "mean_conf": 0.68,
            "min_conf": 0.50,
            "drop": 0.14,
            "conf": 0.67,
            "r_d": 0.05,
        },
        {
            "sample_id": "demo-correct",
            "image_path": "synthetic/demo-correct.png",
            "gt": "HELLO",
            "T_A": "HELLO",
            "final_text_if_upgraded": "HELLO",
            "vlm_raw_output": "HELLO",
            "latency_ms": 64.0,
            "token_usage": 9,
            "error_type": "none",
            "mean_conf": 0.96,
            "min_conf": 0.90,
            "drop": 0.02,
            "conf": 0.95,
            "r_d": 0.01,
        },
        {
            "sample_id": "demo-low-risk",
            "image_path": "synthetic/demo-low-risk.png",
            "gt": "CASH",
            "T_A": "CASH",
            "final_text_if_upgraded": "CASHX",
            "vlm_raw_output": "CASHX",
            "latency_ms": 73.0,
            "token_usage": 10,
            "error_type": "overcorrection",
            "mean_conf": 0.85,
            "min_conf": 0.55,
            "drop": 0.12,
            "conf": 0.84,
            "r_d": 0.40,
        },
    ]


def build_replay_rows(cache_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from l2w1.ocr.cache_only import CacheOnlyOCREngine
    from l2w1.ocr.types import OCRRequest
    from l2w1.vlm.cache_only import CacheOnlyVLMExpert
    from l2w1.vlm.types import VLMRequest

    ocr_engine = CacheOnlyOCREngine(cache_rows)
    vlm_expert = CacheOnlyVLMExpert(cache_rows)
    replay_rows: list[dict[str, Any]] = []

    for cache_row in cache_rows:
        sample_id = str(cache_row["sample_id"])
        image_path = str(cache_row["image_path"])
        ocr_result = ocr_engine.recognize(OCRRequest(image_path=image_path, sample_id=sample_id))
        vlm_response = vlm_expert.query(
            VLMRequest(image_path=image_path, sample_id=sample_id, t_a=ocr_result.text)
        )

        replay_rows.append(
            {
                "sample_id": sample_id,
                "image_path": image_path,
                "source_image_id": sample_id,
                "domain": "synthetic",
                "split": "demo",
                "gt": str(cache_row["gt"]),
                "ocr_text": ocr_result.text,
                "final_text_if_upgraded": vlm_response.corrected_text,
                "final_text": vlm_response.corrected_text,
                "vlm_raw_output": vlm_response.raw_output,
                "latency_ms": vlm_response.latency_ms,
                "token_usage": vlm_response.token_usage,
                "error_type": vlm_response.error_type,
                "has_professional_terms": False,
                "professional_terms": [],
                "domain_risk_score": cache_row["r_d"],
                "mean_conf": ocr_result.mean_conf,
                "min_conf": ocr_result.min_conf,
                "drop": ocr_result.drop,
                "conf": ocr_result.conf,
                "r_d": ocr_result.r_d,
                "is_correct_ocr": ocr_result.text == str(cache_row["gt"]),
                "edit_distance_ocr": 0,
                "vlm_model": "cache_only",
                "prompt_version": PROMPT_VERSION,
                "run_id": RUN_ID,
            }
        )

    return replay_rows


def run_demo(*, strategy: str = "WUR", budget: float = 0.5) -> dict[str, Any]:
    from l2w1.replay.offline import replay_offline
    from l2w1.replay.scoring import router_score

    replay_rows = build_replay_rows(build_synthetic_full_call_cache())
    score_map = [router_score(strategy, row) for row in replay_rows]
    return replay_offline(
        strategy,
        budget,
        replay_rows,
        score_map,
        prompt_version=PROMPT_VERSION,
        run_id=RUN_ID,
    )


def main() -> int:
    result = run_demo()
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
