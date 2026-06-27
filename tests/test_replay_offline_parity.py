from __future__ import annotations

import pytest
from scripts.experiments.efficiency_frontier import (
    summarize_extended_metrics,
    summarize_latency_and_token_usage,
)
from scripts.paper1 import mainA_runner as old_main

from l2w1.replay.offline import replay_offline, select_offline_upgrades
from l2w1.replay.scoring import normalize_format, router_score


def _offline_full_rows() -> list[dict]:
    return [
        {
            "sample_id": "s1",
            "image_path": "synthetic/s1.png",
            "source_image_id": "img-1",
            "domain": "geology",
            "split": "test",
            "gt": "地层(A)",
            "ocr_text": "地层（A）",
            "final_text_if_upgraded": "地层(A)",
            "final_text": "地层(A)",
            "vlm_raw_output": "地层(A)",
            "latency_ms": 101.0,
            "token_usage": 11,
            "error_type": "none",
            "has_professional_terms": True,
            "professional_terms": ["地层"],
            "domain_risk_score": 0.22,
            "mean_conf": 0.82,
            "min_conf": 0.30,
            "drop": 0.25,
            "conf": 0.78,
            "r_d": 0.22,
            "is_correct_ocr": False,
            "edit_distance_ocr": 0,
            "vlm_model": "Synthetic Agent B",
            "prompt_version": "cache_prompt",
            "run_id": "cache_run",
        },
        {
            "sample_id": "s2",
            "image_path": "synthetic/s2.png",
            "source_image_id": "img-2",
            "domain": "finance",
            "split": "test",
            "gt": "ABCD",
            "ocr_text": "ABXD",
            "final_text_if_upgraded": "ABCD",
            "final_text": "ABCD",
            "vlm_raw_output": "ABCD",
            "latency_ms": 205.5,
            "token_usage": 17,
            "error_type": "none",
            "has_professional_terms": False,
            "professional_terms": [],
            "domain_risk_score": 0.05,
            "mean_conf": 0.58,
            "min_conf": 0.44,
            "drop": 0.18,
            "conf": 0.55,
            "r_d": 0.05,
            "is_correct_ocr": False,
            "edit_distance_ocr": 1,
            "vlm_model": "Synthetic Agent B",
            "prompt_version": "cache_prompt",
            "run_id": "cache_run",
        },
        {
            "sample_id": "s3",
            "image_path": "synthetic/s3.png",
            "source_image_id": "img-3",
            "domain": "medicine",
            "split": "test",
            "gt": "HELLO",
            "ocr_text": "HELLO",
            "final_text_if_upgraded": "HELLO",
            "final_text": "HELLO",
            "vlm_raw_output": "HELLO",
            "latency_ms": 55.0,
            "token_usage": 9,
            "error_type": "none",
            "has_professional_terms": False,
            "professional_terms": [],
            "domain_risk_score": 0.01,
            "mean_conf": 0.96,
            "min_conf": 0.91,
            "drop": 0.02,
            "conf": 0.94,
            "r_d": 0.01,
            "is_correct_ocr": True,
            "edit_distance_ocr": 0,
            "vlm_model": "Synthetic Agent B",
            "prompt_version": "cache_prompt",
            "run_id": "cache_run",
        },
        {
            "sample_id": "s4",
            "image_path": "synthetic/s4.png",
            "source_image_id": "img-4",
            "domain": "geology",
            "split": "test",
            "gt": "边界XYZ",
            "ocr_text": "界XYZ",
            "final_text_if_upgraded": "边界XYZ",
            "final_text": "边界XYZ",
            "vlm_raw_output": "边界XYZ",
            "latency_ms": 310.25,
            "token_usage": 25,
            "error_type": "boundary",
            "has_professional_terms": True,
            "professional_terms": ["边界"],
            "domain_risk_score": 0.40,
            "mean_conf": 0.40,
            "min_conf": 0.20,
            "drop": 0.35,
            "conf": 0.42,
            "r_d": 0.40,
            "is_correct_ocr": False,
            "edit_distance_ocr": 1,
            "vlm_model": "Synthetic Agent B",
            "prompt_version": "cache_prompt",
            "run_id": "cache_run",
        },
        {
            "sample_id": "s5",
            "image_path": "synthetic/s5.png",
            "source_image_id": "img-5",
            "domain": "finance",
            "split": "test",
            "gt": "CASH",
            "ocr_text": "CASH",
            "final_text_if_upgraded": "CASHX",
            "final_text": "CASHX",
            "vlm_raw_output": "CASHX",
            "latency_ms": 88.0,
            "token_usage": 13,
            "error_type": "overcorrection",
            "has_professional_terms": False,
            "professional_terms": [],
            "domain_risk_score": 0.31,
            "mean_conf": 0.88,
            "min_conf": 0.62,
            "drop": 0.08,
            "conf": 0.87,
            "r_d": 0.31,
            "is_correct_ocr": True,
            "edit_distance_ocr": 0,
            "vlm_model": "Synthetic Agent B",
            "prompt_version": "cache_prompt",
            "run_id": "cache_run",
        },
        {
            "sample_id": "s6",
            "image_path": "synthetic/s6.png",
            "source_image_id": "img-6",
            "domain": "medicine",
            "split": "test",
            "gt": "QWER",
            "ocr_text": "QWXR",
            "final_text_if_upgraded": "QWXR",
            "final_text": "QWXR",
            "vlm_raw_output": "QWXR",
            "latency_ms": 140.0,
            "token_usage": 15,
            "error_type": "none",
            "has_professional_terms": False,
            "professional_terms": [],
            "domain_risk_score": 0.12,
            "mean_conf": 0.70,
            "min_conf": 0.52,
            "drop": 0.12,
            "conf": 0.69,
            "r_d": 0.12,
            "is_correct_ocr": False,
            "edit_distance_ocr": 1,
            "vlm_model": "Synthetic Agent B",
            "prompt_version": "cache_prompt",
            "run_id": "cache_run",
        },
    ]


def test_router_score_and_format_normalization_match_main_a() -> None:
    rows = _offline_full_rows()
    for row in rows:
        for strategy in ["GCR", "WUR", "DGCR", "DWUR"]:
            assert router_score(strategy, row, eta=0.7) == old_main.score(strategy, row, eta=0.7)

    text = "（A）【B】｛C｝，：；！？。"
    assert normalize_format(text) == old_main.norm(text)


def test_router_score_rejects_unsupported_strategy() -> None:
    with pytest.raises(ValueError, match="Unsupported strategy"):
        router_score("BAD", _offline_full_rows()[0])


def test_select_offline_upgrades_uses_descending_scores_and_one_based_rank() -> None:
    upgrades, rank_map = select_offline_upgrades([0.1, 0.9, 0.5, 0.2], 0.5)

    assert upgrades == {1, 2}
    assert rank_map == {1: 1, 2: 2, 3: 3, 0: 4}


@pytest.mark.parametrize("strategy", ["GCR", "WUR", "DGCR", "DWUR"])
@pytest.mark.parametrize("budget", [0.0, 0.33, 0.5, 1.0])
def test_replay_offline_matches_main_a_replay(strategy: str, budget: float) -> None:
    full_rows = _offline_full_rows()
    score_map = [router_score(strategy, row, eta=0.7) for row in full_rows]

    old_result = old_main.replay(
        strategy,
        budget,
        full_rows,
        score_map,
        "prompt_v_test",
        "run_parity",
    )
    new_result = replay_offline(
        strategy,
        budget,
        full_rows,
        score_map,
        prompt_version="prompt_v_test",
        run_id="run_parity",
        extended_metrics_fn=summarize_extended_metrics,
        usage_metrics_fn=summarize_latency_and_token_usage,
    )

    assert new_result["summary"] == old_result["summary"]
    assert new_result["per_sample"] == old_result["per_sample"]


def test_replay_offline_sets_injected_summary_metrics_to_none_when_absent() -> None:
    full_rows = _offline_full_rows()
    score_map = [router_score("GCR", row) for row in full_rows]

    result = replay_offline(
        "GCR",
        0.5,
        full_rows,
        score_map,
        prompt_version="prompt_v_test",
    )

    assert result["summary"]["BoundaryDeletionRecallAtB"] is None
    assert result["summary"]["SubstitutionCER"] is None
    assert result["summary"]["p95_latency_ms"] is None
    assert result["summary"]["avg_token_usage"] is None
