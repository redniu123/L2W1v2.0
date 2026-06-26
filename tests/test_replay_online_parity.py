from __future__ import annotations

import pytest
from scripts import paper1_online_budget_check as old_online
from scripts.run_efficiency_frontier import (
    summarize_extended_metrics,
    summarize_latency_and_token_usage,
)

from l2w1.replay.online_budget import replay_online
from l2w1.replay.scoring import router_score
from l2w1.routing.budget import BudgetControllerConfig


def _online_all_results() -> list[dict]:
    return [
        {
            "sample_id": "s1",
            "image_path": "synthetic/s1.png",
            "source_image_id": "img-1",
            "domain": "geology",
            "split": "test",
            "T_A": "地层（A）",
            "T_GT": "地层(A)",
            "has_professional_terms": True,
            "professional_terms": ["地层"],
            "mean_conf": 0.82,
            "min_conf": 0.30,
            "drop": 0.25,
            "conf": 0.78,
            "r_d": 0.22,
        },
        {
            "sample_id": "s2",
            "img_path": "synthetic/s2.png",
            "source_image_id": "img-2",
            "domain": "finance",
            "split": "test",
            "T_A": "ABXD",
            "T_GT": "ABCD",
            "has_professional_terms": False,
            "professional_terms": [],
            "mean_conf": 0.58,
            "min_conf": 0.44,
            "drop": 0.18,
            "conf": 0.55,
            "r_d": 0.05,
        },
        {
            "sample_id": "s3",
            "image_path": "synthetic/s3.png",
            "source_image_id": "img-3",
            "domain": "medicine",
            "split": "test",
            "T_A": "HELLO",
            "T_GT": "HELLO",
            "has_professional_terms": False,
            "professional_terms": [],
            "mean_conf": 0.96,
            "min_conf": 0.91,
            "drop": 0.02,
            "conf": 0.94,
            "r_d": 0.01,
        },
        {
            "sample_id": "s4",
            "image_path": "synthetic/s4.png",
            "source_image_id": "img-4",
            "domain": "geology",
            "split": "test",
            "T_A": "界XYZ",
            "T_GT": "边界XYZ",
            "has_professional_terms": True,
            "professional_terms": ["边界"],
            "mean_conf": 0.40,
            "min_conf": 0.20,
            "drop": 0.35,
            "conf": 0.42,
            "r_d": 0.40,
        },
        {
            "sample_id": "s5",
            "image_path": "synthetic/s5.png",
            "source_image_id": "img-5",
            "domain": "finance",
            "split": "test",
            "T_A": "CASH",
            "T_GT": "CASH",
            "has_professional_terms": False,
            "professional_terms": [],
            "mean_conf": 0.88,
            "min_conf": 0.62,
            "drop": 0.08,
            "conf": 0.87,
            "r_d": 0.31,
        },
        {
            "sample_id": "s6",
            "image_path": "synthetic/s6.png",
            "source_image_id": "img-6",
            "domain": "medicine",
            "split": "test",
            "T_A": "QWXR",
            "T_GT": "QWER",
            "has_professional_terms": False,
            "professional_terms": [],
            "mean_conf": 0.70,
            "min_conf": 0.52,
            "drop": 0.12,
            "conf": 0.69,
            "r_d": 0.12,
        },
    ]


def _cached_lookup() -> dict[str, dict]:
    return {
        "s1": {
            "sample_id": "s1",
            "final_text_if_upgraded": "地层(A)",
            "vlm_raw_output": "地层(A)",
            "latency_ms": 101.0,
            "token_usage": 11,
            "error_type": "none",
        },
        "s2": {
            "sample_id": "s2",
            "final_text_if_upgraded": "ABCD",
            "vlm_raw_output": "ABCD",
            "latency_ms": 205.5,
            "token_usage": 17,
            "error_type": "none",
        },
        "s4": {
            "sample_id": "s4",
            "final_text_if_upgraded": "边界XYZ",
            "vlm_raw_output": "边界XYZ",
            "latency_ms": 310.25,
            "token_usage": 25,
            "error_type": "boundary",
        },
        "s5": {
            "sample_id": "s5",
            "final_text_if_upgraded": "CASHX",
            "vlm_raw_output": "CASHX",
            "latency_ms": 88.0,
            "token_usage": 13,
            "error_type": "overcorrection",
        },
    }


@pytest.mark.parametrize("strategy", ["GCR", "WUR", "DGCR", "DWUR"])
def test_replay_online_matches_legacy_routeronly(strategy: str, monkeypatch) -> None:
    monkeypatch.setattr(old_online, "tqdm", lambda iterable, **_kwargs: iterable)
    all_results = _online_all_results()
    cached_lookup = _cached_lookup()
    budget_cfg = BudgetControllerConfig(
        window_size=3,
        warmup_samples=0,
        k=0.0,
        lambda_min=0.0,
        lambda_max=2.0,
        lambda_init=0.5,
        target_budget=0.5,
    )

    for row in all_results:
        assert router_score(strategy, row, eta=0.7) == old_online.build_router_score(
            strategy,
            row,
            eta=0.7,
        )

    old_result = old_online.run_online_routeronly(
        strategy,
        0.5,
        all_results,
        cached_lookup,
        budget_cfg,
        run_id="run_online_parity",
        prompt_version="prompt_v_test",
        agent_b_label="Synthetic Agent B",
        eta=0.7,
    )
    new_result = replay_online(
        strategy,
        0.5,
        all_results,
        cached_lookup,
        budget_cfg,
        run_id="run_online_parity",
        prompt_version="prompt_v_test",
        agent_b_label="Synthetic Agent B",
        eta=0.7,
        extended_metrics_fn=summarize_extended_metrics,
        usage_metrics_fn=summarize_latency_and_token_usage,
    )

    assert new_result["summary"] == old_result["summary"]
    assert new_result["per_sample"] == old_result["per_sample"]
    assert new_result["validation_logs"] == old_result["validation_logs"]


def test_replay_online_sets_injected_summary_metrics_to_none_when_absent() -> None:
    result = replay_online(
        "GCR",
        0.5,
        _online_all_results(),
        _cached_lookup(),
        BudgetControllerConfig(warmup_samples=0, k=0.0, lambda_init=0.5, target_budget=0.5),
    )

    assert result["summary"]["BoundaryDeletionRecallAtB"] is None
    assert result["summary"]["SubstitutionCER"] is None
    assert result["summary"]["p95_latency_ms"] is None
    assert result["summary"]["avg_token_usage"] is None
