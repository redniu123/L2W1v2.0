from __future__ import annotations

import pytest

from l2w1.metrics.extended import (
    compute_cer,
    count_substitutions,
    identify_boundary_deletion,
    normalize_eval_text,
    summarize_extended_metrics,
    summarize_latency_and_token_usage,
)


def test_normalize_eval_text_maps_full_width_punctuation_only():
    assert normalize_eval_text("（甲）【乙】｛丙｝，：；！？。") == "(甲)[乙]{丙},:;!?."


def test_compute_cer_returns_zero_when_ground_truth_is_empty():
    assert compute_cer("nonempty", "") == 0.0
    assert compute_cer("A（B）", "A(B)") == 0.0


@pytest.mark.parametrize(
    ("agent_a_text", "gt_text", "expected"),
    [
        ("BCDE", "ABCDE", True),
        ("ABCD", "ABCDE", True),
        ("ABDE", "ABCDE", False),
        ("ABCDE", "", False),
    ],
)
def test_identify_boundary_deletion(agent_a_text: str, gt_text: str, expected: bool):
    assert identify_boundary_deletion(agent_a_text, gt_text, k=2) is expected


@pytest.mark.parametrize(
    ("prediction", "reference", "expected"),
    [
        ("ABXDE", "ABCDE", 1),
        ("AXYDE", "ABCDE", 2),
        ("ABDE", "ABCDE", 0),
        ("ABCDE", "ABCDE", 0),
    ],
)
def test_count_substitutions(prediction: str, reference: str, expected: int):
    assert count_substitutions(prediction, reference) == expected


def test_summarize_extended_metrics_uses_boundary_recall_and_substitution_cer():
    per_sample = [
        {
            "gt": "ABCDE",
            "ocr_text": "BCDE",
            "final_text": "ABCXE",
            "selected_for_upgrade": True,
        },
        {
            "gt": "ABCDE",
            "ocr_text": "ABCD",
            "final_text": "ABYDE",
            "selected_for_upgrade": False,
        },
        {
            "gt": "ABCDE",
            "ocr_text": "ABDE",
            "final_text": "ABCDE",
            "selected_for_upgrade": True,
        },
    ]

    assert summarize_extended_metrics(per_sample) == {
        "Boundary_Deletion_Recall@B": 0.5,
        "Substitution_CER": 0.133333,
    }


def test_summarize_extended_metrics_returns_zeroes_for_empty_input():
    assert summarize_extended_metrics([]) == {
        "Boundary_Deletion_Recall@B": 0.0,
        "Substitution_CER": 0.0,
    }


def test_summarize_latency_and_token_usage_filters_missing_values():
    per_sample = [
        {"latency_ms": 10, "token_usage": 100},
        {"latency_ms": None, "token_usage": 200},
        {"latency_ms": 30, "token_usage": None},
        {"token_usage": 50},
        {"latency_ms": 20},
    ]

    assert summarize_latency_and_token_usage(per_sample) == {
        "P95_Latency_MS": 29.0,
        "Avg_Token_Usage": 116.667,
        "Total_Token_Usage": 350.0,
        "N_Latency_Valid": 3,
        "N_Token_Valid": 3,
    }


def test_summarize_latency_and_token_usage_returns_zeroes_for_empty_input():
    assert summarize_latency_and_token_usage([]) == {
        "P95_Latency_MS": 0.0,
        "Avg_Token_Usage": 0.0,
        "Total_Token_Usage": 0.0,
        "N_Latency_Valid": 0,
        "N_Token_Valid": 0,
    }
