import json
from pathlib import Path

import pytest

from l2w1.metrics.reliability import (
    calculate_correction_rate,
    calculate_cvr_aer,
    calculate_ocr_r,
    classify_correction,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load_cases():
    return json.loads((FIXTURES / "tiny_reliability_cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", _load_cases(), ids=lambda case: case["name"])
def test_classify_correction_fixture_cases(case):
    assert (
        classify_correction(
            case["agent_a_text"],
            case["system_text"],
            case["reference_text"],
        )
        == case["expected_class"]
    )


def test_calculate_ocr_r_overcorrection_rate():
    ocr_r, details = calculate_ocr_r("中国地质", "中囯地质", "中国地质")

    assert ocr_r == pytest.approx(0.25)
    assert details["total_correct_in_a"] == 4
    assert details["overcorrected"] == 1


def test_calculate_ocr_r_no_correct_baseline_chars():
    ocr_r, details = calculate_ocr_r("甲乙", "中国", "中国")

    assert ocr_r == 0.0
    assert details["total_correct_in_a"] == 0
    assert details["overcorrected"] == 0


def test_calculate_ocr_r_empty_reference():
    ocr_r, details = calculate_ocr_r("", "abc", "")

    assert ocr_r == 0.0
    assert details["error"] == "reference_text is empty"


@pytest.mark.parametrize("case", _load_cases(), ids=lambda case: case["name"])
def test_calculate_correction_rate_fixture_cases(case):
    correction_rate, details = calculate_correction_rate(
        case["agent_a_text"],
        case["system_text"],
        case["reference_text"],
    )

    assert correction_rate == pytest.approx(case["expected_correction_rate"])
    if case["reference_text"]:
        assert "total_wrong_in_a" in details


def test_system_worse_has_positive_ocr_r_and_zero_correction_rate():
    ocr_r, ocr_details = calculate_ocr_r("中囯地质", "中囯地貭", "中国地质")
    correction_rate, corr_details = calculate_correction_rate("中囯地质", "中囯地貭", "中国地质")

    assert classify_correction("中囯地质", "中囯地貭", "中国地质") == "worsened"
    assert ocr_r == pytest.approx(1 / 3)
    assert ocr_details["overcorrected"] == 1
    assert correction_rate == 0.0
    assert corr_details["corrected"] == 0


def test_calculate_correction_rate_perfect_agent_a_returns_one():
    correction_rate, details = calculate_correction_rate("中国地质", "中国地质", "中国地质")

    assert correction_rate == 1.0
    assert details["total_wrong_in_a"] == 0


def test_calculate_cvr_aer_matches_old_semantics():
    records = [
        {
            "upgrade": True,
            "agent_a_text": "中国地质",
            "agent_b_text": "中国地貭",
            "final_text": "中国地貭",
        },
        {
            "upgrade": True,
            "agent_a_text": "中国地质",
            "agent_b_text": "地质中国额外",
            "final_text": "中国地质",
        },
        {
            "is_hard": True,
            "agent_a": {"text": "金融市场"},
            "agent_b": {"text": "金融市场"},
            "final_text": "金融市场",
        },
        {
            "upgrade": False,
            "agent_a_text": "不计入",
            "agent_b_text": "不计入x",
            "final_text": "不计入x",
        },
    ]

    cvr, aer, details = calculate_cvr_aer(records)

    assert cvr == pytest.approx(1 / 3)
    assert aer == pytest.approx(1 / 3)
    assert details["upgraded_count"] == 3
    assert details["violation_count"] == 1
    assert details["accepted_edit_count"] == 1


def test_calculate_cvr_aer_no_upgraded_samples():
    cvr, aer, details = calculate_cvr_aer([{"upgrade": False}])

    assert cvr == 0.0
    assert aer == 0.0
    assert details["upgraded_count"] == 0
