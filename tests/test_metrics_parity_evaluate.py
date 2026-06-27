import importlib.util
import json
import sys
from pathlib import Path

import pytest

from l2w1.metrics.cer import calculate_cer as new_calculate_cer
from l2w1.metrics.edit_distance import levenshtein_distance as new_levenshtein_distance
from l2w1.metrics.reliability import (
    calculate_correction_rate as new_calculate_correction_rate,
)
from l2w1.metrics.reliability import (
    calculate_cvr_aer as new_calculate_cvr_aer,
)
from l2w1.metrics.reliability import (
    calculate_ocr_r as new_calculate_ocr_r,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


def _load_old_evaluate_module():
    module_path = ROOT / "scripts" / "tools" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("paper1_evaluate_for_parity", module_path)
    if spec is None or spec.loader is None:
        pytest.skip("scripts/tools/evaluate.py could not be loaded with importlib")
    module = importlib.util.module_from_spec(spec)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
    return module


@pytest.fixture(scope="module")
def old_eval():
    return _load_old_evaluate_module()


def _metrics_cases():
    return json.loads((FIXTURES / "tiny_metrics_cases.json").read_text(encoding="utf-8"))["cases"]


def _reliability_cases():
    return json.loads((FIXTURES / "tiny_reliability_cases.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", _metrics_cases())
def test_levenshtein_distance_matches_evaluate(old_eval, case):
    prediction = case["prediction"]
    reference = case["reference"]

    assert new_levenshtein_distance(prediction, reference) == old_eval.levenshtein_distance(
        prediction,
        reference,
    )


@pytest.mark.parametrize("case", _metrics_cases())
def test_calculate_cer_matches_evaluate_without_normalization(old_eval, case):
    prediction = case["prediction"]
    reference = case["reference"]

    new_details = new_calculate_cer(
        prediction,
        reference,
        normalize=False,
        return_details=True,
    )
    old_cer, old_ops = old_eval.calculate_cer(prediction, reference, return_details=True)

    assert new_details["cer"] == pytest.approx(old_cer)
    assert new_details["edit_distance"] == old_ops.total
    assert new_details["reference_length"] == len(reference)


@pytest.mark.parametrize("case", _reliability_cases(), ids=lambda case: case["name"])
def test_calculate_ocr_r_matches_evaluate(old_eval, case):
    new_ocr_r, new_details = new_calculate_ocr_r(
        case["agent_a_text"],
        case["system_text"],
        case["reference_text"],
        normalize=False,
    )
    old_ocr_r, old_details = old_eval.calculate_ocr_r(
        case["agent_a_text"],
        case["system_text"],
        case["reference_text"],
    )

    assert new_ocr_r == pytest.approx(old_ocr_r)
    assert new_details["total_correct_in_a"] == old_details["total_correct_in_a"]
    assert new_details["overcorrected"] == old_details["overcorrected"]


@pytest.mark.parametrize("case", _reliability_cases(), ids=lambda case: case["name"])
def test_calculate_correction_rate_matches_evaluate(old_eval, case):
    new_rate, new_details = new_calculate_correction_rate(
        case["agent_a_text"],
        case["system_text"],
        case["reference_text"],
        normalize=False,
    )
    old_rate, old_details = old_eval.calculate_correction_rate(
        case["agent_a_text"],
        case["system_text"],
        case["reference_text"],
    )

    assert new_rate == pytest.approx(old_rate)
    assert new_details["total_wrong_in_a"] == old_details["total_wrong_in_a"]
    assert new_details["corrected"] == old_details["corrected"]


def test_calculate_cvr_aer_matches_evaluate(old_eval):
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
            "upgrade": True,
            "agent_a_text": "医学",
            "agent_b_text": "医学术语",
            "final_text": "",
        },
        {
            "upgrade": False,
            "agent_a_text": "不计入",
            "agent_b_text": "不计入x",
            "final_text": "不计入x",
        },
    ]

    new_cvr, new_aer, new_details = new_calculate_cvr_aer(records, normalize=False)
    old_cvr, old_aer, old_details = old_eval.calculate_cvr_aer(records)

    assert new_cvr == pytest.approx(old_cvr)
    assert new_aer == pytest.approx(old_aer)
    assert new_details["upgraded_count"] == old_details["upgraded_count"]
    assert new_details["violation_count"] == old_details["violation_count"]
    assert new_details["accepted_edit_count"] == old_details["accepted_edit_count"]
