import json

import pytest

from l2w1.metrics.cer import calculate_cer
from l2w1.metrics.edit_distance import levenshtein_distance
from l2w1.metrics.text import normalize_text_for_eval

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


def test_normalize_text_for_eval_is_conservative():
    assert normalize_text_for_eval(None) == ""
    assert normalize_text_for_eval(" 中国\t金融\n ") == "中国金融"
    assert normalize_text_for_eval("， 。") == "，。"
    assert normalize_text_for_eval(123) == "123"


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        ("", "", 0),
        ("abc", "", 3),
        ("", "abc", 3),
        ("中国", "中国", 0),
        ("中囯", "中国", 1),
        ("中国A", "中国", 1),
        ("中国", "中国A", 1),
    ],
)
def test_levenshtein_distance(left, right, expected):
    assert levenshtein_distance(left, right) == expected


def test_calculate_cer_fixture_cases():
    cases = json.loads((FIXTURES / "tiny_metrics_cases.json").read_text(encoding="utf-8"))["cases"]

    for case in cases:
        details = calculate_cer(case["prediction"], case["reference"], return_details=True)
        assert details["edit_distance"] == case["edit_distance"]
        assert details["reference_length"] == case["reference_length"]
        assert details["cer"] == pytest.approx(case["cer"])
        assert calculate_cer(case["prediction"], case["reference"]) == pytest.approx(case["cer"])


def test_calculate_cer_normalizes_by_default():
    assert calculate_cer(" 中国 金融 ", "中国金融") == 0.0


def test_calculate_cer_can_disable_normalization():
    details = calculate_cer(" 中国 ", "中国", normalize=False, return_details=True)

    assert details["cer"] == pytest.approx(1.0)
    assert details["edit_distance"] == 2
    assert details["prediction"] == " 中国 "
    assert details["reference"] == "中国"
