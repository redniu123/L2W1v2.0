import pytest

from l2w1.io.jsonl import read_jsonl
from l2w1.metrics.summary import acc_rows, cer_rows

FIXTURES = __import__("pathlib").Path(__file__).parent / "fixtures"


def test_cer_rows_returns_global_cer_not_row_average():
    rows = read_jsonl(FIXTURES / "tiny_rows.jsonl")

    assert cer_rows(rows, "pred_text", "gt_text") == pytest.approx(1 / 12)


def test_acc_rows_returns_exact_match_accuracy_after_normalization():
    rows = read_jsonl(FIXTURES / "tiny_rows.jsonl")

    assert acc_rows(rows, "pred_text", "gt_text") == pytest.approx(2 / 3)


def test_summary_metrics_empty_rows_are_zero():
    assert cer_rows([], "pred_text", "gt_text") == 0.0
    assert acc_rows([], "pred_text", "gt_text") == 0.0


def test_cer_rows_with_empty_references():
    assert cer_rows([{"pred_text": "", "gt_text": ""}], "pred_text", "gt_text") == 0.0
    assert cer_rows([{"pred_text": "abc", "gt_text": ""}], "pred_text", "gt_text") == 1.0
