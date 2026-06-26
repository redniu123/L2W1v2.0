from collections.abc import Iterable
from typing import Any

from .edit_distance import levenshtein_distance
from .text import normalize_text_for_eval


def cer_rows(
    rows: Iterable[dict[str, Any]],
    pred_key: str,
    ref_key: str = "gt_text",
) -> float:
    total_edit_distance = 0
    total_reference_length = 0

    for row in rows:
        prediction = normalize_text_for_eval(row.get(pred_key, ""))
        reference = normalize_text_for_eval(row.get(ref_key, ""))
        total_edit_distance += levenshtein_distance(prediction, reference)
        total_reference_length += len(reference)

    if total_reference_length == 0:
        return 0.0 if total_edit_distance == 0 else 1.0
    return total_edit_distance / total_reference_length


def acc_rows(
    rows: Iterable[dict[str, Any]],
    pred_key: str,
    ref_key: str = "gt_text",
) -> float:
    row_list = list(rows)
    if not row_list:
        return 0.0

    correct = 0
    for row in row_list:
        prediction = normalize_text_for_eval(row.get(pred_key, ""))
        reference = normalize_text_for_eval(row.get(ref_key, ""))
        correct += int(prediction == reference)
    return correct / len(row_list)
