"""Metrics helpers for L2W1 cleanup modules."""

from .cer import calculate_cer
from .edit_distance import levenshtein_distance
from .reliability import (
    calculate_correction_rate,
    calculate_cvr_aer,
    calculate_ocr_r,
    classify_correction,
)
from .summary import acc_rows, cer_rows
from .text import normalize_text_for_eval

__all__ = [
    "acc_rows",
    "calculate_cer",
    "calculate_correction_rate",
    "calculate_cvr_aer",
    "calculate_ocr_r",
    "cer_rows",
    "classify_correction",
    "levenshtein_distance",
    "normalize_text_for_eval",
]
