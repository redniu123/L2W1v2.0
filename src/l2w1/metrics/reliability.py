"""Reliability and correction metrics extracted from paper1 evaluation code."""

from __future__ import annotations

import difflib
from collections.abc import Iterable
from typing import Any

from .edit_distance import levenshtein_distance
from .text import normalize_text_for_eval


def _maybe_normalize(text: object, normalize: bool) -> str:
    if normalize:
        return normalize_text_for_eval(text)
    if text is None:
        return ""
    return str(text)


def _equal_gt_positions(prediction: str, reference: str) -> set[int]:
    matcher = difflib.SequenceMatcher(None, prediction, reference)
    positions: set[int] = set()
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            positions.update(range(j1, j2))
    return positions


def _wrong_gt_positions(prediction: str, reference: str) -> set[int]:
    matcher = difflib.SequenceMatcher(None, prediction, reference)
    positions: set[int] = set()
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            positions.update(range(j1, j2))
    return positions


def calculate_ocr_r(
    agent_a_text: object,
    system_text: object,
    reference_text: object,
    *,
    normalize: bool = True,
) -> tuple[float, dict[str, Any]]:
    """Return paper1 OCR-R: Agent A correct chars changed wrong by system.

    The value is a 0-1 ratio:
    Count(Agent A correct -> System wrong) / Total correct chars in Agent A.
    """
    agent_a = _maybe_normalize(agent_a_text, normalize)
    system = _maybe_normalize(system_text, normalize)
    reference = _maybe_normalize(reference_text, normalize)

    if len(reference) == 0:
        return 0.0, {
            "error": "reference_text is empty",
            "total_correct_in_a": 0,
            "overcorrected": 0,
        }

    correct_in_a = _equal_gt_positions(agent_a, reference)
    total_correct_in_a = len(correct_in_a)
    if total_correct_in_a == 0:
        return 0.0, {
            "total_correct_in_a": 0,
            "overcorrected": 0,
            "note": "Agent A has no correct characters",
        }

    system_correct_positions = _equal_gt_positions(system, reference)
    overcorrected_positions = sorted(correct_in_a - system_correct_positions)
    overcorrected = len(overcorrected_positions)
    ocr_r = overcorrected / total_correct_in_a

    return ocr_r, {
        "total_correct_in_a": total_correct_in_a,
        "overcorrected": overcorrected,
        "ocr_r": ocr_r,
        "overcorrected_positions": [
            {
                "gt_position": pos,
                "gt_char": reference[pos] if pos < len(reference) else "",
            }
            for pos in overcorrected_positions[:10]
        ],
        "note": "Agent A Correct -> System Wrong transitions",
    }


def calculate_correction_rate(
    agent_a_text: object,
    system_text: object,
    reference_text: object,
    *,
    normalize: bool = True,
) -> tuple[float, dict[str, Any]]:
    """Return paper1 correction rate: Agent A wrong chars fixed by system.

    The value is a 0-1 ratio:
    Count(Agent A wrong -> System correct) / Total wrong chars in Agent A.
    """
    agent_a = _maybe_normalize(agent_a_text, normalize)
    system = _maybe_normalize(system_text, normalize)
    reference = _maybe_normalize(reference_text, normalize)

    if len(reference) == 0:
        return 0.0, {
            "error": "reference_text is empty",
            "total_wrong_in_a": 0,
            "corrected": 0,
        }

    wrong_in_a = _wrong_gt_positions(agent_a, reference)
    total_wrong_in_a = len(wrong_in_a)
    if total_wrong_in_a == 0:
        return 1.0, {
            "total_wrong_in_a": 0,
            "corrected": 0,
            "note": "Agent A has no errors (perfect recognition)",
        }

    system_correct_positions = _equal_gt_positions(system, reference)
    corrected_positions = sorted(wrong_in_a & system_correct_positions)
    corrected = len(corrected_positions)
    correction_rate = corrected / total_wrong_in_a

    return correction_rate, {
        "total_wrong_in_a": total_wrong_in_a,
        "corrected": corrected,
        "correction_rate": correction_rate,
        "corrected_positions": [
            {
                "gt_position": pos,
                "gt_char": reference[pos] if pos < len(reference) else "",
            }
            for pos in corrected_positions[:10]
        ],
        "note": "Agent A Wrong -> System Correct transitions",
    }


def classify_correction(
    agent_a_text: object,
    system_text: object,
    reference_text: object,
    *,
    normalize: bool = True,
) -> str:
    """Classify the sample-level correction outcome by edit distance to GT."""
    agent_a = _maybe_normalize(agent_a_text, normalize)
    system = _maybe_normalize(system_text, normalize)
    reference = _maybe_normalize(reference_text, normalize)

    baseline_ed = levenshtein_distance(agent_a, reference)
    system_ed = levenshtein_distance(system, reference)

    if baseline_ed == 0:
        if system_ed == 0:
            return "unchanged_correct"
        return "overcorrected"

    if system_ed == 0:
        return "corrected"
    if system_ed < baseline_ed:
        return "improved"
    if system_ed > baseline_ed:
        return "worsened"
    return "unchanged_wrong"


def calculate_cvr_aer(
    records: Iterable[dict[str, Any]],
    *,
    ed_threshold: int = 2,
    length_change_threshold: float = 0.2,
    normalize: bool = True,
) -> tuple[float, float, dict[str, Any]]:
    """Return paper1-style CVR and AER for upgraded records.

    CVR is the proportion of upgraded samples whose proposed Agent B edit
    violates edit-distance or length-change constraints and is rejected.
    AER is the proportion of upgraded samples whose final text differs from
    Agent A text.
    """
    upgraded_samples = [
        record for record in records if record.get("upgrade", record.get("is_hard", False))
    ]

    if not upgraded_samples:
        return (
            0.0,
            0.0,
            {
                "upgraded_count": 0,
                "violation_count": 0,
                "accepted_edit_count": 0,
                "note": "No upgraded samples",
            },
        )

    violation_count = 0
    accepted_edit_count = 0

    for record in upgraded_samples:
        agent_a_text = _record_text(record, "agent_a_text", "agent_a", normalize)
        agent_b_text = _record_text(record, "agent_b_text", "agent_b", normalize)
        final_text = _maybe_normalize(record.get("final_text", ""), normalize)

        if not final_text:
            final_text = agent_b_text if agent_b_text else agent_a_text

        if final_text != agent_a_text:
            accepted_edit_count += 1

        if agent_b_text and agent_b_text != agent_a_text:
            ed = levenshtein_distance(agent_a_text, agent_b_text)
            len_a = len(agent_a_text) if agent_a_text else 1
            len_b = len(agent_b_text) if agent_b_text else 0
            length_change = abs(len_b - len_a) / len_a

            if ed > ed_threshold or length_change > length_change_threshold:
                if final_text == agent_a_text:
                    violation_count += 1

    upgraded_count = len(upgraded_samples)
    cvr = violation_count / upgraded_count
    aer = accepted_edit_count / upgraded_count

    return (
        cvr,
        aer,
        {
            "upgraded_count": upgraded_count,
            "violation_count": violation_count,
            "accepted_edit_count": accepted_edit_count,
            "ed_threshold": ed_threshold,
            "length_change_threshold": length_change_threshold,
        },
    )


def _record_text(
    record: dict[str, Any],
    flat_key: str,
    nested_key: str,
    normalize: bool,
) -> str:
    value = record.get(flat_key)
    if value is None:
        nested = record.get(nested_key, {})
        if isinstance(nested, dict):
            value = nested.get("text", "")
        else:
            value = ""
    return _maybe_normalize(value, normalize)
