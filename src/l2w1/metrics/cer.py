from typing import Any

from .edit_distance import levenshtein_distance
from .text import normalize_text_for_eval


def calculate_cer(
    prediction: object,
    reference: object,
    *,
    normalize: bool = True,
    return_details: bool = False,
) -> float | dict[str, Any]:
    if normalize:
        pred_text = normalize_text_for_eval(prediction)
        ref_text = normalize_text_for_eval(reference)
    else:
        pred_text = "" if prediction is None else str(prediction)
        ref_text = "" if reference is None else str(reference)

    edit_distance = levenshtein_distance(pred_text, ref_text)
    reference_length = len(ref_text)
    if reference_length == 0:
        cer = 0.0 if pred_text == "" else 1.0
    else:
        cer = edit_distance / reference_length

    if not return_details:
        return cer

    return {
        "cer": cer,
        "edit_distance": edit_distance,
        "reference_length": reference_length,
        "prediction": pred_text,
        "reference": ref_text,
    }
