from __future__ import annotations

import pytest

from l2w1.ocr import CacheOnlyOCREngine, OCRRequest


def test_cache_hit_builds_ocr_result_from_agent_a_row() -> None:
    engine = CacheOnlyOCREngine(
        [
            {
                "sample_id": "s1",
                "T_A": "agent a text",
                "mean_conf": "0.91",
                "min_conf": "0.42",
                "drop": "0.08",
                "conf": "0.88",
                "r_d": "0.12",
            }
        ]
    )

    result = engine.recognize(OCRRequest(image_path="synthetic.png", sample_id="s1"))

    assert result.text == "agent a text"
    assert result.mean_conf == 0.91
    assert result.min_conf == 0.42
    assert result.drop == 0.08
    assert result.conf == 0.88
    assert result.r_d == 0.12
    assert result.sample_id == "s1"


def test_cache_miss_raises_key_error() -> None:
    engine = CacheOnlyOCREngine([])

    with pytest.raises(KeyError, match="missing"):
        engine.recognize(OCRRequest(image_path="synthetic.png", sample_id="missing"))
