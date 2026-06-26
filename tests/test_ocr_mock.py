from __future__ import annotations

from l2w1.ocr import MockOCREngine, OCRRequest


def test_mock_ocr_uses_sample_mapping_deterministically() -> None:
    engine = MockOCREngine(
        texts_by_sample_id={"s1": "mapped text"},
        fixed_text="fallback text",
        mean_conf=0.9,
        min_conf=0.7,
        drop=0.2,
        conf=0.85,
        r_d=0.1,
    )
    request = OCRRequest(image_path="synthetic.png", sample_id="s1")

    first = engine.recognize(request)
    second = engine.recognize(request)

    assert first == second
    assert first.text == "mapped text"
    assert first.mean_conf == 0.9
    assert first.min_conf == 0.7
    assert first.drop == 0.2
    assert first.conf == 0.85
    assert first.r_d == 0.1


def test_mock_ocr_uses_fixed_text_fallback_and_to_dict_contains_t_a() -> None:
    engine = MockOCREngine(fixed_text="fixed")

    result = engine.recognize(OCRRequest(image_path="synthetic.png", sample_id="unknown"))

    assert result.text == "fixed"
    assert result.to_dict()["T_A"] == "fixed"
    assert result.to_dict() == {
        "T_A": "fixed",
        "text": "fixed",
        "mean_conf": 1.0,
        "min_conf": 1.0,
        "drop": 0.0,
        "conf": 1.0,
        "r_d": 0.0,
        "sample_id": "unknown",
    }
