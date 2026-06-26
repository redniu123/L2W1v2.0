from __future__ import annotations

from l2w1.vlm import CacheOnlyVLMExpert, VLMRequest


def test_cache_hit_returns_cached_vlm_fields() -> None:
    expert = CacheOnlyVLMExpert(
        [
            {
                "sample_id": "s1",
                "vlm_raw_output": "raw corrected",
                "final_text_if_upgraded": "final corrected",
                "latency_ms": "42.5",
                "token_usage": "17",
                "error_type": "none",
            }
        ]
    )

    response = expert.query(VLMRequest(image_path="synthetic.png", t_a="agent a", sample_id="s1"))

    assert response.corrected_text == "final corrected"
    assert response.raw_output == "raw corrected"
    assert response.latency_ms == 42.5
    assert response.token_usage == 17
    assert response.error_type == "none"
    assert response.to_dict() == {
        "corrected_text": "final corrected",
        "latency_ms": 42.5,
        "token_usage": 17,
        "error_type": "none",
    }


def test_cache_miss_returns_t_a_with_missing_error() -> None:
    expert = CacheOnlyVLMExpert([])

    response = expert.query(
        VLMRequest(image_path="synthetic.png", t_a="agent a fallback", sample_id="missing")
    )

    assert response.corrected_text == "agent a fallback"
    assert response.error_type == "cached_result_missing"
    assert response.latency_ms is None
    assert response.token_usage is None
