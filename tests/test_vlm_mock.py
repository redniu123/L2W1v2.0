from __future__ import annotations

from l2w1.vlm import MockVLMExpert, VLMRequest


def test_bracketed_prompt_returns_inner_text() -> None:
    expert = MockVLMExpert(latency_ms=1.25)

    response = expert.query(
        VLMRequest(
            image_path="synthetic.png",
            t_a="fallback",
            user_prompt="Please check 【  corrected text  】.",
        )
    )

    assert response.corrected_text == "corrected text"
    assert response.latency_ms == 1.25
    assert response.error_type == "none"


def test_no_bracket_falls_back_to_t_a() -> None:
    expert = MockVLMExpert()

    response = expert.query(
        VLMRequest(
            image_path="synthetic.png",
            t_a="agent a text",
            user_prompt="No bracketed text here.",
        )
    )

    assert response.corrected_text == "agent a text"


def test_query_dict_matches_query_contract() -> None:
    expert = MockVLMExpert(latency_ms=2.0)
    prompt = {
        "T_A": "fallback",
        "image_path": "synthetic.png",
        "sample_id": "s1",
        "min_conf_idx": 3,
        "user_prompt": "OCR text: 【fixed】",
    }

    response = expert.query(
        VLMRequest(
            image_path="synthetic.png",
            t_a="fallback",
            sample_id="s1",
            min_conf_idx=3,
            user_prompt="OCR text: 【fixed】",
        )
    )

    assert expert.query_dict(prompt) == response.to_dict()


def test_to_dict_keys_and_latency_are_deterministic() -> None:
    expert = MockVLMExpert(latency_ms=3.5)
    request = VLMRequest(image_path="synthetic.png", t_a="same")

    first = expert.query(request)
    second = expert.query(request)

    assert first.latency_ms == second.latency_ms == 3.5
    assert first.to_dict() == {
        "corrected_text": "same",
        "latency_ms": 3.5,
        "token_usage": None,
        "error_type": "none",
    }
