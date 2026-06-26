from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, kw_only=True)
class VLMRequest:
    image_path: str
    t_a: str
    sample_id: str = ""
    min_conf_idx: int = -1
    user_prompt: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "T_A": self.t_a,
            "sample_id": self.sample_id,
            "min_conf_idx": self.min_conf_idx,
            "user_prompt": self.user_prompt,
        }


@dataclass(frozen=True, kw_only=True)
class VLMResponse:
    corrected_text: str
    latency_ms: float | None = None
    token_usage: int | None = None
    error_type: str = "none"
    raw_output: str = ""
    provider: str = ""

    def __post_init__(self) -> None:
        if self.raw_output == "":
            object.__setattr__(self, "raw_output", self.corrected_text)

    def to_dict(self) -> dict[str, Any]:
        return {
            "corrected_text": self.corrected_text,
            "latency_ms": self.latency_ms,
            "token_usage": self.token_usage,
            "error_type": self.error_type,
        }
