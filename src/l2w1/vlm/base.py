from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .types import VLMRequest, VLMResponse


class BaseVLMExpert(ABC):
    backend: str = "base"
    model_label: str = "base"
    supports_parallel: bool = False
    max_concurrency: int = 1
    key_count: int = 1

    @abstractmethod
    def query(self, request: VLMRequest) -> VLMResponse:
        """Run Agent B correction for one request."""

    def query_dict(self, prompt: dict[str, Any]) -> dict[str, Any]:
        request = VLMRequest(
            image_path=str(prompt.get("image_path", "")),
            t_a=str(prompt.get("T_A", "")),
            sample_id=str(prompt.get("sample_id", "")),
            min_conf_idx=_to_int(prompt.get("min_conf_idx", -1), default=-1),
            user_prompt=str(prompt.get("user_prompt", "")),
        )
        return self.query(request).to_dict()


def _to_int(value: object, *, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return default
