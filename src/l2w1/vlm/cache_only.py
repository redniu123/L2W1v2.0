from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from l2w1.replay.paper1_cache import build_cached_result_lookup

from .base import BaseVLMExpert
from .types import VLMRequest, VLMResponse


class CacheOnlyVLMExpert(BaseVLMExpert):
    backend: str = "cache_only"
    model_label: str = "cache_only"
    supports_parallel: bool = True
    max_concurrency: int = 1
    key_count: int = 1

    def __init__(
        self,
        cache_rows: Iterable[dict[str, Any]] | None = None,
        *,
        lookup: Mapping[str, dict[str, Any]] | None = None,
        provider: str = "cache_only",
    ) -> None:
        if lookup is not None:
            self._lookup = dict(lookup)
        else:
            self._lookup = build_cached_result_lookup(cache_rows or ())
        self.provider = provider

    def query(self, request: VLMRequest) -> VLMResponse:
        row = self._lookup.get(request.sample_id)
        if row is None:
            return VLMResponse(
                corrected_text=request.t_a,
                error_type="cached_result_missing",
                raw_output="",
                provider=self.provider,
            )

        corrected_text = _to_str(row.get("final_text_if_upgraded", ""))
        raw_output = _to_str(row.get("vlm_raw_output", corrected_text))
        if corrected_text == "":
            corrected_text = raw_output

        return VLMResponse(
            corrected_text=corrected_text,
            latency_ms=_to_optional_float(row.get("latency_ms")),
            token_usage=_to_optional_int(row.get("token_usage")),
            error_type=_to_str(row.get("error_type", "none")) or "none",
            raw_output=raw_output,
            provider=self.provider,
        )


def _to_str(value: object) -> str:
    return "" if value is None else str(value)


def _to_optional_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None


def _to_optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(str(value))
    except ValueError:
        return None
