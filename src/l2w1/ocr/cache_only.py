from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from l2w1.replay.paper1_cache import build_cached_result_lookup

from .base import BaseOCREngine
from .types import OCRRequest, OCRResult


class CacheOnlyOCREngine(BaseOCREngine):
    """Read OCR results from an Agent A cache indexed by sample_id.

    Cache hits build an OCRResult from T_A, mean_conf, min_conf, drop, conf,
    and r_d. Cache misses raise KeyError because this engine is read-only and
    must not fall back to live OCR model execution.
    """

    def __init__(
        self,
        cache_rows: Iterable[dict[str, Any]] | None = None,
        *,
        lookup: Mapping[str, dict[str, Any]] | None = None,
    ) -> None:
        if lookup is not None:
            self._lookup = dict(lookup)
        else:
            self._lookup = build_cached_result_lookup(cache_rows or ())

    def recognize(self, request: OCRRequest) -> OCRResult:
        row = self._lookup.get(request.sample_id)
        if row is None:
            raise KeyError(f"missing cached OCR result for sample_id: {request.sample_id!r}")

        return OCRResult(
            text=_to_str(row.get("T_A", "")),
            mean_conf=_to_float(row.get("mean_conf")),
            min_conf=_to_float(row.get("min_conf")),
            drop=_to_float(row.get("drop")),
            conf=_to_float(row.get("conf")),
            r_d=_to_float(row.get("r_d")),
            sample_id=request.sample_id,
        )


def _to_str(value: object) -> str:
    return "" if value is None else str(value)


def _to_float(value: object) -> float:
    if value is None or value == "":
        return 0.0
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return 0.0
