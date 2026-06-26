from __future__ import annotations

from collections.abc import Mapping

from .base import BaseOCREngine
from .types import OCRRequest, OCRResult


class MockOCREngine(BaseOCREngine):
    def __init__(
        self,
        *,
        texts_by_sample_id: Mapping[str, str] | None = None,
        fixed_text: str = "",
        mean_conf: float = 1.0,
        min_conf: float = 1.0,
        drop: float = 0.0,
        conf: float = 1.0,
        r_d: float = 0.0,
    ) -> None:
        self._texts_by_sample_id = dict(texts_by_sample_id or {})
        self._fixed_text = fixed_text
        self._mean_conf = mean_conf
        self._min_conf = min_conf
        self._drop = drop
        self._conf = conf
        self._r_d = r_d

    def recognize(self, request: OCRRequest) -> OCRResult:
        text = self._texts_by_sample_id.get(request.sample_id, self._fixed_text)
        return OCRResult(
            text=text,
            mean_conf=self._mean_conf,
            min_conf=self._min_conf,
            drop=self._drop,
            conf=self._conf,
            r_d=self._r_d,
            sample_id=request.sample_id,
        )
