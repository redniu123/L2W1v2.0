from __future__ import annotations

import re

from .base import BaseVLMExpert
from .types import VLMRequest, VLMResponse

_BRACKETED_TEXT_RE = re.compile(r"\u3010\s*(.+?)\s*\u3011")


class MockVLMExpert(BaseVLMExpert):
    backend: str = "mock"
    model_label: str = "mock"
    supports_parallel: bool = False
    max_concurrency: int = 1
    key_count: int = 1

    def __init__(self, *, latency_ms: float = 0.0, provider: str = "mock") -> None:
        self.latency_ms = latency_ms
        self.provider = provider

    def query(self, request: VLMRequest) -> VLMResponse:
        match = _BRACKETED_TEXT_RE.search(request.user_prompt)
        corrected_text = match.group(1).strip() if match else request.t_a
        return VLMResponse(
            corrected_text=corrected_text,
            latency_ms=self.latency_ms,
            token_usage=None,
            error_type="none",
            raw_output=corrected_text,
            provider=self.provider,
        )
