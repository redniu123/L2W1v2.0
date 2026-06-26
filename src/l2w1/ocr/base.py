from __future__ import annotations

from abc import ABC, abstractmethod

from .types import OCRRequest, OCRResult


class BaseOCREngine(ABC):
    @abstractmethod
    def recognize(self, request: OCRRequest) -> OCRResult:
        """Recognize Agent A text for one image request."""
