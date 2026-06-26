"""OCR engine interfaces for Agent A."""

from .base import BaseOCREngine
from .cache_only import CacheOnlyOCREngine
from .mock import MockOCREngine
from .types import OCRRequest, OCRResult

__all__ = [
    "BaseOCREngine",
    "CacheOnlyOCREngine",
    "MockOCREngine",
    "OCRRequest",
    "OCRResult",
]
