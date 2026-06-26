from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, kw_only=True)
class OCRRequest:
    image_path: str
    sample_id: str = ""


@dataclass(frozen=True, kw_only=True)
class OCRResult:
    text: str
    mean_conf: float = 0.0
    min_conf: float = 0.0
    drop: float = 0.0
    conf: float = 0.0
    r_d: float = 0.0
    sample_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "T_A": self.text,
            "text": self.text,
            "mean_conf": self.mean_conf,
            "min_conf": self.min_conf,
            "drop": self.drop,
            "conf": self.conf,
            "r_d": self.r_d,
            "sample_id": self.sample_id,
        }
