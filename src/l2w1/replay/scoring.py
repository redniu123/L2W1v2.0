from __future__ import annotations

from collections.abc import Mapping
from typing import Any

WUR_MEAN_WEIGHT = 0.5
WUR_MIN_WEIGHT = 0.3
WUR_DROP_WEIGHT = 0.2
WUR_MIN_CONF_GATE_THRESHOLD = 0.35
WUR_DROP_GATE_THRESHOLD = 0.20
WUR_GATE_BONUS = 0.10

_FORMAT_EQUIVALENCE = str.maketrans(
    {
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "｛": "{",
        "｝": "}",
        "，": ",",
        "：": ":",
        "；": ";",
        "！": "!",
        "？": "?",
        "。": ".",
    }
)


def normalize_format(text: str) -> str:
    return "" if not text else text.translate(_FORMAT_EQUIVALENCE)


def router_score(strategy: str, row: Mapping[str, Any], *, eta: float = 0.5) -> float:
    mean_conf = float(row.get("mean_conf", row.get("conf", 0.0)))
    min_conf = float(row.get("min_conf", row.get("conf", 0.0)))
    drop = float(row.get("drop", 0.0))
    conf = float(row.get("conf", mean_conf))
    r_d = float(row.get("r_d", 0.0))

    if strategy == "GCR":
        return 1.0 - conf
    if strategy == "DGCR":
        return (1.0 - conf) + r_d

    wur = (
        WUR_MEAN_WEIGHT * (1.0 - mean_conf)
        + WUR_MIN_WEIGHT * (1.0 - min_conf)
        + WUR_DROP_WEIGHT * drop
    )
    if min_conf < WUR_MIN_CONF_GATE_THRESHOLD:
        wur += WUR_GATE_BONUS
    if drop > WUR_DROP_GATE_THRESHOLD:
        wur += WUR_GATE_BONUS

    if strategy == "WUR":
        return float(wur)
    if strategy == "DWUR":
        return float(wur + eta * r_d)
    raise ValueError(f"Unsupported strategy: {strategy}")
