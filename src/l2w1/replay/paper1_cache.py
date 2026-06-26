from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from l2w1.io.jsonl import read_jsonl


def load_cache_rows(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


def build_cached_result_lookup(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id", "")
        if sample_id:
            lookup[sample_id] = row
    return lookup
