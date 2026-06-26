from collections.abc import Iterable
from typing import Any


def index_rows_by_sample_id(
    rows: Iterable[dict[str, Any]],
    *,
    key: str = "sample_id",
    on_duplicate: str = "error",
) -> dict[Any, dict[str, Any]]:
    if on_duplicate not in {"error", "first", "last"}:
        raise ValueError("on_duplicate must be one of: error, first, last")

    indexed: dict[Any, dict[str, Any]] = {}
    for row in rows:
        if key not in row:
            raise KeyError(f"missing required key: {key}")
        sample_id = row[key]
        if sample_id in indexed:
            if on_duplicate == "error":
                raise ValueError(f"duplicate {key}: {sample_id!r}")
            if on_duplicate == "first":
                continue
        indexed[sample_id] = row
    return indexed
