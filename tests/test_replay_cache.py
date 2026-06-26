from __future__ import annotations

from typing import Any

from l2w1.io.jsonl import write_jsonl
from l2w1.replay.paper1_cache import build_cached_result_lookup, load_cache_rows


def test_load_cache_rows_roundtrip(tmp_path) -> None:
    rows: list[dict[str, Any]] = [
        {"sample_id": "s1", "final_text_if_upgraded": "alpha"},
        {"sample_id": "s2", "final_text_if_upgraded": "beta", "token_usage": 42},
    ]
    path = tmp_path / "cache.jsonl"

    write_jsonl(path, rows)

    assert load_cache_rows(path) == rows


def test_build_cached_result_lookup_skips_empty_ids_and_overrides_later_rows() -> None:
    first = {"sample_id": "dup", "value": 1}
    empty = {"sample_id": "", "value": "skip-empty"}
    missing = {"value": "skip-missing"}
    replacement = {"sample_id": "dup", "value": 2}
    kept = {"sample_id": "kept", "value": 3}

    lookup = build_cached_result_lookup([first, empty, missing, replacement, kept])

    assert lookup == {"dup": replacement, "kept": kept}
