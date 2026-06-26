import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from .paths import ensure_parent_dir


def iter_jsonl(path: str | Path, *, skip_blank: bool = True) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if skip_blank and not line.strip():
                continue
            yield json.loads(line)


def read_jsonl(path: str | Path, *, skip_blank: bool = True) -> list[dict[str, Any]]:
    return list(iter_jsonl(path, skip_blank=skip_blank))


def write_jsonl(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    *,
    ensure_ascii: bool = False,
) -> None:
    target = ensure_parent_dir(path)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=ensure_ascii))
            handle.write("\n")
