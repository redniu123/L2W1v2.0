import csv
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from .paths import ensure_parent_dir


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
    fieldnames: Sequence[str] | None = None,
) -> None:
    target = ensure_parent_dir(path)
    row_list = list(rows)

    if fieldnames is None:
        if not row_list:
            target.write_text("", encoding="utf-8")
            return
        fieldnames = list(row_list[0].keys())

    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in row_list:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
