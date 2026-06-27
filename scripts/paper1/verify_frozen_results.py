#!/usr/bin/env python3
"""Verify paper1 frozen result integrity without OCR, VLM, or API calls."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from scripts._common import add_repo_root_to_path

EXPECTED_N = 3424
REPO_ROOT = add_repo_root_to_path()
FROZEN_ROOT = REPO_ROOT / "paper1_workspace" / "02_frozen_results"
REPORT_PATH = REPO_ROOT / "server_reproduction_runs" / "frozen_verification_report.md"


@dataclass
class Check:
    status: str
    item: str
    detail: str


class FrozenVerifier:
    def __init__(self) -> None:
        self.checks: list[Check] = []

    def add(self, status: str, item: str, detail: str) -> None:
        self.checks.append(Check(status, item, detail))

    def pass_(self, item: str, detail: str) -> None:
        self.add("PASS", item, detail)

    def fail(self, item: str, detail: str) -> None:
        self.add("FAIL", item, detail)

    def warn(self, item: str, detail: str) -> None:
        self.add("WARN", item, detail)

    def require_path(self, rel: str, is_dir: bool | None = None) -> Path:
        path = FROZEN_ROOT / rel
        if not path.exists():
            self.fail(rel or "frozen root", "missing")
            return path
        if is_dir is True and not path.is_dir():
            self.fail(rel, "exists but is not a directory")
            return path
        if is_dir is False and not path.is_file():
            self.fail(rel, "exists but is not a file")
            return path
        self.pass_(rel or "frozen root", "exists")
        return path

    def count_json_array_or_jsonl(self, path: Path, label: str, expected: int | None = None) -> int | None:
        if not path.exists():
            return None
        try:
            text = path.read_text(encoding="utf-8")
            stripped = text.lstrip()
            if stripped.startswith("["):
                data = json.loads(text)
                count = len(data) if isinstance(data, list) else None
            else:
                count = sum(1 for line in text.splitlines() if line.strip())
        except Exception as exc:  # noqa: BLE001
            self.fail(label, f"cannot read: {type(exc).__name__}: {exc}")
            return None
        if count is None:
            self.fail(label, "JSON content is not a list and not JSONL")
            return None
        if expected is not None and count != expected:
            self.fail(label, f"expected {expected}, got {count}")
        else:
            self.pass_(label, f"{count} records")
        return count

    def read_csv(self, rel: str, min_rows: int = 1, expected_data_rows: int | None = None) -> int | None:
        path = FROZEN_ROOT / rel
        if not path.exists():
            self.fail(rel, "missing")
            return None
        try:
            with path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
                fields = f"fields={len(rows[0]) if rows else 'unknown'}"
        except Exception as exc:  # noqa: BLE001
            self.fail(rel, f"cannot read csv: {type(exc).__name__}: {exc}")
            return None
        if len(rows) < min_rows:
            self.fail(rel, f"expected at least {min_rows} data rows, got {len(rows)}")
        elif expected_data_rows is not None and len(rows) != expected_data_rows:
            self.fail(rel, f"expected {expected_data_rows} data rows, got {len(rows)}")
        else:
            self.pass_(rel, f"{len(rows)} data rows, {fields}")
        return len(rows)

    def write_report(self) -> int:
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        counts = {
            "PASS": sum(1 for c in self.checks if c.status == "PASS"),
            "WARN": sum(1 for c in self.checks if c.status == "WARN"),
            "FAIL": sum(1 for c in self.checks if c.status == "FAIL"),
        }
        lines = [
            "# Paper1 Frozen Results Verification Report",
            "",
            f"Generated: {datetime.now().isoformat(timespec='seconds')}",
            f"Project root: `{REPO_ROOT}`",
            f"Frozen root: `{FROZEN_ROOT}`",
            "",
            f"Summary: PASS={counts['PASS']} WARN={counts['WARN']} FAIL={counts['FAIL']}",
            "",
            "| Status | Item | Detail |",
            "|---|---|---|",
        ]
        for check in self.checks:
            item = check.item.replace("|", "/")
            detail = check.detail.replace("|", "/").replace("\n", " ")
            lines.append(f"| {check.status} | `{item}` | {detail} |")
        lines.append("")
        if counts["FAIL"]:
            lines.append("Result: FAIL. Fix missing or inconsistent frozen files before rerun.")
        else:
            lines.append("Result: PASS. Frozen files passed the checks implemented in this script.")
        REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
        print(f"Wrote {REPORT_PATH}")
        print(f"Summary: PASS={counts['PASS']} WARN={counts['WARN']} FAIL={counts['FAIL']}")
        return 1 if counts["FAIL"] else 0


def main() -> int:
    v = FrozenVerifier()
    if FROZEN_ROOT.exists() and FROZEN_ROOT.is_dir():
        v.pass_("paper1_workspace/02_frozen_results", "exists")
    else:
        v.fail("paper1_workspace/02_frozen_results", "missing")

    agent_a = v.require_path("official_agent_a_cache/paper1_official_agent_a_cache.json", is_dir=False)
    v.count_json_array_or_jsonl(agent_a, "official Agent A cache sample count", EXPECTED_N)

    v.read_csv("mainA_final/tab_mainA_results.csv", min_rows=1)
    v.read_csv("mainA_final/tab_mainA_budget_check.csv", min_rows=1)
    v.read_csv("mainA_final/tab_mainA_domain_results.csv", min_rows=1)
    v.read_csv("mainA_final/router_score_matrix.csv", min_rows=1, expected_data_rows=EXPECTED_N)

    maina_cache = v.require_path("mainA_final/shared_repmodel_full_call_cache.jsonl", is_dir=False)
    v.count_json_array_or_jsonl(maina_cache, "Main A shared full-call cache line count", EXPECTED_N)

    v.read_csv("mainC_final/tab_mainC_results.csv", min_rows=1)
    v.read_csv("mainC_final/tab_mainC_budget_check.csv", min_rows=1)
    for model_id in ["V1", "V2", "V3", "V4"]:
        path = v.require_path(f"mainC_final/{model_id}_full_call_cache.jsonl", is_dir=False)
        v.count_json_array_or_jsonl(path, f"Main C {model_id} full-call cache readable")

    v.read_csv("upper_lower_bounds_final/tab_upper_lower_bounds.csv", min_rows=1)

    return v.write_report()


if __name__ == "__main__":
    raise SystemExit(main())
