#!/usr/bin/env python3
"""Create a minimal paper1 reproduction package from frozen results only.

This script is intentionally read-only with respect to frozen results. It does
not run OCR, does not call VLM APIs, and writes only under
server_reproduction_runs/from_frozen/.
"""

from __future__ import annotations

import csv
import shutil
from datetime import datetime
from pathlib import Path

from scripts._common import add_repo_root_to_path

REPO_ROOT = add_repo_root_to_path()
FROZEN_ROOT = REPO_ROOT / "paper1_workspace" / "02_frozen_results"
OUT_DIR = REPO_ROOT / "server_reproduction_runs" / "from_frozen"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def copy_csv(src: Path, dst: Path) -> int:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return len(read_csv(dst))


def safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_budget_summary(maina_rows: list[dict[str, str]], mainc_rows: list[dict[str, str]], bounds_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    official_budgets = {0.10, 0.20, 0.30}

    for row in maina_rows:
        budget = round(safe_float(row.get("budget", "")), 2)
        if budget in official_budgets:
            summary.append(
                {
                    "source": "copied_mainA_frozen",
                    "system": row.get("router_name", ""),
                    "model": row.get("agentB_model", ""),
                    "budget": f"{budget:.2f}",
                    "actual_call_rate": row.get("actual_call_rate", ""),
                    "CER": row.get("CER", ""),
                    "AER": row.get("AER", ""),
                    "n_valid": row.get("n_valid", ""),
                }
            )

    for row in mainc_rows:
        budget = round(safe_float(row.get("budget", "")), 2)
        if budget in official_budgets:
            summary.append(
                {
                    "source": "copied_mainC_frozen",
                    "system": row.get("router_name", ""),
                    "model": row.get("model_name", ""),
                    "budget": f"{budget:.2f}",
                    "actual_call_rate": row.get("actual_call_rate", ""),
                    "CER": row.get("CER", ""),
                    "AER": row.get("AER", ""),
                    "n_valid": row.get("n_valid", ""),
                }
            )

    for row in bounds_rows:
        summary.append(
            {
                "source": "copied_bounds_frozen",
                "system": row.get("system_name", ""),
                "model": row.get("backend_label", ""),
                "budget": "NA",
                "actual_call_rate": "NA",
                "CER": row.get("CER", ""),
                "AER": "NA",
                "n_valid": row.get("n_samples", ""),
            }
        )

    return summary


def markdown_table(rows: list[dict[str, object]], limit: int = 20) -> list[str]:
    if not rows:
        return ["No rows."]
    fields = list(rows[0].keys())
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows[:limit]:
        lines.append("| " + " | ".join(str(row.get(f, "")).replace("|", "/") for f in fields) + " |")
    if len(rows) > limit:
        lines.append(f"| ... | ... | ... | ... | ... | ... | ... | ... |")
    return lines


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    src_maina = FROZEN_ROOT / "mainA_final" / "tab_mainA_results.csv"
    src_mainc = FROZEN_ROOT / "mainC_final" / "tab_mainC_results.csv"
    src_maina_budget = FROZEN_ROOT / "mainA_final" / "tab_mainA_budget_check.csv"
    src_mainc_budget = FROZEN_ROOT / "mainC_final" / "tab_mainC_budget_check.csv"
    src_bounds = FROZEN_ROOT / "upper_lower_bounds_final" / "tab_upper_lower_bounds.csv"

    copied = {
        "tab_mainA_results": copy_csv(src_maina, OUT_DIR / "copied_or_reproduced_tab_mainA_results.csv"),
        "tab_mainC_results": copy_csv(src_mainc, OUT_DIR / "copied_or_reproduced_tab_mainC_results.csv"),
    }

    maina_rows = read_csv(src_maina)
    mainc_rows = read_csv(src_mainc)
    maina_budget_rows = read_csv(src_maina_budget)
    mainc_budget_rows = read_csv(src_mainc_budget)
    bounds_rows = read_csv(src_bounds)
    budget_summary = build_budget_summary(maina_budget_rows, mainc_budget_rows, bounds_rows)
    write_csv(OUT_DIR / "copied_or_reproduced_budget_check_summary.csv", budget_summary)

    lines = [
        "# Paper1 Minimal Reproduction From Frozen Results",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Project root: `{REPO_ROOT}`",
        f"Frozen root: `{FROZEN_ROOT}`",
        f"Output directory: `{OUT_DIR}`",
        "",
        "## Safety",
        "",
        "- OCR was not run.",
        "- VLM APIs were not called.",
        "- API key files were not read.",
        "- Frozen result files were not overwritten.",
        "",
        "## Output Files",
        "",
        "| Output | Source | Mode | Data rows |",
        "|---|---|---|---:|",
        f"| `copied_or_reproduced_tab_mainA_results.csv` | `{src_maina.relative_to(REPO_ROOT)}` | copied | {copied['tab_mainA_results']} |",
        f"| `copied_or_reproduced_tab_mainC_results.csv` | `{src_mainc.relative_to(REPO_ROOT)}` | copied | {copied['tab_mainC_results']} |",
        f"| `copied_or_reproduced_budget_check_summary.csv` | Main A/Main C budget checks + bounds | summarized from copied frozen CSV | {len(budget_summary)} |",
        "",
        "## Reproduction Status",
        "",
        "| Component | Status | Note |",
        "|---|---|---|",
        "| Main A full dense results | copied | No metric recomputation in this stage. |",
        "| Main C cross-model results | copied | No metric recomputation in this stage. |",
        "| Budget check summary | summarized | Combines official budget rows and bounds rows from frozen CSV files. |",
        "| CER / AER recomputation | not recomputed | Planned for cache-only replay Level 3. |",
        "| Agent A OCR | not run | Planned for optional Level 4. |",
        "| VLM correction | not run | API calls are prohibited in this stage. |",
        "",
        "## Budget Summary Preview",
        "",
    ]
    lines.extend(markdown_table(budget_summary, limit=24))
    lines.extend(
        [
            "",
            "## Next Step",
            "",
            "If this script passes on the company server, proceed to Level 3 only after adding or reviewing a cache-only replay wrapper that writes outside frozen results.",
            "",
        ]
    )
    summary_path = OUT_DIR / "reproduction_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {OUT_DIR}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
