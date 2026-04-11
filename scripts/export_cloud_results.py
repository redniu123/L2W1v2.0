#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将云端结果从 results/ 导出到可提交的 cloud_result_sync/ 目录。"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_ROOT = PROJECT_ROOT / "results"
EXPORT_ROOT = PROJECT_ROOT / "cloud_result_sync"


def copy_path(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export selected cloud run artifacts into cloud_result_sync/")
    parser.add_argument("--source", action="append", required=True, help="Path relative to repo root, e.g. results/stage2_v51_online/20260410_run171751")
    parser.add_argument("--tag", default=None, help="Optional export tag; defaults to timestamp")
    parser.add_argument("--clean", action="store_true", help="Remove target export directory before copying")
    args = parser.parse_args()

    tag = args.tag or datetime.now().strftime("%Y%m%d_export%H%M%S")
    export_dir = EXPORT_ROOT / tag
    if args.clean and export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "tag": tag,
        "sources": [],
    }

    for source in args.source:
        src = PROJECT_ROOT / source
        if not src.exists():
            raise FileNotFoundError(f"Source does not exist: {source}")
        rel = src.relative_to(PROJECT_ROOT)
        dst = export_dir / rel
        copy_path(src, dst)
        manifest["sources"].append(str(rel))

    (export_dir / "export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Exported to: {export_dir}")
    for item in manifest["sources"]:
        print(f" - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
