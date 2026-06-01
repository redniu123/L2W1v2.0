#!/usr/bin/env python3
"""Read-only server migration environment check for the L2W1 paper project.

This script does not run OCR, does not call VLM APIs, and does not read key
files. It only verifies imports, path presence, and basic row counts.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


EXPECTED_TEST_ROWS = 3424
REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class CheckResult:
    status: str
    name: str
    detail: str = ""


class Checker:
    def __init__(self) -> None:
        self.results: list[CheckResult] = []

    def pass_(self, name: str, detail: str = "") -> None:
        self.results.append(CheckResult("PASS", name, detail))

    def warn(self, name: str, detail: str = "") -> None:
        self.results.append(CheckResult("WARN", name, detail))

    def fail(self, name: str, detail: str = "") -> None:
        self.results.append(CheckResult("FAIL", name, detail))

    def require_path(self, rel_path: str, *, is_dir: bool | None = None) -> None:
        path = REPO_ROOT / rel_path
        if not path.exists():
            self.fail(rel_path, "missing")
            return
        if is_dir is True and not path.is_dir():
            self.fail(rel_path, "exists but is not a directory")
            return
        if is_dir is False and not path.is_file():
            self.fail(rel_path, "exists but is not a file")
            return
        kind = "directory" if path.is_dir() else "file"
        self.pass_(rel_path, kind)

    def import_module(self, module_name: str, package_label: str | None = None) -> None:
        label = package_label or module_name
        snippet = (
            "import importlib, sys\n"
            "mod = importlib.import_module(sys.argv[1])\n"
            "print(getattr(mod, '__version__', ''))\n"
        )
        try:
            proc = subprocess.run(
                [sys.executable, "-c", snippet, module_name],
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except Exception as exc:  # noqa: BLE001 - diagnostics should catch subprocess failures
            self.warn(label, f"import check failed: {type(exc).__name__}: {exc}")
            return
        if proc.returncode == 0:
            version = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
            detail = f"imported {module_name}"
            if version:
                detail += f" {version}"
            self.pass_(label, detail)
            return
        err_lines = [line.strip() for line in (proc.stderr or proc.stdout).splitlines() if line.strip()]
        detail = err_lines[-1] if err_lines else f"returncode={proc.returncode}"
        self.warn(label, f"import failed: {detail}")

    def count_lines(self, rel_path: str, expected: int | None = None) -> None:
        path = REPO_ROOT / rel_path
        if not path.exists():
            self.fail(rel_path, "missing")
            return
        try:
            with path.open("r", encoding="utf-8") as f:
                count = sum(1 for line in f if line.strip())
        except Exception as exc:  # noqa: BLE001
            self.fail(rel_path, f"cannot read: {type(exc).__name__}: {exc}")
            return
        if expected is not None and count != expected:
            self.fail(rel_path, f"expected {expected} non-empty lines, got {count}")
        else:
            self.pass_(rel_path, f"{count} non-empty lines")

    def summarize(self) -> int:
        widths = {"status": 4, "name": 0}
        for result in self.results:
            widths["name"] = max(widths["name"], len(result.name))

        for result in self.results:
            detail = f" - {result.detail}" if result.detail else ""
            print(f"[{result.status}] {result.name.ljust(widths['name'])}{detail}")

        counts = {
            "PASS": sum(1 for r in self.results if r.status == "PASS"),
            "WARN": sum(1 for r in self.results if r.status == "WARN"),
            "FAIL": sum(1 for r in self.results if r.status == "FAIL"),
        }
        print()
        print(f"Summary: PASS={counts['PASS']} WARN={counts['WARN']} FAIL={counts['FAIL']}")
        return 1 if counts["FAIL"] else 0


def run_nvidia_smi() -> str | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def check_cuda_gpu(checker: Checker) -> None:
    smi = run_nvidia_smi()
    if smi:
        first_line = smi.splitlines()[0]
        checker.pass_("GPU visibility", f"nvidia-smi visible: {first_line}")
    else:
        checker.warn("GPU visibility", "nvidia-smi not available or no GPU visible")

    snippet = (
        "import paddle\n"
        "compiled = bool(paddle.device.is_compiled_with_cuda())\n"
        "count = int(paddle.device.cuda.device_count()) if compiled else 0\n"
        "print(f'{compiled},{count}')\n"
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", snippet],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception as exc:  # noqa: BLE001
        checker.warn("Paddle CUDA", f"could not query CUDA: {type(exc).__name__}: {exc}")
        return
    if proc.returncode != 0:
        err_lines = [line.strip() for line in (proc.stderr or proc.stdout).splitlines() if line.strip()]
        detail = err_lines[-1] if err_lines else f"returncode={proc.returncode}"
        checker.warn("Paddle CUDA", f"could not query CUDA: {detail}")
        return
    try:
        last_line = proc.stdout.strip().splitlines()[-1]
        compiled_text, count_text = last_line.split(",", 1)
        compiled = compiled_text == "True"
        cuda_count = int(count_text)
    except Exception as exc:  # noqa: BLE001
        checker.warn("Paddle CUDA", f"unexpected CUDA query output: {type(exc).__name__}: {proc.stdout.strip()}")
        return
    if compiled and cuda_count > 0:
        checker.pass_("Paddle CUDA", f"compiled with CUDA, devices={cuda_count}")
    elif compiled:
        checker.warn("Paddle CUDA", "compiled with CUDA but no CUDA device visible")
    else:
        checker.warn("Paddle CUDA", "Paddle is not compiled with CUDA")


def validate_jsonl_first_line(checker: Checker, rel_path: str) -> None:
    path = REPO_ROOT / rel_path
    if not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8") as f:
            first = next((line for line in f if line.strip()), "")
        if first:
            json.loads(first)
            checker.pass_(f"{rel_path} JSONL format", "first non-empty line is JSON")
        else:
            checker.fail(f"{rel_path} JSONL format", "file has no non-empty rows")
    except Exception as exc:  # noqa: BLE001
        checker.fail(f"{rel_path} JSONL format", f"invalid first row: {type(exc).__name__}: {exc}")


def main() -> int:
    checker = Checker()

    checker.pass_("Python version", sys.version.replace("\n", " "))
    checker.pass_("Platform", platform.platform())
    checker.pass_("Current working directory", str(Path.cwd()))
    checker.pass_("Resolved project root", str(REPO_ROOT))

    if Path.cwd().resolve() != REPO_ROOT.resolve():
        checker.warn("Working directory", "run from project root for reproducible relative paths")

    for module_name, label in [
        ("paddle", "paddle"),
        ("cv2", "cv2"),
        ("yaml", "yaml"),
        ("Levenshtein", "Levenshtein"),
        ("pandas", "pandas"),
    ]:
        checker.import_module(module_name, label)

    check_cuda_gpu(checker)

    checker.count_lines("data/l2w1data/train.jsonl")
    checker.count_lines("data/l2w1data/val.jsonl")
    checker.count_lines("data/l2w1data/test.jsonl", expected=EXPECTED_TEST_ROWS)
    validate_jsonl_first_line(checker, "data/l2w1data/test.jsonl")

    required_paths: Iterable[tuple[str, bool | None]] = [
        ("data/l2w1data/images", True),
        ("data/dicts/Geology.txt", False),
        ("data/dicts/Finance.txt", False),
        ("data/dicts/Medicine.txt", False),
        ("models/agent_a_ppocr/PP-OCRv5_server_rec_infer", True),
        ("ppocr/utils/ppocrv5_dict.txt", False),
        ("paper1_workspace/02_frozen_results/official_agent_a_cache/paper1_official_agent_a_cache.json", False),
        ("paper1_workspace/02_frozen_results/mainA_final/shared_repmodel_full_call_cache.jsonl", False),
        ("paper1_workspace/02_frozen_results/mainC_final/V1_full_call_cache.jsonl", False),
        ("paper1_workspace/02_frozen_results/mainC_final/V2_full_call_cache.jsonl", False),
        ("paper1_workspace/02_frozen_results/mainC_final/V3_full_call_cache.jsonl", False),
        ("paper1_workspace/02_frozen_results/mainC_final/V4_full_call_cache.jsonl", False),
    ]
    for rel_path, is_dir in required_paths:
        checker.require_path(rel_path, is_dir=is_dir)

    return checker.summarize()


if __name__ == "__main__":
    raise SystemExit(main())
