#!/usr/bin/env python3
"""Repository safety guard for secrets and generated research artifacts.

The checker scans tracked files only. It reports paths and rule names, never the
matching secret text.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

FORBIDDEN_PATH_PATTERNS = (
    "key.txt",
    "GPTkey.txt",
    "GPT.config",
    ".env",
    ".env.*",
    "cloud_result_sync/*",
    "data/*",
    "models/*",
    "outputs/*",
    "paper1_runs/*",
    "paper1_runs_old/*",
    "results/*",
    "logs/*",
    "*.bat",
    "*.ps1",
    "CODEBASE_STATUS_REPORT.md",
)

TEXT_EXTENSIONS = {
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

SECRET_PATTERNS = {
    "openai_or_relay_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "anthropic_key": re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{36,}\b"),
    "private_key_block": re.compile(r"-----BEGIN (?:OPENSSH |RSA |EC |DSA )?PRIVATE KEY-----"),
    "assigned_secret": re.compile(
        r"(?i)\b(?:api[_-]?key|secret|access[_-]?token|auth[_-]?token)\b"
        r"\s*[:=]\s*[\"'][A-Za-z0-9_./+=:-]{24,}[\"']"
    ),
}


@dataclass(frozen=True)
class Finding:
    rule: str
    path: str


def _git_tracked_files(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    raw = result.stdout.decode("utf-8")
    return [path for path in raw.split("\0") if path]


def _matches_forbidden_path(path: str) -> str | None:
    normalized = path.replace("\\", "/")
    for pattern in FORBIDDEN_PATH_PATTERNS:
        if fnmatch.fnmatchcase(normalized, pattern):
            return pattern
    return None


def _is_text_candidate(path: str) -> bool:
    return Path(path).suffix.lower() in TEXT_EXTENSIONS


def _read_text(path: Path) -> str | None:
    try:
        data = path.read_bytes()
    except OSError:
        return None

    if b"\0" in data:
        return None

    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def check_repo(repo_root: Path) -> list[Finding]:
    findings: list[Finding] = []
    tracked_files = _git_tracked_files(repo_root)

    for rel_path in tracked_files:
        forbidden_pattern = _matches_forbidden_path(rel_path)
        if forbidden_pattern is not None:
            findings.append(Finding(rule=f"forbidden_path:{forbidden_pattern}", path=rel_path))
            continue

        if not _is_text_candidate(rel_path):
            continue

        text = _read_text(repo_root / rel_path)
        if text is None:
            continue

        for rule_name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(Finding(rule=f"secret_pattern:{rule_name}", path=rel_path))

    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Check tracked files for repo safety hazards.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root to inspect. Defaults to this script's repository.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    findings = check_repo(repo_root)
    if findings:
        print("Repository safety guard failed. No secret values are printed.")
        for finding in findings:
            print(f"- {finding.rule}: {finding.path}")
        return 1

    print("Repository safety guard passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
