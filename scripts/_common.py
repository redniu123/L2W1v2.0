"""Shared path bootstrap for script entry points."""
from __future__ import annotations
import sys
from pathlib import Path
def add_repo_root_to_path() -> Path:
    root = Path(__file__).resolve().parent.parent
    for p in (root, root / "src"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return root
