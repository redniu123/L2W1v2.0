from pathlib import Path


def ensure_parent_dir(path: str | Path) -> Path:
    """Create the parent directory for a path and return it as a Path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target
