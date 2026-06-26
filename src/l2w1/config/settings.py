from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast


@dataclass(frozen=True)
class L2W1Paths:
    project_root: Path
    data_root: Path
    paper1_runs: Path
    model_root: Path
    results_root: Path
    configs_root: Path


@dataclass(frozen=True)
class L2W1Secrets:
    provider_key_path: Path


@dataclass(frozen=True)
class Settings:
    paths: L2W1Paths
    secrets: L2W1Secrets


_CONFIG_KEYS = (
    "data_root",
    "paper1_runs",
    "model_root",
    "results_root",
    "configs_root",
    "provider_key_path",
)

_ENV_TO_CONFIG_KEY = {
    "L2W1_DATA_ROOT": "data_root",
    "L2W1_PAPER1_RUNS": "paper1_runs",
    "L2W1_MODEL_ROOT": "model_root",
    "L2W1_RESULTS_ROOT": "results_root",
    "L2W1_CONFIGS_ROOT": "configs_root",
    "L2W1_KEY_FILE": "provider_key_path",
}


def load_settings(
    *,
    project_root: Path | None = None,
    env: Mapping[str, str] | None = None,
    dotenv_path: Path | None = None,
    config_path: Path | None = None,
) -> Settings:
    root = _resolve_project_root(project_root)

    values: dict[str, str] = {}
    if config_path is not None:
        values.update(_load_yaml_values(config_path))

    dotenv = root / ".env" if dotenv_path is None else dotenv_path
    if dotenv.exists():
        values.update(_load_dotenv_values(dotenv))

    env_values = os.environ if env is None else env
    values.update(_load_env_values(env_values))

    data_root = _configured_path(
        project_root=root,
        value=values.get("data_root"),
        default=root / "data" / "l2w1data",
    )
    paper1_runs = _configured_path(
        project_root=root,
        value=values.get("paper1_runs"),
        default=root / "paper1_runs",
    )
    model_root = _configured_path(
        project_root=root,
        value=values.get("model_root"),
        default=root / "models",
    )
    results_root = _configured_path(
        project_root=root,
        value=values.get("results_root"),
        default=root / "results",
    )
    configs_root = _configured_path(
        project_root=root,
        value=values.get("configs_root"),
        default=root / "configs",
    )
    provider_key_path = _configured_path(
        project_root=root,
        value=values.get("provider_key_path"),
        default=root / "key.txt",
    )

    return Settings(
        paths=L2W1Paths(
            project_root=root,
            data_root=data_root,
            paper1_runs=paper1_runs,
            model_root=model_root,
            results_root=results_root,
            configs_root=configs_root,
        ),
        secrets=L2W1Secrets(provider_key_path=provider_key_path),
    )


def _resolve_project_root(project_root: Path | None) -> Path:
    if project_root is not None:
        return Path(project_root).resolve()

    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate

    msg = f"Could not find project root from {current}; no parent contains pyproject.toml"
    raise FileNotFoundError(msg)


def _load_env_values(env: Mapping[str, str]) -> dict[str, str]:
    values: dict[str, str] = {}
    for env_key, config_key in _ENV_TO_CONFIG_KEY.items():
        if env_key in env:
            values[config_key] = env[env_key]
    return values


def _load_dotenv_values(dotenv_path: Path) -> dict[str, str]:
    if _looks_like_key_file(dotenv_path):
        return {}

    values: dict[str, str] = {}
    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        key, separator, raw_value = line.partition("=")
        if separator != "=":
            continue

        key = key.strip()
        if key not in _CONFIG_KEYS:
            continue

        values[key] = _parse_dotenv_value(raw_value.strip())
    return values


def _parse_dotenv_value(value: str) -> str:
    if value and value[0] in {'"', "'"}:
        quote = value[0]
        end_quote = value.find(quote, 1)
        if end_quote >= 0:
            return value[1:end_quote]

    comment_start = value.find("#")
    if comment_start >= 0:
        return value[:comment_start].rstrip()
    return value


def _load_yaml_values(config_path: Path) -> dict[str, str]:
    if _looks_like_key_file(config_path):
        return {}

    try:
        import yaml  # type: ignore[import-not-found, import-untyped, unused-ignore]
    except ImportError:
        return {}

    raw = config_path.read_text(encoding="utf-8")
    safe_load = getattr(yaml, "safe_load", None)
    if not callable(safe_load):
        return {}

    yaml_safe_load = cast(Callable[[str], object], safe_load)
    parsed = yaml_safe_load(raw)

    if not isinstance(parsed, Mapping):
        return {}

    values: dict[str, str] = {}
    for key in _CONFIG_KEYS:
        value = parsed.get(key)
        if isinstance(value, str):
            values[key] = value
    return values


def _configured_path(*, project_root: Path, value: str | None, default: Path) -> Path:
    if value is None:
        return default

    path = Path(value)
    if path.is_absolute():
        return path
    return project_root / path


def _looks_like_key_file(path: Path) -> bool:
    return path.name == "key.txt"
