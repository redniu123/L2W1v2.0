from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ProviderPool:
    name: str
    model_name: str
    keys: List[str]
    base_url: str


_DEFAULT_BASE_URL = "https://new.lemonapi.site/v1"
_DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-6-thinking"


def _extract_base_url(text: str) -> str:
    match = re.search(r'BASE_URL\s*=\s*["\']([^"\']+)["\']', text)
    return match.group(1).strip() if match else _DEFAULT_BASE_URL


def _extract_model_name(text: str, variable_name: str, default: str) -> str:
    pattern = rf"{re.escape(variable_name)}\s*=\s*[\"']([^\"']+)[\"']"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else default


def _extract_key_list_block(text: str, variable_name: str) -> List[str]:
    pattern = rf"{re.escape(variable_name)}\s*=\s*\[(.*?)\]"
    match = re.search(pattern, text, flags=re.DOTALL)
    if not match:
        return []
    block = match.group(1)
    return re.findall(r'["\'](sk-[^"\']+)["\']', block)


def _extract_claude_pool(text: str, base_url: str) -> Optional[ProviderPool]:
    marker_match = re.search(
        r"模型名称为[：:]\s*([A-Za-z0-9_.\-]+)",
        text,
        flags=re.IGNORECASE,
    )
    marker_start = marker_match.end() if marker_match else -1
    model_name = marker_match.group(1).strip() if marker_match else _DEFAULT_CLAUDE_MODEL
    if marker_start < 0:
        return None

    tail = text[marker_start:]
    keys = re.findall(r"(?m)^\s*(sk-[A-Za-z0-9]+)\s*$", tail)
    if not keys:
        return None

    return ProviderPool(
        name="claude_sonnet_46",
        model_name=model_name,
        keys=keys,
        base_url=base_url,
    )


def load_provider_pools(key_file: str | Path = "key.txt") -> Dict[str, ProviderPool]:
    path = Path(key_file)
    if not path.exists():
        raise FileNotFoundError(f"Provider key file not found: {path}")

    text = path.read_text(encoding="utf-8")
    base_url = _extract_base_url(text)

    pools: Dict[str, ProviderPool] = {}

    gemini_specs = [
        ("gemini_1x", "API_KEYS_GEMINI_1X_MODEL_NAME", "API_KEYS_GEMINI_1X"),
        ("gemini_flash_1x", "API_KEYS_GEMINI_1X_FLASH_MODEL_NAME", "API_KEYS_GEMINI_FLASH_1X"),
        ("gemini_wending_3x", "API_KEYS_GEMINI_WENDING_3X_MODEL_NAME", "API_KEYS_GEMINI_WENDING_3X"),
    ]

    for pool_name, model_var, keys_var in gemini_specs:
        keys = _extract_key_list_block(text, keys_var)
        if not keys:
            continue
        pools[pool_name] = ProviderPool(
            name=pool_name,
            model_name=_extract_model_name(text, model_var, "gemini-3-flash-preview"),
            keys=keys,
            base_url=base_url,
        )

    claude_pool = _extract_claude_pool(text, base_url)
    if claude_pool is not None:
        pools[claude_pool.name] = claude_pool

    if not pools:
        raise ValueError(f"No provider pools found in {path}")

    return pools


def get_provider_pool(pool_name: str, key_file: str | Path = "key.txt") -> ProviderPool:
    pools = load_provider_pools(key_file)
    if pool_name not in pools:
        available = ", ".join(sorted(pools))
        raise KeyError(f"Unknown provider pool '{pool_name}'. Available: {available}")
    return pools[pool_name]
