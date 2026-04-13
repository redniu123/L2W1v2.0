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


_DEFAULT_BASE_URL = "https://www.lemonapi.site/v1"
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


def _extract_standalone_pool(
    text: str,
    *,
    name: str,
    marker_pattern: str,
    model_pattern: str,
    default_model: str,
    base_url: str,
) -> Optional[ProviderPool]:
    marker_match = re.search(marker_pattern, text, flags=re.IGNORECASE)
    if not marker_match:
        return None

    tail = text[marker_match.end():]
    model_match = re.search(model_pattern, tail, flags=re.IGNORECASE)
    model_name = model_match.group(1).strip() if model_match else default_model
    keys = re.findall(r"(?m)^\s*(sk-[A-Za-z0-9]+)\s*$", tail)
    if not keys:
        return None

    return ProviderPool(
        name=name,
        model_name=model_name,
        keys=keys,
        base_url=base_url,
    )


def _extract_claude_pool(text: str, base_url: str) -> Optional[ProviderPool]:
    return _extract_standalone_pool(
        text,
        name="claude_sonnet_46",
        marker_pattern=r"模型名称为[：:]\s*[A-Za-z0-9_.\-]+",
        model_pattern=r"^\s*(?:模型名称为[：:]\s*)?([A-Za-z0-9_.\-]+)",
        default_model=_DEFAULT_CLAUDE_MODEL,
        base_url=base_url,
    )


def _extract_payg_gemini_pool(text: str, base_url: str) -> Optional[ProviderPool]:
    model_match = re.search(r'MODEL_NAME_liang\s*=\s*["\']([^"\']+)["\']', text, flags=re.IGNORECASE)
    if not model_match:
        model_match = re.search(r'按量gemini分组[\s\S]{0,200}?模型名称为[：:]\s*([^\n\r]+)', text, flags=re.IGNORECASE)
    if not model_match:
        return None

    model_name = model_match.group(1).strip()
    tail = text[model_match.end():]
    keys = re.findall(r"(?m)^\s*(sk-[A-Za-z0-9]+)\s*$", tail)
    if not keys:
        return None

    return ProviderPool(
        name="gemini_payg",
        model_name=model_name,
        keys=keys,
        base_url=base_url,
    )


def _extract_gemini_1x_pool(text: str, base_url: str) -> Optional[ProviderPool]:
    model_match = re.search(r'MODEL_NAME_1x\s*=\s*["\']([^"\']+)["\']', text, flags=re.IGNORECASE)
    if not model_match:
        return None

    model_name = model_match.group(1).strip()
    tail = text[model_match.end():]
    keys = re.findall(r"(?m)^\s*(sk-[A-Za-z0-9]+)\s*$", tail)
    if not keys:
        return None

    return ProviderPool(
        name="gemini_1x",
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

    standalone_gemini_1x = _extract_gemini_1x_pool(text, base_url)
    if standalone_gemini_1x is not None:
        pools[standalone_gemini_1x.name] = standalone_gemini_1x

    claude_pool = _extract_claude_pool(text, base_url)
    if claude_pool is not None:
        pools[claude_pool.name] = claude_pool

    payg_gemini_pool = _extract_payg_gemini_pool(text, base_url)
    if payg_gemini_pool is not None:
        pools[payg_gemini_pool.name] = payg_gemini_pool

    if not pools:
        raise ValueError(f"No provider pools found in {path}")

    return pools


def get_provider_pool(pool_name: str, key_file: str | Path = "key.txt") -> ProviderPool:
    pools = load_provider_pools(key_file)
    if pool_name not in pools:
        available = ", ".join(sorted(pools))
        raise KeyError(f"Unknown provider pool '{pool_name}'. Available: {available}")
    return pools[pool_name]
