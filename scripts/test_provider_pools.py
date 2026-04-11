#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provider pool accuracy-first test harness for Gemini/Claude relay pools."""

from __future__ import annotations

import argparse
import base64
import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from modules.vlm_expert.provider_pools import ProviderPool, get_provider_pool, load_provider_pools


DEFAULT_IMAGE = PROJECT_ROOT / "data" / "l2w1data" / "images" / "FinP0001_L004.jpg"


def build_text_payload(pool: ProviderPool, prompt: str, max_tokens: int) -> Dict[str, Any]:
    return {
        "model": pool.model_name,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }


def build_image_payload(pool: ProviderPool, prompt: str, image_path: Path, max_tokens: int) -> Dict[str, Any]:
    image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return {
        "model": pool.model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
    }


def _extract_text(data: Dict[str, Any]) -> str:
    if not isinstance(data, dict):
        return ""
    choices = data.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts).strip()
    return str(content).strip()


def single_request(pool: ProviderPool, key: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        resp = requests.post(
            f"{pool.base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
        latency = time.perf_counter() - started
        text = ""
        try:
            data = resp.json()
            text = _extract_text(data)
        except Exception:
            data = None
        if resp.status_code == 200:
            return {
                "status": "ok",
                "http_status": 200,
                "latency": latency,
                "response_text": text,
                "error": "",
                "key_suffix": key[-6:],
            }
        return {
            "status": f"http_{resp.status_code}",
            "http_status": resp.status_code,
            "latency": latency,
            "response_text": text,
            "error": resp.text[:300],
            "key_suffix": key[-6:],
        }
    except requests.exceptions.Timeout:
        return {
            "status": "timeout",
            "http_status": None,
            "latency": time.perf_counter() - started,
            "response_text": "",
            "error": "timeout",
            "key_suffix": key[-6:],
        }
    except Exception as e:
        return {
            "status": type(e).__name__,
            "http_status": None,
            "latency": time.perf_counter() - started,
            "response_text": "",
            "error": str(e)[:300],
            "key_suffix": key[-6:],
        }


def run_phase(pool: ProviderPool, phase_name: str, keys: List[str], payload_factory, timeout: int, requests_per_key: int, concurrency: int) -> Dict[str, Any]:
    jobs = []
    for key in keys:
        for _ in range(requests_per_key):
            jobs.append((key, payload_factory()))
    random.shuffle(jobs)

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
        futures = [executor.submit(single_request, pool, key, payload, timeout) for key, payload in jobs]
        for future in as_completed(futures):
            results.append(future.result())

    ok_results = [r for r in results if r["status"] == "ok"]
    latencies = [r["latency"] for r in ok_results]
    error_breakdown: Dict[str, int] = {}
    for r in results:
        error_breakdown[r["status"]] = error_breakdown.get(r["status"], 0) + 1

    p95 = 0.0
    if latencies:
        ordered = sorted(latencies)
        p95 = ordered[max(0, int(len(ordered) * 0.95) - 1)]

    return {
        "phase": phase_name,
        "n_requests": len(results),
        "success_rate": round(len(ok_results) / len(results), 6) if results else 0.0,
        "timeout_rate": round(sum(1 for r in results if r["status"] == "timeout") / len(results), 6) if results else 0.0,
        "avg_latency_sec": round(statistics.mean(latencies), 4) if latencies else 0.0,
        "p95_latency_sec": round(p95, 4),
        "error_breakdown": error_breakdown,
        "sample_results": results[: min(12, len(results))],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Accuracy-first provider pool tester")
    parser.add_argument("--key_file", default="key.txt")
    parser.add_argument("--pool", default="all", help="Pool name or 'all'")
    parser.add_argument("--sample_keys", type=int, default=5)
    parser.add_argument("--light_timeout", type=int, default=20)
    parser.add_argument("--real_timeout", type=int, default=45)
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    parser.add_argument("--output", default="results/provider_pool_tests")
    args = parser.parse_args()

    all_pools = load_provider_pools(args.key_file)
    selected = all_pools if args.pool == "all" else {args.pool: get_provider_pool(args.pool, args.key_file)}

    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = (PROJECT_ROOT / image_path).resolve()

    output_dir = PROJECT_ROOT / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image_path": str(image_path),
        "pools": {},
    }

    for pool_name, pool in selected.items():
        keys = pool.keys[: min(args.sample_keys, len(pool.keys))]
        light_prompt = "Reply with exactly OK"
        real_prompt = "你是OCR纠错助手。请阅读图像内容，只输出图中单行文本，不要解释。"

        phase_light = run_phase(
            pool,
            "light_text",
            keys,
            lambda: build_text_payload(pool, light_prompt, max_tokens=8),
            timeout=args.light_timeout,
            requests_per_key=1,
            concurrency=1,
        )
        phase_real = run_phase(
            pool,
            "real_image",
            keys,
            lambda: build_image_payload(pool, real_prompt, image_path, max_tokens=64),
            timeout=args.real_timeout,
            requests_per_key=1,
            concurrency=1,
        )
        phase_concurrent = run_phase(
            pool,
            "concurrent_image",
            keys,
            lambda: build_image_payload(pool, real_prompt, image_path, max_tokens=64),
            timeout=args.real_timeout,
            requests_per_key=2,
            concurrency=min(4, max(1, len(keys))),
        )

        report["pools"][pool_name] = {
            "model_name": pool.model_name,
            "key_count_total": len(pool.keys),
            "key_count_tested": len(keys),
            "phases": [phase_light, phase_real, phase_concurrent],
        }

    out_path = output_dir / f"provider_pool_test_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved report to: {out_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
