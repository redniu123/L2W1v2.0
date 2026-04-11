#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Detached mainline formal runner for unattended server execution."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = "configs/router_config.yaml"
DEFAULT_TEST_JSONL = "data/l2w1data/test.jsonl"
DEFAULT_IMAGE_ROOT = "data/l2w1data/images"
DEFAULT_BATCH_OUTPUT = "results/stage2_v51"
DEFAULT_ONLINE_OUTPUT = "results/stage2_v51_online"
DEFAULT_AUTOMATION_OUTPUT = "results/automation_runs"
DEFAULT_SYNC_ROOT = "cloud_result_sync"


def cache_guard(cache_path: Path, test_jsonl: Path) -> Dict[str, Any]:
    test_count = sum(1 for line in test_jsonl.read_text(encoding="utf-8").splitlines() if line.strip())
    if not cache_path.exists():
        return {
            "cache_exists": False,
            "cache_count": -1,
            "test_count": test_count,
            "is_complete": False,
        }
    cache_count = len(json.loads(cache_path.read_text(encoding="utf-8")))
    return {
        "cache_exists": True,
        "cache_count": cache_count,
        "test_count": test_count,
        "is_complete": cache_count == test_count,
    }


def build_steps(args: argparse.Namespace) -> List[Dict[str, Any]]:
    py = args.python_executable or sys.executable
    steps: List[Dict[str, Any]] = []

    if args.cleanup_failed_runs:
        root = str((PROJECT_ROOT / args.batch_output_dir).resolve()).replace("\\", "/")
        steps.append(
            {
                "id": "cleanup_failed_runs",
                "description": "Remove known failed run directories before formal execution",
                "command": [
                    py,
                    "-c",
                    (
                        "from pathlib import Path; import shutil; "
                        f"root=Path(r'{root}'); "
                        "targets=['20260411_run140204','20260411_run140237']; "
                        "removed=[]; "
                        "[shutil.rmtree(root/t, ignore_errors=True) or removed.append(t) for t in targets if (root/t).exists()]; "
                        "print('removed', removed)"
                    ),
                ],
            }
        )

    rebuild_cache = [
        py, "scripts/run_efficiency_frontier.py",
        "--config", args.config,
        "--test_jsonl", args.test_jsonl,
        "--image_root", args.image_root,
        "--output_dir", args.batch_output_dir,
        "--rebuild_cache",
        "--budgets", args.cache_rebuild_budget,
        "--strategies", args.cache_rebuild_strategy,
        "--skip_offline_replay",
    ]
    if args.use_gpu:
        rebuild_cache.append("--use_gpu")

    sanity = [
        py, "scripts/run_efficiency_frontier.py",
        "--config", args.config,
        "--test_jsonl", args.test_jsonl,
        "--image_root", args.image_root,
        "--output_dir", args.batch_output_dir,
        "--n_samples", str(args.sanity_samples),
        "--budgets", args.batch_budgets,
        "--strategies", args.router_strategies,
        "--skip_offline_replay",
        "--use_cache",
    ]
    if args.use_gpu:
        sanity.append("--use_gpu")

    formal_batch = [
        py, "scripts/run_efficiency_frontier.py",
        "--config", args.config,
        "--test_jsonl", args.test_jsonl,
        "--image_root", args.image_root,
        "--output_dir", args.batch_output_dir,
        "--budgets", args.batch_budgets,
        "--strategies", args.router_strategies,
        "--use_cache",
    ]
    if args.use_gpu:
        formal_batch.append("--use_gpu")
    if args.skip_offline_replay_in_formal:
        formal_batch.append("--skip_offline_replay")
    else:
        formal_batch.extend(["--offline_replay_budgets", args.offline_replay_budgets])

    online = [
        py, "scripts/run_online_budget_control.py",
        "--config", args.config,
        "--test_jsonl", args.test_jsonl,
        "--image_root", args.image_root,
        "--output_dir", args.online_output_dir,
        "--strategy", args.system_strategy,
        "--target_budget", args.online_budget,
        "--use_cache",
    ]
    if args.use_gpu:
        online.append("--use_gpu")

    export_command = [
        py, "scripts/export_cloud_results.py",
        "--tag", args.sync_tag,
        "--source", args.batch_output_dir,
        "--source", args.online_output_dir,
        "--source", f"{args.automation_output_dir}/{args.run_stamp}",
    ]

    steps.extend(
        [
            {"id": "rebuild_full_test_cache_if_needed", "description": "Rebuild full Agent A cache when incomplete", "command": rebuild_cache, "requires_incomplete_cache": True},
            {"id": "m1_m3_sanity_router_compare", "description": f"Sanity router comparison on {args.sanity_samples} samples after cache validation", "command": sanity},
            {"id": "m1_m3_formal_router_compare", "description": "Formal full router comparison on the complete test set", "command": formal_batch},
            {"id": "m4_online_system_validation", "description": f"Online system validation for {args.system_strategy} @ B={args.online_budget}", "command": online},
            {"id": "export_to_cloud_result_sync", "description": "Export formal outputs into cloud_result_sync", "command": export_command},
        ]
    )
    return steps


def run_step(step: Dict[str, Any], logs_dir: Path, timeout: int | None) -> Dict[str, Any]:
    log_path = logs_dir / f"{step['id']}.log"
    started_at = datetime.now().isoformat(timespec="seconds")
    start_ts = time.time()
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"# started_at: {started_at}\n")
        log_file.write("# command: " + " ".join(step["command"]) + "\n\n")
        log_file.flush()
        try:
            proc = subprocess.run(step["command"], cwd=PROJECT_ROOT, stdout=log_file, stderr=subprocess.STDOUT, text=True, timeout=timeout, check=False)
            return_code, error = proc.returncode, None
        except subprocess.TimeoutExpired as exc:
            return_code, error = -1, f"TimeoutExpired: {exc}"
            log_file.write(f"\n[ERROR] {error}\n")
    result = {
        "id": step["id"],
        "description": step["description"],
        "command": step["command"],
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "duration_sec": round(time.time() - start_ts, 3),
        "return_code": return_code,
        "status": "success" if return_code == 0 else "failed",
        "log_path": str(log_path.relative_to(PROJECT_ROOT)),
    }
    if error:
        result["error"] = error
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Detached mainline formal runner")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--test_jsonl", default=DEFAULT_TEST_JSONL)
    parser.add_argument("--image_root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--batch_output_dir", default=DEFAULT_BATCH_OUTPUT)
    parser.add_argument("--online_output_dir", default=DEFAULT_ONLINE_OUTPUT)
    parser.add_argument("--automation_output_dir", default=DEFAULT_AUTOMATION_OUTPUT)
    parser.add_argument("--sync_root", default=DEFAULT_SYNC_ROOT)
    parser.add_argument("--sync_tag", default=None)
    parser.add_argument("--python_executable", default=None)
    parser.add_argument("--router_strategies", default="GCR,BAUR,DAR")
    parser.add_argument("--batch_budgets", default="0.10,0.20,0.30")
    parser.add_argument("--offline_replay_budgets", default="0.10,0.20,0.30,1.00")
    parser.add_argument("--skip_offline_replay_in_formal", action="store_true", default=False)
    parser.add_argument("--sanity_samples", type=int, default=100)
    parser.add_argument("--system_strategy", default="SH-DA++", choices=["BAUR-only", "SH-DA++"])
    parser.add_argument("--online_budget", default="0.10")
    parser.add_argument("--cache_rebuild_strategy", default="GCR")
    parser.add_argument("--cache_rebuild_budget", default="0.10")
    parser.add_argument("--timeout_minutes_per_step", type=int, default=0)
    parser.add_argument("--cleanup_failed_runs", action="store_true", default=True)
    parser.add_argument("--use_gpu", action="store_true", default=False)
    parser.add_argument("--continue_on_failure", action="store_true", default=False)
    args = parser.parse_args()

    args.run_stamp = datetime.now().strftime("%Y%m%d_run%H%M%S")
    args.sync_tag = args.sync_tag or f"{args.run_stamp}_formal_bundle"
    run_root = PROJECT_ROOT / args.automation_output_dir / args.run_stamp
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    cache_info = cache_guard((PROJECT_ROOT / args.batch_output_dir / "agent_a_cache.json"), PROJECT_ROOT / args.test_jsonl)
    timeout = None if args.timeout_minutes_per_step <= 0 else args.timeout_minutes_per_step * 60
    steps = build_steps(args)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "run_root": str(run_root),
        "sync_target": f"{args.sync_root}/{args.sync_tag}",
        "config": vars(args),
        "cache_guard": {"before_run": cache_info},
        "steps": [],
        "overall_status": "running",
    }
    manifest_path = run_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    overall_success = True
    for step in steps:
        if step.get("requires_incomplete_cache") and cache_info.get("is_complete"):
            manifest["steps"].append({"id": step["id"], "description": step["description"], "status": "skipped", "reason": "cache_already_complete"})
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            continue

        result = run_step(step, logs_dir, timeout)
        manifest["steps"].append(result)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if result["status"] != "success":
            overall_success = False
            if not args.continue_on_failure:
                break

        if step["id"] == "rebuild_full_test_cache_if_needed":
            cache_info = cache_guard((PROJECT_ROOT / args.batch_output_dir / "agent_a_cache.json"), PROJECT_ROOT / args.test_jsonl)
            manifest["cache_guard"]["after_rebuild"] = cache_info
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            if not cache_info.get("is_complete"):
                overall_success = False
                manifest["steps"].append({"id": "cache_guard_halt", "description": "Stop because cache is still incomplete after rebuild", "status": "failed", "details": cache_info})
                manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
                break

    if not cache_info.get("is_complete") and all(step["id"] != "rebuild_full_test_cache_if_needed" or step.get("requires_incomplete_cache") for step in steps):
        overall_success = False

    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["overall_status"] = "success" if overall_success and cache_info.get("is_complete") else "failed"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[SUMMARY] overall_status={manifest['overall_status']}")
    print(f"[SUMMARY] manifest={manifest_path.relative_to(PROJECT_ROOT)}")
    print(f"[SUMMARY] sync_target={args.sync_root}/{args.sync_tag}")
    return 0 if manifest["overall_status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
