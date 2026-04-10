#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按主线 runbook 自动执行第一轮 M1-M5 正式实验。

执行顺序：
1. SH-DA++ online smoke test (100 samples)
2. BAUR-only online smoke test (100 samples)
3. SH-DA++ online full validation @ B=0.10
4. BAUR-only online full validation @ B=0.10
5. M1-M5 offline/batch formal run @ 0.10,0.20,0.30

特性：
- 串行执行，任一步失败即停止（默认）
- 每一步标准输出/错误输出落盘
- 自动生成 manifest.json 供第二天验收
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = "configs/router_config.yaml"
DEFAULT_TEST_JSONL = "data/l2w1data/test.jsonl"
DEFAULT_IMAGE_ROOT = "data/l2w1data/images"
DEFAULT_ONLINE_OUTPUT = "results/stage2_v51_online"
DEFAULT_BATCH_OUTPUT = "results/stage2_v51"
DEFAULT_AUTOMATION_OUTPUT = "results/automation_runs"


def build_steps(args: argparse.Namespace) -> List[Dict[str, Any]]:
    py = args.python_executable or sys.executable
    common = [
        py,
        "scripts/run_online_budget_control.py",
        "--config",
        args.config,
        "--test_jsonl",
        args.test_jsonl,
        "--image_root",
        args.image_root,
        "--output_dir",
        args.online_output_dir,
        "--target_budget",
        args.online_budget,
    ]
    if args.use_cache:
        common.append("--use_cache")
    if args.use_gpu:
        common.append("--use_gpu")

    smoke_shda = common + ["--strategy", "SH-DA++", "--n_samples", str(args.smoke_samples)]
    smoke_baur = common + ["--strategy", "BAUR-only", "--n_samples", str(args.smoke_samples)]
    full_shda = common + ["--strategy", "SH-DA++"]
    full_baur = common + ["--strategy", "BAUR-only"]

    batch = [
        py,
        "scripts/run_efficiency_frontier.py",
        "--config",
        args.config,
        "--test_jsonl",
        args.test_jsonl,
        "--image_root",
        args.image_root,
        "--output_dir",
        args.batch_output_dir,
        "--budgets",
        args.batch_budgets,
        "--offline_replay_budgets",
        args.offline_replay_budgets,
    ]
    if args.use_cache:
        batch.append("--use_cache")
    if args.use_gpu:
        batch.append("--use_gpu")

    return [
        {"id": "smoke_shda_online_b10", "description": "Smoke test SH-DA++ online @ B=0.10", "command": smoke_shda},
        {"id": "smoke_baur_only_online_b10", "description": "Smoke test BAUR-only online @ B=0.10", "command": smoke_baur},
        {"id": "full_shda_online_b10", "description": "Full SH-DA++ online validation @ B=0.10", "command": full_shda},
        {"id": "full_baur_only_online_b10", "description": "Full BAUR-only online validation @ B=0.10", "command": full_baur},
        {"id": "m1_m5_batch_formal", "description": "Formal M1-M5 batch run @ 0.10,0.20,0.30", "command": batch},
    ]


def run_step(step: Dict[str, Any], logs_dir: Path, timeout: int | None) -> Dict[str, Any]:
    log_path = logs_dir / f"{step['id']}.log"
    started_at = datetime.now().isoformat(timespec="seconds")
    start_ts = time.time()
    print(f"\n[RUN] {step['id']} :: {step['description']}")
    print("[CMD] " + " ".join(step["command"]))
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"# started_at: {started_at}\n")
        log_file.write("# command: " + " ".join(step["command"]) + "\n\n")
        log_file.flush()
        try:
            proc = subprocess.run(
                step["command"],
                cwd=PROJECT_ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                check=False,
            )
            return_code = proc.returncode
            error = None
        except subprocess.TimeoutExpired as exc:
            return_code = -1
            error = f"TimeoutExpired: {exc}"
            log_file.write(f"\n[ERROR] {error}\n")
    finished_at = datetime.now().isoformat(timespec="seconds")
    duration_sec = round(time.time() - start_ts, 3)
    result = {
        "id": step["id"],
        "description": step["description"],
        "command": step["command"],
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_sec": duration_sec,
        "return_code": return_code,
        "status": "success" if return_code == 0 else "failed",
        "log_path": str(log_path.relative_to(PROJECT_ROOT)),
    }
    if error:
        result["error"] = error
    print(f"[DONE] {step['id']} -> {result['status']} ({duration_sec}s), log={result['log_path']}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Automate mainline nightly formal run")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--test_jsonl", default=DEFAULT_TEST_JSONL)
    parser.add_argument("--image_root", default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--online_output_dir", default=DEFAULT_ONLINE_OUTPUT)
    parser.add_argument("--batch_output_dir", default=DEFAULT_BATCH_OUTPUT)
    parser.add_argument("--automation_output_dir", default=DEFAULT_AUTOMATION_OUTPUT)
    parser.add_argument("--python_executable", default=None, help="Python executable to use; defaults to current interpreter")
    parser.add_argument("--online_budget", default="0.10")
    parser.add_argument("--smoke_samples", type=int, default=100)
    parser.add_argument("--batch_budgets", default="0.10,0.20,0.30")
    parser.add_argument("--offline_replay_budgets", default="0.10,0.20,0.30,1.00")
    parser.add_argument("--timeout_minutes_per_step", type=int, default=0, help="0 means no timeout")
    parser.add_argument("--use_cache", dest="use_cache", action="store_true")
    parser.add_argument("--use_gpu", action="store_true", default=False)
    parser.add_argument("--no_use_cache", dest="use_cache", action="store_false")
    parser.set_defaults(use_cache=True)
    parser.add_argument("--continue_on_failure", action="store_true", default=False)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_run%H%M%S")
    run_root = PROJECT_ROOT / args.automation_output_dir / timestamp
    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    timeout = None if args.timeout_minutes_per_step <= 0 else args.timeout_minutes_per_step * 60
    steps = build_steps(args)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "project_root": str(PROJECT_ROOT),
        "run_root": str(run_root),
        "config": {
            "config": args.config,
            "test_jsonl": args.test_jsonl,
            "image_root": args.image_root,
            "online_output_dir": args.online_output_dir,
            "batch_output_dir": args.batch_output_dir,
            "online_budget": args.online_budget,
            "smoke_samples": args.smoke_samples,
            "batch_budgets": args.batch_budgets,
            "offline_replay_budgets": args.offline_replay_budgets,
            "use_cache": args.use_cache,
            "use_gpu": args.use_gpu,
            "continue_on_failure": args.continue_on_failure,
            "python_executable": args.python_executable or sys.executable,
            "timeout_minutes_per_step": args.timeout_minutes_per_step,
        },
        "steps": [],
        "overall_status": "running",
    }

    manifest_path = run_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    overall_success = True
    for step in steps:
        result = run_step(step, logs_dir, timeout)
        manifest["steps"].append(result)
        manifest["overall_status"] = "running"
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        if result["status"] != "success":
            overall_success = False
            if not args.continue_on_failure:
                break

    manifest["finished_at"] = datetime.now().isoformat(timespec="seconds")
    manifest["overall_status"] = "success" if overall_success else "failed"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n[SUMMARY] overall_status={manifest['overall_status']}")
    print(f"[SUMMARY] manifest={manifest_path.relative_to(PROJECT_ROOT)}")
    return 0 if overall_success else 1


if __name__ == "__main__":
    raise SystemExit(main())
