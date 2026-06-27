# Scripts Directory Guide

Date: 2026-06-26

Stage 9 keeps `scripts/` conservative: active root scripts remain in place, dev
and legacy helpers are isolated, and new work should use thin entrypoints over
`src/l2w1`.

Do not assume a script is safe to run because it is importable. Many historical
entries can load OCR/VLM models, call provider APIs, or write experiment output.

## Stage 9 Inventory

Status meanings:

- `DO NOT MOVE (imported)`: keep-set imported by other code or tests.
- `deferred - needs owner review`: experiment runners with subprocess or manual
  interdependencies; intentionally not moved in Stage 9.
- `moved`: isolated from the active script root with `git mv`.

| Category | Script | Status | Notes |
|---|---|---|---|
| paper1-core | `scripts/paper1/mainA_runner.py` | DO NOT MOVE (imported) | Main A RouterOnly replay runner. |
| paper1-core | `scripts/paper1/mainC_runner.py` | root | Main C cross-model RouterOnly runner. |
| paper1-core | `scripts/paper1/online_budget_check.py` | DO NOT MOVE (imported) | Online budget legality/check helper. |
| paper1-core | `scripts/paper1/upper_lower_bounds.py` | root | Upper/lower bound runner. |
| paper1-core | `scripts/paper1/merge_upper_lower_bounds.py` | root | Upper/lower bound merge helper. |
| paper1-core | `scripts/analysis/merge_mainc_runs.py` | root | Main C merge helper. |
| paper1-core | `scripts/experiments/efficiency_frontier.py` | DO NOT MOVE (imported) | Efficiency frontier utilities reused by tests/scripts. |
| paper1-core | `scripts/experiments/online_budget_control.py` | DO NOT MOVE (imported) | Online budget-control runner reused by other scripts. |
| reproduction | `scripts/paper1/reproduce_from_frozen.py` | root | Frozen-result reproduction package helper. |
| reproduction | `scripts/paper1/verify_frozen_results.py` | root | Frozen-result verification helper. |
| reproduction | `scripts/paper1/acceptance_check.sh` | root | Shell acceptance check helper. |
| reproduction | `scripts/paper1/rerun_commands.sh` | root | Shell rerun recipe; may launch expensive workflows. |
| analysis | `scripts/tools/evaluate.py` | root | General metrics/evaluation utility. |
| analysis | `scripts/analysis/batch1_analysis.py` | root | Paper1 batch/domain/cost analysis. |
| analysis | `scripts/analysis/gpt_case_analysis.py` | root | GPT/cross-model case analysis. |
| analysis | `scripts/tools/convert_old_gemini_to_mainc_v3.py` | root | Historical Gemini-to-MainC cache converter. |
| analysis | `scripts/tools/export_cloud_results.py` | root | Export selected run artifacts. |
| analysis | `scripts/analysis/phase_b_five_pool_analysis.py` | root | Phase B five-pool analysis. |
| visualization | `scripts/tools/visualize_master_frontier.py` | root | Frontier/scaling-law visualization. |
| visualization | `scripts/tools/visualize_results.py` | root | General result visualization. |
| data-utils | `scripts/tools/adapt_geology_data.py` | root | Geology data adaptation helper. |
| data-utils | `scripts/tools/split_dataset.py` | root | Dataset split helper. |
| data-utils | `scripts/tools/augment_hard_cases.py` | root | Hard-case augmentation helper. |
| data-utils | `scripts/tools/prepare_calibration_data.py` | DO NOT MOVE (imported) | Calibration data generation; imported by tests/dev scripts. |
| data-utils | `scripts/tools/train_calibrator.py` | root | Calibrator training helper. |
| data-utils | `scripts/analysis/top2_coverage.py` | root | Top-2 coverage analysis helper. |
| data-utils | `scripts/tools/download_vlms.py` | root | Model downloader; do not run during cleanup. |
| dev (moved) | `scripts/dev/check_server_environment.py` | moved | Deprecated server/environment check. |
| dev (moved) | `scripts/dev/smoke_test_agent_b.py` | moved | Deprecated Agent B smoke test. |
| dev (moved) | `scripts/dev/smoke_test_multi_vlm.py` | moved | Deprecated multi-VLM smoke test. |
| dev (moved) | `scripts/dev/test_efficiency_100.py` | moved | Deprecated 100-sample development run. |
| dev (moved) | `scripts/dev/test_provider_pools.py` | moved | Deprecated provider-pool API harness. |
| dev (moved) | `scripts/dev/test_single_api.py` | moved | Deprecated single Gemini API call test. |
| dev (moved) | `scripts/dev/test_stage2_integration.py` | moved | Deprecated script-style Stage 2 integration test. |
| dev (moved) | `scripts/dev/test_stage2_modules.py` | moved | Deprecated script-style Stage 2 module test. |
| legacy (moved) | `scripts/legacy/fix_frontier.py` | moved | Historical patch helper from repository root. |
| legacy (moved) | `scripts/legacy/fix_prompter.py` | moved | Historical patch helper from repository root. |
| deferred | `scripts/experiments/main_exp_b.py` | deferred - needs owner review | Experiment runner with subprocess/interdependency risk. |
| deferred | `scripts/experiments/main_exp_c.py` | deferred - needs owner review | Experiment runner with subprocess/interdependency risk. |
| deferred | `scripts/experiments/full_budget_call.py` | deferred - needs owner review | Full-budget call/cache runner. |
| deferred | `scripts/experiments/all_frontiers.py` | deferred - needs owner review | Multi-frontier experiment runner. |
| deferred | `scripts/experiments/gemini_ceiling.py` | deferred - needs owner review | Gemini ceiling experiment runner. |
| deferred | `scripts/experiments/formal_detached.py` | deferred - needs owner review | Detached formal run automation. |
| deferred | `scripts/experiments/mainline_nightly.py` | deferred - needs owner review | Nightly/mainline automation. |
| deferred | `scripts/experiments/phase_a_m5_budget_scan.py` | deferred - needs owner review | Phase A M5 budget scan runner. |
| deferred | `scripts/experiments/stage2_execution.py` | deferred - needs owner review | Stage 2 adaptation/training workflow. |
| deferred | `scripts/experiments/l2w1_pipeline.py` | deferred - needs owner review | Early end-to-end pipeline runner. |

## Thin Entrypoint Pattern

New scripts should be thin assembly layers only:

1. Build or load inputs.
2. Call tested `l2w1.*` library functions.
3. Print or write outputs at explicit user-provided paths.

Do not put reusable routing, replay, OCR, VLM, or metrics logic in scripts.
Move that logic into `src/l2w1` first, add fixture-based tests, then keep the
script as a small CLI wrapper.

See `scripts/examples/replay_offline_demo.py` for the target pattern. It uses
synthetic in-memory cache rows plus `l2w1.ocr.cache_only`,
`l2w1.vlm.cache_only`, `l2w1.replay.scoring.router_score`, and
`l2w1.replay.offline.replay_offline`. It does not load models, read real data,
or call APIs.
