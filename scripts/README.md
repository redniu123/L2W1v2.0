# Scripts Directory Guide

This directory contains paper1 final scripts, legacy experiment scripts, smoke/test scripts, API/debug scripts, and utility scripts. Do not assume every script is safe to run on a company server.

## Safe Server Migration Checks

These scripts were added for server migration and are read-only with respect to paper frozen results:

| Script | Purpose | Calls API | Writes output |
|---|---|---|---|
| `check_server_environment.py` | Check Python, imports, GPU visibility, data paths, model paths, and frozen cache presence. | No | Console only |
| `verify_paper1_frozen_results.py` | Verify frozen result integrity and expected row counts. | No | `server_reproduction_runs/frozen_verification_report.md` |
| `reproduce_paper1_from_frozen.py` | Copy/summarize core frozen CSV files into a minimal server reproduction package. | No | `server_reproduction_runs/from_frozen/` |

## Paper1 Final Strongly Related Scripts

These scripts define or derive the final paper1 experiment results. They should not be edited casually.

| Script | Role | Caution |
|---|---|---|
| `paper1_mainA_runner.py` | Main A RouterOnly dense budget replay. | May call Agent B if not constrained; do not run during Level 0-3 checks. |
| `paper1_mainC_runner.py` | Cross-model replay under a fixed router. | May call or reuse VLM full-call cache depending on cache availability and arguments. |
| `paper1_batch1_analysis.py` | Random/MinConf/domain/cost derived analyses. | Defaults may point to old run paths; pass explicit frozen/final paths. |
| `paper1_gpt_case_analysis.py` | GPT negative optimization and casebook analysis. | Defaults may point to old run paths; pass explicit final paths. |
| `paper1_upper_lower_bounds.py` | A-only, B-only, and A+B bounds. | Can call Gemini/local VLM for B-only recognition. Do not run without approval. |
| `paper1_merge_upper_lower_bounds.py` | Merge bounds base and patch run. | File-producing utility; write to a new output directory only. |
| `paper1_online_budget_check.py` | Online legality/budget validation using cache. | Reads cache; ensure it does not regenerate Agent A cache in frozen paths. |

## Figure and Table Scripts

Most final figure/table scripts live outside this directory:

```text
paper1_workspace/03_figures_tables/
```

Use `paper1_workspace/03_figures_tables/fig_source_map.md` and `table_source_map.md` to identify final figure/table sources.

## Legacy or Prototype Experiment Scripts

These scripts belong to earlier SH-DA++/stage2/exA/exB development. Keep them for traceability, but do not use them as paper1 final entry points unless you know exactly which result they reproduce.

| Script | Notes |
|---|---|
| `run_l2w1_pipeline.py` | Early pipeline entry. |
| `run_efficiency_frontier.py` | v5.1 efficiency frontier grand loop; paper1 final runners reuse utility functions from it. |
| `run_all_frontiers.py` | Multi-frontier experiment runner. |
| `run_formal_detached.py` | Formal detached experiment runner. |
| `run_main_exp_b.py` | exB-era runner. |
| `run_main_exp_c.py` | exC-era runner. |
| `run_mainline_nightly.py` | Nightly/mainline automation style runner. |
| `run_online_budget_control.py` | Online controller experiment runner. |
| `run_phase_a_m5_budget_scan.py` | Phase A / M5 budget scan. |
| `run_phase_b_five_pool_analysis.py` | Phase B provider-pool analysis. |
| `run_stage2_execution.py` | Stage2 execution script. |
| `run_gemini_ceiling.py` | Gemini ceiling experiment; likely API-calling. |
| `run_full_budget_call.py` | Full budget call script; likely API-calling. |

## Data, Calibration, and Utility Scripts

| Script | Role |
|---|---|
| `adapt_geology_data.py` | Geology data adaptation. |
| `augment_hard_cases.py` | Hard-case augmentation. |
| `check_top2_coverage.py` | Top-2 coverage analysis. |
| `prepare_calibration_data.py` | Calibration dataset preparation. |
| `train_calibrator.py` | Calibrated scorer training. |
| `split_dataset.py` | Dataset split helper. |
| `merge_mainc_runs.py` | Merge split Main C runs. |
| `convert_old_gemini_to_mainc_v3.py` | Convert old Gemini cache to Main C V3 cache format. |
| `export_cloud_results.py` | Copy/export cloud result files. |
| `visualize_master_frontier.py` | Legacy visualization. |
| `visualize_results.py` | General/legacy visualization. |
| `evaluate.py` | General evaluation utilities; paper1 final scripts also implement local metrics. |

## Debug, Test, and Smoke Scripts

These are useful during development but should not be run as paper reproduction commands:

| Script | Caution |
|---|---|
| `_test_api.py` | Debug script with historical hard-coded API token risk. Do not run or publish. |
| `_debug_agentb.py` | Debug script with local path assumptions. |
| `test_single_api.py` | API test; do not run without approval. |
| `test_provider_pools.py` | Reads provider/key configuration; do not run on shared logs. |
| `smoke_test_agent_b.py` | Agent B smoke test; may load/call VLM depending on config. |
| `smoke_test_multi_vlm.py` | Multi-VLM smoke test; may load/call VLM. |
| `test_efficiency_100.py` | Development test. |
| `test_stage2_modules.py` | Stage2 test. |
| `test_stage2_integration.py` | Stage2 integration test. |

## Rule of Thumb

For company server migration:

1. Run only `check_server_environment.py`.
2. Then run only `verify_paper1_frozen_results.py`.
3. Then run only `reproduce_paper1_from_frozen.py`.
4. Do not run API/debug scripts.
5. Do not paste, print, or commit key file contents.
