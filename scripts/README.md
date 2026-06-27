# Scripts

Command-line entry points, grouped by purpose. Business logic lives in the
`l2w1` library (`src/l2w1/`); scripts are intended as thin orchestration layers
on top of it.

> **Safety:** many scripts load OCR/VLM models, call provider APIs, or write
> experiment output. Importing a module is safe; *running* one may not be.
> Never run these in a cleanup/CI context.

## Layout

| Directory | Purpose |
|-----------|---------|
| `paper1/` | Paper 1 reproduction chain: Main A/C runners, bound runners, online-budget check, reproduce/verify-from-frozen. |
| `experiments/` | Experiment runners: efficiency frontier, budget control/scans, per-experiment runners, automation. |
| `analysis/` | Result analysis: batch/case analysis, five-pool analysis, coverage, run merging. |
| `tools/` | Data & ops utilities: evaluate, calibration prep/training, dataset split, augmentation, VLM download, visualization, cache conversion/export. |
| `examples/` | Thin-entrypoint demos built only on `l2w1.*` (runnable on synthetic data). |
| `dev/` | Smoke/API/integration test scripts (DEPRECATED for runtime; kept for traceability). |
| `legacy/` | Historical patch helpers (DEPRECATED). |

## Conventions

- **Path bootstrap:** every entry point calls `add_repo_root_to_path()` from
  [`scripts/_common.py`](_common.py) instead of ad-hoc `sys.path` edits.
- **Reuse the library:** prefer `l2w1.config`, `l2w1.replay`, `l2w1.metrics`,
  `l2w1.routing` over re-implementing logic. See
  [`examples/replay_offline_demo.py`](examples/replay_offline_demo.py).
- **Import-stable modules** (referenced by tests / other scripts — keep their
  import paths stable): `experiments/efficiency_frontier.py`,
  `experiments/online_budget_control.py`, `tools/prepare_calibration_data.py`,
  `paper1/mainA_runner.py`, `paper1/online_budget_check.py`.

## Notes for reviewers

The `paper1/` runners keep their original `score()` / `norm()` / `replay()` /
`run_online_routeronly()` functions as **golden references** for the parity tests
in `tests/test_replay_*_parity.py`; their `main()` functions delegate to the
equivalent, parity-verified `l2w1.replay` implementations.
