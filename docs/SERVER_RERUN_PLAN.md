# Server Rerun Plan

This plan defines staged server checks for the L2W1 paper1 project. Levels 0-3 are safe by default and do not call VLM APIs. Levels 4-5 are optional and should be run only after Level 0-3 pass.

All new outputs should go under `server_reproduction_runs/`.

## Level 0: Environment Check

| Item | Description |
|---|---|
| Purpose | Confirm Python, import dependencies, current working directory, basic GPU/CUDA visibility, data paths, model paths, and frozen cache paths. |
| Input files | Project root, `requirements*.txt`, `data/l2w1data/*.jsonl`, `data/dicts/*.txt`, `models/agent_a_ppocr/PP-OCRv5_server_rec_infer/`, frozen cache files. |
| Output directory | Console only. No result files are required. |
| Command | `python scripts/check_server_environment.py` |
| Calls API | No. |
| Risk | Low. It imports packages but does not run OCR or VLM. Paddle import may emit CUDA warnings. |
| Acceptance standard | No FAIL entries. WARN entries are acceptable only if they match the intended server mode, such as GPU not available for frozen-only checks. |

## Level 1: Frozen Results Integrity Check

| Item | Description |
|---|---|
| Purpose | Verify `paper1_workspace/02_frozen_results/` exists and key CSV/JSONL/cache files are readable with expected sample counts. |
| Input files | `paper1_workspace/02_frozen_results/**` |
| Output directory | `server_reproduction_runs/` |
| Command | `python scripts/verify_paper1_frozen_results.py` |
| Calls API | No. |
| Risk | Low. It reads frozen result files only. |
| Acceptance standard | `server_reproduction_runs/frozen_verification_report.md` reports no FAIL entries; official Agent A cache, router score matrix, and Main A full-call cache each contain 3424 samples/rows. |

## Level 2: Reproduce Core Tables From Frozen Files

| Item | Description |
|---|---|
| Purpose | Build a minimal server reproduction artifact by reading frozen CSV/JSONL files and copying/summarizing core paper tables into a new output folder. |
| Input files | `mainA_final/tab_mainA_results.csv`, `mainC_final/tab_mainC_results.csv`, `mainA_final/tab_mainA_budget_check.csv`, `mainC_final/tab_mainC_budget_check.csv`, `upper_lower_bounds_final/tab_upper_lower_bounds.csv` |
| Output directory | `server_reproduction_runs/from_frozen/` |
| Command | `python scripts/reproduce_paper1_from_frozen.py` |
| Calls API | No. |
| Risk | Low. It does not recompute OCR/VLM outputs and does not overwrite frozen files. |
| Acceptance standard | Output CSV files and `reproduction_summary.md` exist; row counts match frozen source files; copied vs recomputed status is clearly documented. |

## Level 3: Replay Main A / Main C From Existing Caches

| Item | Description |
|---|---|
| Purpose | Reuse Agent A cache and VLM full-call cache to rerun offline budget replay into a new server output directory, without calling APIs. |
| Input files | Official Agent A cache, Main A `shared_repmodel_full_call_cache.jsonl`, Main C `V1..V4_full_call_cache.jsonl`, router config, test split. |
| Output directory | `server_reproduction_runs/mainA_replay/` and `server_reproduction_runs/mainC_replay/` |
| Command | Not added in this stage. Recommended future wrapper: `python scripts/replay_paper1_from_cache.py --output_root server_reproduction_runs/replay_from_cache` |
| Calls API | No, if implemented as cache-only replay. |
| Risk | Medium. Existing `paper1_mainA_runner.py` and `paper1_mainC_runner.py` can call Agent B unless carefully constrained; use a future cache-only wrapper rather than direct runner invocation. |
| Acceptance standard | Generated `tab_mainA_results.csv` and `tab_mainC_results.csv` match frozen key metrics within exact or documented tolerance; output stays outside `paper1_workspace/02_frozen_results/`. |

Current note: Level 3 should not be run by directly calling the original runners unless a cache-only/no-API wrapper has been added and reviewed.

## Level 4: Small-Sample Agent A OCR Test

| Item | Description |
|---|---|
| Purpose | Confirm the server can load PP-OCRv5 Agent A and OCR a very small number of samples. |
| Input files | `data/l2w1data/test.jsonl`, `data/l2w1data/images/`, `models/agent_a_ppocr/PP-OCRv5_server_rec_infer/`, `ppocr/utils/ppocrv5_dict.txt`. |
| Output directory | `server_reproduction_runs/agent_a_smoke/` |
| Command | Not added in this stage. Recommended future wrapper: `python scripts/check_agent_a_ocr_smoke.py --n_samples 5 --output_dir server_reproduction_runs/agent_a_smoke` |
| Calls API | No. |
| Risk | Medium. Loads Paddle model and may require GPU/CUDA/Paddle compatibility. |
| Acceptance standard | Script processes selected images, writes OCR smoke report, and does not modify any frozen cache. |

## Level 5: Full Rerun

| Item | Description |
|---|---|
| Purpose | Fully rerun Agent A OCR, VLM correction, Main A/Main C replay, baseline analysis, bounds, and figure/table generation. |
| Input files | Full data, models, configs, approved API credentials if VLM API is used, or approved local VLM weights if offline. |
| Output directory | `server_reproduction_runs/full_rerun/YYYYMMDD_runHHMMSS/` |
| Command | To be defined only after Levels 0-4 pass. |
| Calls API | Yes if using Gemini/GPT/Claude providers; no if using only local VLM and frozen cache. |
| Risk | High. Expensive, non-deterministic API behavior, key management risk, model version drift, long runtime, possible server quota issues. |
| Acceptance standard | Full provenance recorded: command, git status, config snapshot, environment snapshot, model versions, API/provider version/date, prompt version, failure rates, and comparison to frozen paper1 metrics. |

Level 5 should not be started until API usage is approved in writing and real keys are provided through company-approved secret management. Never paste keys into scripts or logs.
