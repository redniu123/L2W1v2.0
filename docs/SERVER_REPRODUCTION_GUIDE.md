# Server Reproduction Guide

This guide describes how to move the L2W1 paper project to a company server and run safe, read-only checks before any expensive or model-calling experiment.

## Goals

The server-side reproduction goal for this stage is limited to:

- confirm the Python environment can import required packages;
- confirm data, model directories, dictionaries, frozen caches, and frozen results are present;
- confirm the frozen result files are readable and have expected row counts;
- generate server-side verification reports under `server_reproduction_runs/`;
- avoid modifying `paper1_workspace/02_frozen_results/`;
- avoid calling any VLM API by default.

This stage is not a full rerun of the paper. It is a migration and integrity check stage.

## Safety Rules

- Do not delete existing files.
- Do not move or rename files in `data/`, `models/`, `modules/`, `scripts/`, `paper1_workspace/02_frozen_results/`, or `paper1_workspace/03_figures_tables/`.
- Do not edit the core experiment logic in:
  - `scripts/paper1_mainA_runner.py`
  - `scripts/paper1_mainC_runner.py`
  - `scripts/paper1_batch1_analysis.py`
  - `scripts/paper1_gpt_case_analysis.py`
  - `scripts/paper1_upper_lower_bounds.py`
- Do not read, print, copy, or commit real API keys.
- Do not call Gemini, GPT, Claude, Qwen API, or any external VLM service during Level 0-3 checks.
- Write all new server outputs to `server_reproduction_runs/`.

## Environment Setup

Recommended baseline:

- Python 3.10
- CUDA-compatible GPU if running Agent A OCR or local VLM later
- PaddlePaddle GPU matching the server CUDA version
- PyTorch only if local VLM tests are planned

Install options:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Windows or CPU-only smoke checks:

```bash
pip install -r requirements_windows.txt
```

If the company server has a managed CUDA/Paddle environment, prefer the company-provided install command and then run:

```bash
python scripts/check_server_environment.py
```

## Required Data Layout

The project root on the server should contain:

```text
data/
├── dicts/
│   ├── Finance.txt
│   ├── Geology.txt
│   └── Medicine.txt
└── l2w1data/
    ├── train.jsonl
    ├── val.jsonl
    ├── test.jsonl
    └── images/
```

Expected split sizes:

| File | Expected lines |
|---|---:|
| `data/l2w1data/train.jsonl` | 8522 |
| `data/l2w1data/val.jsonl` | 3470 |
| `data/l2w1data/test.jsonl` | 3424 |

The server check script currently treats the `test.jsonl` count of 3424 as required because the frozen paper results use 3424 test samples.

## Required Model Layout

For integrity checks, the following must exist:

```text
models/agent_a_ppocr/PP-OCRv5_server_rec_infer/
ppocr/utils/ppocrv5_dict.txt
```

Level 0-3 checks do not run OCR and do not load the PP-OCR model weights. They only verify that the model directory and dictionary exist.

Level 4 and Level 5 may load PaddleOCR and use GPU/CPU inference.

## Required Frozen Results Layout

The paper1 frozen results should remain in:

```text
paper1_workspace/02_frozen_results/
```

Minimum required files:

```text
paper1_workspace/02_frozen_results/official_agent_a_cache/paper1_official_agent_a_cache.json
paper1_workspace/02_frozen_results/mainA_final/tab_mainA_results.csv
paper1_workspace/02_frozen_results/mainA_final/tab_mainA_budget_check.csv
paper1_workspace/02_frozen_results/mainA_final/tab_mainA_domain_results.csv
paper1_workspace/02_frozen_results/mainA_final/router_score_matrix.csv
paper1_workspace/02_frozen_results/mainA_final/shared_repmodel_full_call_cache.jsonl
paper1_workspace/02_frozen_results/mainC_final/tab_mainC_results.csv
paper1_workspace/02_frozen_results/mainC_final/tab_mainC_budget_check.csv
paper1_workspace/02_frozen_results/mainC_final/V1_full_call_cache.jsonl
paper1_workspace/02_frozen_results/mainC_final/V2_full_call_cache.jsonl
paper1_workspace/02_frozen_results/mainC_final/V3_full_call_cache.jsonl
paper1_workspace/02_frozen_results/mainC_final/V4_full_call_cache.jsonl
paper1_workspace/02_frozen_results/upper_lower_bounds_final/tab_upper_lower_bounds.csv
```

## Recommended Run Order

From the project root:

```bash
python scripts/check_server_environment.py
python scripts/verify_paper1_frozen_results.py
python scripts/reproduce_paper1_from_frozen.py
```

Outputs are written to:

```text
server_reproduction_runs/
├── frozen_verification_report.md
└── from_frozen/
    ├── copied_or_reproduced_tab_mainA_results.csv
    ├── copied_or_reproduced_tab_mainC_results.csv
    ├── copied_or_reproduced_budget_check_summary.csv
    └── reproduction_summary.md
```

## Steps That Do Not Call API

The following commands do not call API and are safe for server migration checks:

```bash
python scripts/check_server_environment.py
python scripts/verify_paper1_frozen_results.py
python scripts/reproduce_paper1_from_frozen.py
```

They do not read `key.txt`, `GPTkey.txt`, or `GPT.config`.

## Steps That May Call API

These scripts may call API or load VLM backends depending on arguments/config:

- `scripts/paper1_mainA_runner.py`
- `scripts/paper1_mainC_runner.py`
- `scripts/paper1_upper_lower_bounds.py`
- `scripts/run_efficiency_frontier.py`
- `scripts/run_full_budget_call.py`
- `scripts/run_gemini_ceiling.py`
- `scripts/test_provider_pools.py`
- `scripts/test_single_api.py`
- `scripts/_test_api.py`
- `scripts/_debug_agentb.py`

Do not run them during Level 0-3 checks. If a later stage requires them, first replace real key files with a company-approved secret manager or environment-based key loading flow, and explicitly document who approved API usage.

## API Calls Are Disabled by Default

This server reproduction stage is designed around frozen caches:

- Agent A OCR cache already exists.
- VLM full-call caches already exist.
- Main A and Main C replay can be checked without calling VLM APIs.

Any script added in this stage should fail closed: no API call should occur unless a future script has an explicit opt-in flag such as `--enable_api_calls`. No such opt-in is provided by the scripts in this stage.

## Common Errors and Troubleshooting

| Symptom | Likely cause | Check | Fix |
|---|---|---|---|
| `FAIL data/l2w1data/test.jsonl missing` | Data not copied or wrong working directory | `pwd` and `ls data/l2w1data` | Run from project root or copy data |
| `FAIL test.jsonl line count expected 3424` | Wrong split version | `wc -l data/l2w1data/test.jsonl` | Use the paper1 frozen dataset split |
| `FAIL official Agent A cache missing` | Frozen results not copied | `ls paper1_workspace/02_frozen_results/official_agent_a_cache` | Copy the full `paper1_workspace/02_frozen_results/` directory |
| `FAIL mainA full-call cache line count` | Cache truncated during transfer | `wc -l ...shared_repmodel_full_call_cache.jsonl` | Recopy with checksum verification |
| `WARN paddle import failed` | Paddle not installed or CUDA mismatch | `python -c "import paddle; print(paddle.__version__)"` | Install Paddle matching server CUDA |
| `WARN GPU not visible` | No GPU, container missing GPU, driver issue | `nvidia-smi` | Fix container runtime or driver; Level 0-3 can still pass without GPU |
| `cv2 import failed` | OpenCV missing | `python -c "import cv2"` | `pip install opencv-python` |
| `yaml import failed` | PyYAML missing | `pip install pyyaml` | Install dependency |
| `Levenshtein import failed` | Python Levenshtein missing | `pip install python-Levenshtein` | Install dependency |
| `pandas import failed` | pandas missing | `pip install pandas` | Install dependency |
| CSV row count mismatch | Wrong frozen result version | Compare against `paper1_workspace/01_registry/正式结果索引.md` | Copy the registered frozen results |
| Script accidentally asks for key file | Wrong script was run | Check command history | Stop, do not paste key content, run only Level 0-3 scripts |

## What To Send Back If Server Checks Fail

Send these files, not key files:

- console output from `python scripts/check_server_environment.py`
- `server_reproduction_runs/frozen_verification_report.md`
- `server_reproduction_runs/from_frozen/reproduction_summary.md`
- `python --version`
- `pip freeze` or the company environment module list
- `nvidia-smi` output if GPU-related

Never send `key.txt`, `GPTkey.txt`, `GPT.config`, or copied API responses containing secrets.
