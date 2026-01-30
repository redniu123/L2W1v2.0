#!/usr/bin/env bash
set -euo pipefail

# SH-DA++ v4.0 Stage 0/1 run script
# Usage: bash scripts/run_stage0_1.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG_PATH="${CONFIG_PATH:-configs/router_config.yaml}"
MODEL_DIR="${MODEL_DIR:-./models/agent_a_ppocr/PP-OCRv5_server_rec_infer/}"
DET_MODEL_DIR="${DET_MODEL_DIR:-}"
METADATA_TRAIN="${METADATA_TRAIN:-./data/raw/HWDB_Benchmark/train_metadata.jsonl}"
METADATA_TEST="${METADATA_TEST:-./data/raw/HWDB_Benchmark/test_metadata.jsonl}"
TARGET_B="${TARGET_B:-0.2}"
LIMIT="${LIMIT:-1000}"
OUTPUT_DIR="${OUTPUT_DIR:-./results}"

echo "=========================================="
echo "SH-DA++ v4.0 Stage 0/1 Execution Script"
echo "=========================================="
echo "CONFIG_PATH   = ${CONFIG_PATH}"
echo "MODEL_DIR     = ${MODEL_DIR}"
echo "DET_MODEL_DIR = ${DET_MODEL_DIR:-<empty>}"
echo "METADATA_TRAIN= ${METADATA_TRAIN}"
echo "METADATA_TEST = ${METADATA_TEST}"
echo "TARGET_B      = ${TARGET_B}"
echo "LIMIT         = ${LIMIT}"
echo "OUTPUT_DIR    = ${OUTPUT_DIR}"
echo "=========================================="

# Enforce DET enabled in config (guarantee)
python - <<'PY'
import os
import sys
from datetime import datetime
import yaml

config_path = os.environ.get("CONFIG_PATH", "configs/router_config.yaml")
det_model_dir = os.environ.get("DET_MODEL_DIR", "").strip()

if not os.path.exists(config_path):
    print(f"[FATAL] Config not found: {config_path}")
    sys.exit(1)

with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}

stage0 = config.get("stage0") or {}
changed = False

if stage0.get("use_det") is not True:
    stage0["use_det"] = True
    changed = True

if det_model_dir:
    if stage0.get("det_model_dir") != det_model_dir:
        stage0["det_model_dir"] = det_model_dir
        changed = True

config["stage0"] = stage0

if changed:
    backup_path = f"{config_path}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with open(backup_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=False, sort_keys=False)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=False, sort_keys=False)
    print(f"[Info] DET enforced. Updated config and backed up to: {backup_path}")
else:
    print("[Info] DET already enabled in config.")

if not stage0.get("det_model_dir"):
    print("[Warning] stage0.det_model_dir is empty. If DET init fails, set DET_MODEL_DIR.")
PY

# Step 1: Clean results
echo "[1/4] Cleaning results..."
rm -rf "${OUTPUT_DIR:?}/"*
mkdir -p "${OUTPUT_DIR}"

# Step 2: Calibrate router
echo "[2/4] Calibrating router..."
python scripts/calibrate_router.py \
  --config "${CONFIG_PATH}" \
  --metadata "${METADATA_TRAIN}" \
  --target_b "${TARGET_B}" \
  --limit "${LIMIT}" \
  --model_dir "${MODEL_DIR}"

# Step 3: Stage 1 collection
echo "[3/4] Stage 1 collection..."
python scripts/run_stage1_collection.py \
  --config "${CONFIG_PATH}" \
  --metadata "${METADATA_TEST}" \
  --model_dir "${MODEL_DIR}" \
  --output_dir "${OUTPUT_DIR}"

# Step 4: Budget stability test
echo "[4/4] Budget stability test..."
python scripts/test_budget_stability.py \
  --config "${CONFIG_PATH}" \
  --target_b "${TARGET_B}" \
  --model_dir "${MODEL_DIR}"

# Final audit
echo "[Final] Evaluation..."
python scripts/evaluate.py \
  --predictions "${OUTPUT_DIR}/router_features.jsonl"

echo "[Done] Stage 0/1 pipeline completed."
