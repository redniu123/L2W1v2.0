#!/usr/bin/env bash
set -euo pipefail

# SH-DA++ v4.0 Stage 0/1 run script
# Usage: bash scripts/run_stage0_1.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

CONFIG_PATH="${CONFIG_PATH:-configs/router_config.yaml}"
MODEL_DIR="${MODEL_DIR:-./models/agent_a_ppocr/PP-OCRv5_server_rec_infer/}"
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
echo "METADATA_TRAIN= ${METADATA_TRAIN}"
echo "METADATA_TEST = ${METADATA_TEST}"
echo "TARGET_B      = ${TARGET_B}"
echo "LIMIT         = ${LIMIT}"
echo "OUTPUT_DIR    = ${OUTPUT_DIR}"
echo "=========================================="

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
