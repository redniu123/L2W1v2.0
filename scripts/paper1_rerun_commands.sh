#!/usr/bin/env bash
set -euo pipefail

ROOT_DEFAULT="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${ROOT:-$ROOT_DEFAULT}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_run%H%M%S)}"
USE_GPU="${USE_GPU:-0}"
SKIP_BATCH2="${SKIP_BATCH2:-0}"

CACHE_DIR="${CACHE_DIR:-$ROOT/paper1_runs/shared_agent_a_cache/official_$RUN_TAG}"
CACHE_JSON="${CACHE_JSON:-$CACHE_DIR/paper1_official_agent_a_cache.json}"
MAINA_OUT="${MAINA_OUT:-$ROOT/paper1_runs/mainA_fixed}"
MAINC_OUT="${MAINC_OUT:-$ROOT/paper1_runs/mainC_fixed}"
PHASE1_OUT="${PHASE1_OUT:-$ROOT/paper1_runs/phase2_batch1_fixed}"
PHASE2_OUT="${PHASE2_OUT:-$ROOT/paper1_runs/phase2_batch2_fixed}"

mkdir -p "$CACHE_DIR" "$MAINA_OUT" "$MAINC_OUT" "$PHASE1_OUT" "$PHASE2_OUT"

cd "$ROOT"
echo "[INFO] ROOT=$ROOT"
echo "[INFO] RUN_TAG=$RUN_TAG"
echo "[INFO] CACHE_JSON=$CACHE_JSON"

echo "[STEP 0] Pull latest code"
git pull origin main

echo "[STEP 1] Generate official Agent A cache"
ROOT="$ROOT" CACHE_DIR="$CACHE_DIR" CACHE_JSON="$CACHE_JSON" USE_GPU="$USE_GPU" python - <<'PY'
import json, yaml, argparse, os
from pathlib import Path

ROOT = Path(os.environ['ROOT']).resolve()
cache_dir = Path(os.environ['CACHE_DIR']).resolve()
cache_json = Path(os.environ['CACHE_JSON']).resolve()
use_gpu = os.environ.get('USE_GPU', '0') == '1'

from scripts.run_efficiency_frontier import infer_all_samples, ensure_agent_a_result_schema
from modules.paddle_engine.predict_rec_modified import TextRecognizerWithLogits
from modules.router.domain_knowledge import DomainKnowledgeEngine

cfg = yaml.safe_load((ROOT / 'configs/router_config.yaml').read_text(encoding='utf-8'))
rec = argparse.Namespace(
    rec_model_dir='./models/agent_a_ppocr/PP-OCRv5_server_rec_infer',
    rec_char_dict_path='ppocr/utils/ppocrv5_dict.txt',
    rec_image_shape='3, 48, 320',
    rec_batch_num=6,
    rec_algorithm='SVTR_LCNet',
    use_space_char=True,
    use_gpu=use_gpu,
    use_xpu=False,
    use_npu=False,
    use_mlu=False,
    use_metax_gpu=False,
    use_gcu=False,
    ir_optim=True,
    use_tensorrt=False,
    min_subgraph_size=15,
    precision='fp32',
    gpu_mem=500,
    gpu_id=0,
    enable_mkldnn=None,
    cpu_threads=10,
    warmup=False,
    benchmark=False,
    save_log_path='./log_output/',
    show_log=False,
    use_onnx=False,
    max_batch_size=10,
    return_word_box=False,
    drop_score=0.5,
    max_text_length=25,
    rec_image_inverse=True,
    use_det=False,
    det_model_dir=''
)

samples = [json.loads(x) for x in (ROOT / 'data/l2w1data/test.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
recognizer = TextRecognizerWithLogits(rec)
domain_engine = DomainKnowledgeEngine({
    'geology': 'data/dicts/Geology.txt',
    'finance': 'data/dicts/Finance.txt',
    'medicine': 'data/dicts/Medicine.txt',
})
rows = ensure_agent_a_result_schema(
    infer_all_samples(samples, recognizer, domain_engine, None, str((ROOT / 'data/l2w1data/images').resolve()))
)
cache_dir.mkdir(parents=True, exist_ok=True)
cache_json.write_text(json.dumps(rows, ensure_ascii=False), encoding='utf-8')
(cache_dir / 'manifest.json').write_text(json.dumps({
    'type': 'paper1_official_agent_a_cache',
    'n_samples': len(rows),
    'cache_json': str(cache_json),
}, ensure_ascii=False, indent=2), encoding='utf-8')
(cache_dir / 'config_snapshot.yaml').write_text(
    yaml.safe_dump({'router_config': cfg}, allow_unicode=True, sort_keys=False),
    encoding='utf-8'
)
print(cache_json)
print(f'n_samples={len(rows)}')
PY

echo "[STEP 2] Run Main A on shared cache"
MAINA_ARGS=("scripts/paper1_mainA_runner.py" --output_dir "$MAINA_OUT" --shared_agent_a_cache "$CACHE_JSON" --use_cache)
if [ "$USE_GPU" = "1" ]; then
  MAINA_ARGS+=(--use_gpu)
fi
python "${MAINA_ARGS[@]}"

export MAINA_RUN
MAINA_RUN=$(MAINA_OUT="$MAINA_OUT" python - <<'PY'
import os
from pathlib import Path
p = Path(os.environ['MAINA_OUT'])
runs = sorted([x for x in p.iterdir() if x.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
print(runs[0])
PY
)
echo "[INFO] MAINA_RUN=$MAINA_RUN"

echo "[STEP 3] Run Main C on shared cache"
MAINC_ARGS=("scripts/paper1_mainC_runner.py" --output_dir "$MAINC_OUT" --shared_agent_a_cache "$CACHE_JSON" --use_cache --full_call_cache_dir "$MAINC_OUT")
if [ "$USE_GPU" = "1" ]; then
  MAINC_ARGS+=(--use_gpu)
fi
python "${MAINC_ARGS[@]}"

export MAINC_RUN
MAINC_RUN=$(MAINC_OUT="$MAINC_OUT" python - <<'PY'
import os
from pathlib import Path
p = Path(os.environ['MAINC_OUT'])
runs = sorted([x for x in p.iterdir() if x.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
print(runs[0])
PY
)
echo "[INFO] MAINC_RUN=$MAINC_RUN"

echo "[STEP 4] Run phase2_batch1 analysis"
python "scripts/paper1_batch1_analysis.py" --maina_run_dir "$MAINA_RUN" --mainc_run_dir "$MAINC_RUN" --output_root "$PHASE1_OUT"

export PHASE1_RUN
PHASE1_RUN=$(PHASE1_OUT="$PHASE1_OUT" python - <<'PY'
import os
from pathlib import Path
p = Path(os.environ['PHASE1_OUT'])
runs = sorted([x for x in p.iterdir() if x.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
print(runs[0])
PY
)
echo "[INFO] PHASE1_RUN=$PHASE1_RUN"

if [ "$SKIP_BATCH2" != "1" ]; then
  echo "[STEP 5] Run phase2_batch2 GPT/case analysis"
  python "scripts/paper1_gpt_case_analysis.py" --maina_run_dir "$MAINA_RUN" --mainc_run_dir "$MAINC_RUN" --output_root "$PHASE2_OUT"

  export PHASE2_RUN
  PHASE2_RUN=$(PHASE2_OUT="$PHASE2_OUT" python - <<'PY'
import os
from pathlib import Path
p = Path(os.environ['PHASE2_OUT'])
runs = sorted([x for x in p.iterdir() if x.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
print(runs[0])
PY
)
  echo "[INFO] PHASE2_RUN=$PHASE2_RUN"
fi

echo "[DONE] Rerun pipeline finished"
echo "[DONE] CACHE_JSON=$CACHE_JSON"
echo "[DONE] MAINA_RUN=${MAINA_RUN:-}"
echo "[DONE] MAINC_RUN=${MAINC_RUN:-}"
echo "[DONE] PHASE1_RUN=${PHASE1_RUN:-}"
echo "[DONE] PHASE2_RUN=${PHASE2_RUN:-}"
