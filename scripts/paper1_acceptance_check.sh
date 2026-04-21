#!/usr/bin/env bash
set -euo pipefail

ROOT_DEFAULT="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="${ROOT:-$ROOT_DEFAULT}"
CACHE_JSON="${CACHE_JSON:-}"
MAINA_RUN="${MAINA_RUN:-}"
MAINC_RUN="${MAINC_RUN:-}"
PHASE1_RUN="${PHASE1_RUN:-}"
PHASE2_RUN="${PHASE2_RUN:-}"

cd "$ROOT"

if [ -z "$CACHE_JSON" ]; then
  CACHE_JSON=$(python - <<'PY'
from pathlib import Path
p = Path('paper1_runs/shared_agent_a_cache')
runs = sorted([x for x in p.iterdir() if x.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
print((runs[0] / 'paper1_official_agent_a_cache.json').resolve())
PY
)
fi
if [ -z "$MAINA_RUN" ]; then
  MAINA_RUN=$(python - <<'PY'
from pathlib import Path
p = Path('paper1_runs/mainA_fixed')
runs = sorted([x for x in p.iterdir() if x.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
print(runs[0].resolve())
PY
)
fi
if [ -z "$MAINC_RUN" ]; then
  MAINC_RUN=$(python - <<'PY'
from pathlib import Path
p = Path('paper1_runs/mainC_fixed')
runs = sorted([x for x in p.iterdir() if x.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
print(runs[0].resolve())
PY
)
fi
if [ -z "$PHASE1_RUN" ]; then
  PHASE1_RUN=$(python - <<'PY'
from pathlib import Path
p = Path('paper1_runs/phase2_batch1_fixed')
runs = sorted([x for x in p.iterdir() if x.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
print(runs[0].resolve())
PY
)
fi
if [ -z "$PHASE2_RUN" ]; then
  if [ -d "$ROOT/paper1_runs/phase2_batch2_fixed" ]; then
    PHASE2_RUN=$(python - <<'PY'
from pathlib import Path
p = Path('paper1_runs/phase2_batch2_fixed')
runs = sorted([x for x in p.iterdir() if x.is_dir()], key=lambda x: x.stat().st_mtime, reverse=True)
print(runs[0].resolve() if runs else '')
PY
)
  fi
fi

echo "[CHECK] ROOT=$ROOT"
echo "[CHECK] CACHE_JSON=$CACHE_JSON"
echo "[CHECK] MAINA_RUN=$MAINA_RUN"
echo "[CHECK] MAINC_RUN=$MAINC_RUN"
echo "[CHECK] PHASE1_RUN=$PHASE1_RUN"
echo "[CHECK] PHASE2_RUN=${PHASE2_RUN:-}"

echo "\n[CHECK 1] Agent A cache integrity"
CACHE_JSON="$CACHE_JSON" python - <<'PY'
import json, os, sys
from pathlib import Path
cache_json = Path(os.environ['CACHE_JSON'])
if not cache_json.exists():
    print(f'[FAIL] cache not found: {cache_json}')
    sys.exit(2)
rows = json.loads(cache_json.read_text(encoding='utf-8'))
n = len(rows)
eq_all = 0
eq_mean_min = 0
non_null_idx = 0
bad_range = 0
for r in rows:
    conf = float(r.get('conf', 0))
    mean_conf = float(r.get('mean_conf', 0))
    min_conf = float(r.get('min_conf', 0))
    idx = r.get('min_conf_idx')
    if abs(mean_conf - min_conf) < 1e-12 and abs(mean_conf - conf) < 1e-12:
        eq_all += 1
    if abs(mean_conf - min_conf) < 1e-12:
        eq_mean_min += 1
    if idx is not None:
        non_null_idx += 1
    if not (0.0 <= min_conf <= 1.0 and 0.0 <= mean_conf <= 1.0 and 0.0 <= conf <= 1.0):
        bad_range += 1
print({'n_samples': n, 'mean=min=conf': eq_all, 'mean=min': eq_mean_min, 'non_null_min_conf_idx': non_null_idx, 'bad_range': bad_range})
if eq_all == n:
    print('[FAIL] mean_conf/min_conf/conf still fully collapsed')
    sys.exit(3)
if non_null_idx == 0:
    print('[FAIL] min_conf_idx is empty for all samples')
    sys.exit(4)
if bad_range != 0:
    print('[FAIL] confidence out-of-range values found')
    sys.exit(5)
print('[PASS] Agent A cache integrity passed')
PY

echo "\n[CHECK 2] Main A core outputs"
MAINA_RUN="$MAINA_RUN" python - <<'PY'
import csv, os, sys
from pathlib import Path
run_dir = Path(os.environ['MAINA_RUN'])
need = ['tab_mainA_results.csv', 'tab_mainA_budget_check.csv', 'router_score_matrix.csv']
for name in need:
    p = run_dir / name
    print(name, p.exists(), p.stat().st_size if p.exists() else -1)
    if not p.exists() or p.stat().st_size == 0:
        print(f'[FAIL] missing/empty {name}')
        sys.exit(10)
with (run_dir / 'tab_mainA_budget_check.csv').open('r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
print('mainA_budget_rows=', len(rows))
for r in rows:
    print(r.get('router_name'), r.get('budget'), r.get('CER'), r.get('actual_call_rate'), r.get('call_rate_valid'))
if any(str(r.get('call_rate_valid')).lower() not in ('true', '1') for r in rows):
    print('[FAIL] Main A has invalid call_rate_valid rows')
    sys.exit(11)
print('[PASS] Main A core outputs passed')
PY

echo "\n[CHECK 3] Main C core outputs"
MAINC_RUN="$MAINC_RUN" python - <<'PY'
import csv, os, sys
from pathlib import Path
run_dir = Path(os.environ['MAINC_RUN'])
need = ['tab_mainC_results.csv', 'tab_mainC_budget_check.csv', 'V1_full_call_cache.jsonl', 'V2_full_call_cache.jsonl', 'V3_full_call_cache.jsonl', 'V4_full_call_cache.jsonl']
for name in need:
    p = run_dir / name
    print(name, p.exists(), p.stat().st_size if p.exists() else -1)
    if not p.exists() or p.stat().st_size == 0:
        print(f'[FAIL] missing/empty {name}')
        sys.exit(20)
with (run_dir / 'tab_mainC_budget_check.csv').open('r', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))
print('mainC_budget_rows=', len(rows))
for r in rows:
    print(r.get('model_name'), r.get('budget'), r.get('CER'), r.get('actual_call_rate'), r.get('call_rate_valid'))
if any(str(r.get('call_rate_valid')).lower() not in ('true', '1') for r in rows):
    print('[FAIL] Main C has invalid call_rate_valid rows')
    sys.exit(21)
print('[PASS] Main C core outputs passed')
PY

echo "\n[CHECK 4] phase2_batch1 outputs"
PHASE1_RUN="$PHASE1_RUN" MAINA_RUN="$MAINA_RUN" python - <<'PY'
import csv, os, sys
from pathlib import Path
run_dir = Path(os.environ['PHASE1_RUN'])
maina_run = Path(os.environ['MAINA_RUN'])
need = [
    'tab_random_baseline_results.csv',
    'tab_minconf_baseline_results.csv',
    'tab_domain_budget_curve.csv',
    'tab_domain_router_ranking.csv',
    'tab_domain_model_comparison.csv',
    'tab_cost_effectiveness.csv',
]
for name in need:
    p = run_dir / name
    print(name, p.exists(), p.stat().st_size if p.exists() else -1)
    if not p.exists() or p.stat().st_size == 0:
        print(f'[FAIL] missing/empty {name}')
        sys.exit(30)
with (maina_run / 'tab_mainA_results.csv').open('r', encoding='utf-8') as f:
    maina = [r for r in csv.DictReader(f) if r['router_name'] == 'GCR']
with (run_dir / 'tab_minconf_baseline_results.csv').open('r', encoding='utf-8') as f:
    minconf = list(csv.DictReader(f))
maina_map = {round(float(r['budget']), 2): float(r['CER']) for r in maina}
min_map = {round(float(r['budget']), 2): float(r['CER']) for r in minconf}
all_same = True
for b in sorted(min_map):
    g = maina_map.get(b)
    m = min_map[b]
    delta = None if g is None else round(m - g, 6)
    print({'budget': b, 'GCR': g, 'MinConf': m, 'delta': delta})
    if delta is not None and delta != 0.0:
        all_same = False
if all_same:
    print('[FAIL] MinConf is still identical to GCR at all budgets')
    sys.exit(31)
print('[PASS] phase2_batch1 outputs passed')
PY

if [ -n "${PHASE2_RUN:-}" ] && [ -d "$PHASE2_RUN" ]; then
  echo "\n[CHECK 5] phase2_batch2 outputs"
  PHASE2_RUN="$PHASE2_RUN" python - <<'PY'
import os, sys
from pathlib import Path
run_dir = Path(os.environ['PHASE2_RUN'])
need = ['gpt_error_bucket_stats.csv', 'model_error_style_comparison.csv', 'main_casebook.csv', 'figure_case_samples.md']
for name in need:
    p = run_dir / name
    print(name, p.exists(), p.stat().st_size if p.exists() else -1)
    if not p.exists() or p.stat().st_size == 0:
        print(f'[FAIL] missing/empty {name}')
        sys.exit(40)
print('[PASS] phase2_batch2 outputs passed')
PY
else
  echo "\n[CHECK 5] phase2_batch2 outputs skipped (directory not found)"
fi

echo "\n[ALL PASS] Acceptance checks finished successfully"
