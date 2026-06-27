# L2W1 — Hierarchical Multi-Agent OCR Correction

> A research system (Paper 1) that pairs a fast OCR pass with a cost-aware router and a
> selective VLM "expert" to correct hard cases — and a clean, tested Python library
> (`l2w1`) extracted from that research code.

[![CI](https://github.com/redniu123/L2W1v2.0/actions/workflows/ci.yml/badge.svg)](https://github.com/redniu123/L2W1v2.0/actions)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![checks](https://img.shields.io/badge/ruff%20%7C%20mypy--strict-passing-success)
![tests](https://img.shields.io/badge/tests-134%20passing-success)

---

## What it does

```
Input image
  └─→ Agent A  (PaddleOCR-VL)          fast first-pass recognition
        └─→ Router (SH-DA)             decides per-sample using visual entropy + semantic PPL
              ├─→ easy sample  → return Agent A output directly
              └─→ hard sample  → Agent B (Qwen2.5-VL / Gemini) → fine-grained correction
```

The router spends a limited "upgrade budget" only where it pays off. Key metrics:
**CER** (character error rate), **OCR-R** (over-correction rate), **CR** (correction rate),
and **CVR / AER** (constraint-violation / accepted-edit rate).

---

## The `l2w1` library

The heart of this repo is a from-scratch, fully-tested core library under `src/l2w1/`,
progressively extracted from messy research scripts into clean, importable modules.
Every extracted module is backed by **parity tests** that assert the new implementation
produces *value-identical* output to the original research code — so the refactor provably
does not change any Paper 1 numbers.

| Module | Responsibility |
|--------|----------------|
| `l2w1.config`  | Unified settings loader (`env > .env > yaml > default`). Never reads secret files — only passes their paths. |
| `l2w1.io`      | JSONL / CSV / cache IO helpers. |
| `l2w1.metrics` | CER, edit distance, reliability (OCR-R, CR, CVR/AER), summaries. |
| `l2w1.routing` | Pure routing logic: calibrated scorer, online budget controller, circuit breaker, strict backfill. |
| `l2w1.replay`  | Read-only offline / online-budget replay over cached results (no API, no model). |
| `l2w1.vlm`     | VLM (Agent B) interface abstraction + mock + cache-only implementations. |
| `l2w1.ocr`     | OCR (Agent A) interface abstraction + mock + cache-only implementations. |

Design notes and per-stage reports live in [`docs/`](docs/) (`CLEANUP_STAGE5..9_*.md`).

---

## Quickstart

```bash
# Python 3.10+. The l2w1 core library is pure-stdlib; install it editable:
pip install -e ".[dev]"

# run the test suite (synthetic fixtures only — no data/models/network needed)
pytest -q                 # 134 passed

# lint + type-check
ruff check src/ tests/
mypy src/l2w1             # strict mode, clean
```

> Dependencies are authoritative in `pyproject.toml`. `requirements.txt` pins the full
> GPU research environment (PaddleOCR + Torch + VLM stack) for reproducing Paper 1 runs;
> it is **not** needed to use or test the `l2w1` library.

### Thin-entrypoint example

A self-contained demo wires the cache-only OCR/VLM adapters into the offline replay,
entirely on synthetic in-memory data — no models, no API calls:

```bash
python scripts/examples/replay_offline_demo.py
```

```python
from l2w1.replay.offline import replay_offline
from l2w1.replay.scoring import router_score

scores = [router_score("WUR", row) for row in rows]
result = replay_offline("WUR", budget=0.5, full_rows=rows, score_map=scores,
                        prompt_version="demo")
print(result["summary"]["CER"])
```

---

## Repository layout

```
src/l2w1/        clean, tested core library (the part to reuse)
tests/           pytest suite incl. parity tests vs. original implementations
scripts/         CLI entry points; scripts/{dev,legacy,examples}/ separated
  examples/      thin-entrypoint demos built only on l2w1.*
modules/         transitional research modules (router/vlm/ocr) — kept for parity & repro
configs/         routing config (YAML, no secrets)
docs/            design docs + per-stage cleanup reports
ppocr/, tools/   vendored PaddleOCR code (do not modify)
```

> **Not included in this repository:** Paper 1 experiment data, model weights, and result
> bundles (`data/`, `models/`, `paper1_runs/`, `results/`, `cloud_result_sync/`). These are
> intentionally excluded — the code is open, the unpublished research artifacts are not.

---

## Engineering

- **Packaging:** installable via `pip install -e .` (`pyproject.toml`, hatchling).
- **Quality gate:** `ruff` (lint+format), `mypy --strict`, `pytest` — enforced in CI and pre-commit.
- **Testing philosophy:** synthetic fixtures only; no test depends on real data, models, or network.
- **Secrets:** never committed. API providers read credentials from a local key file path
  resolved through `l2w1.config`; that file is git-ignored and absent from history.

---

## Status

Refactor stages 0–9 complete. The library is stable and tested; remaining work (wiring the
config layer into legacy scripts, thinning the frozen Paper 1 runners) is tracked in the
`docs/CLEANUP_STAGE*` reports.

## License

Research code released for transparency and reuse of the `l2w1` library. See repository for terms.
