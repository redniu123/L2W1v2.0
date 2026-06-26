# Stage 7 计划 — 回放模块 `src/l2w1/replay/`

**日期：** 2026-06-26
**负责：** Claude（接口规格 + review）/ Codex（实现）
**目标：** 把 Paper1「离线回放」「在线预算回放」的**纯计算逻辑**从 `scripts/paper1_mainA_runner.py` 与 `scripts/paper1_online_budget_check.py` 抽取到 `src/l2w1/replay/`。**只读缓存，绝不调 API / 加载模型 / 写实验结果。**

---

## 一、抽取源与边界

| 源脚本 | 抽取的纯逻辑 | 不抽取（留在脚本） |
|--------|--------------|---------------------|
| `paper1_mainA_runner.py` | `score()`（策略打分）、`norm()`（格式归一）、`replay()`（按分排序取 top-budget + per-sample + CER/AER）、`domain_rows()`、`case_pool()` | `build_full()`（调 Agent B）、`main()`（加载 PaddleOCR/argparse/写文件） |
| `paper1_online_budget_check.py` | `build_router_score()`、`build_cached_result_lookup()`、`load_jsonl()`、`run_online_routeronly()`（用 `OnlineBudgetController` 在线选择 + per-sample + CER/AER） | `main()`（加载模型/写文件） |

**关键约束：** `score()` 与 `build_router_score()` 数值完全一致（GCR=1-conf；DGCR=(1-conf)+r_d；WUR=0.5·(1-mean)+0.3·(1-min)+0.2·drop，min<0.35 +0.10，drop>0.20 +0.10；DWUR=WUR+eta·r_d）→ 合并为**一个** `router_score()`。

## 二、交付物

```
src/l2w1/replay/
├── __init__.py
├── paper1_cache.py     # 缓存读取 + 按 sample_id 建索引（复用 l2w1.io.jsonl）
├── scoring.py          # router_score(strategy, row, *, eta) + norm 复用
├── offline.py          # select_offline_upgrades() + replay_offline()
└── online_budget.py    # replay_online()
tests/
├── test_replay_offline_parity.py
├── test_replay_online_parity.py
└── test_replay_cache.py
docs/CLEANUP_STAGE7_REPORT.md
```

## 三、API 契约

```python
# paper1_cache.py
def load_cache_rows(path: str | Path) -> list[dict]:        # 复用 l2w1.io.jsonl.read_jsonl
def build_cached_result_lookup(rows: Iterable[dict]) -> dict[str, dict]:  # 按非空 sample_id 索引（后者覆盖前者，与旧实现一致）

# scoring.py
def router_score(strategy: str, row: Mapping[str, Any], *, eta: float = 0.5) -> float:
    # 与 paper1_mainA_runner.score / online build_router_score 数值逐位一致；非法 strategy 抛 ValueError（与 online 版一致）
def normalize_format(text: str) -> str:   # 与 mainA.norm() 同一张全/半角标点映射表（与 backfill._FORMAT_EQUIVALENCE 同表）

# offline.py
def select_offline_upgrades(scores: Sequence[float], budget: float) -> tuple[set[int], dict[int, int]]:
    # ranked = 按分降序的索引；n = round(len*budget)；upgrade = 前 n；rank_map = {idx: 1-based 排名}
def replay_offline(
    strategy: str, budget: float, full_rows: Sequence[dict], score_map: Sequence[float], *,
    prompt_version: str, run_id: str = '',
    extended_metrics_fn: Callable[[list[dict]], dict] | None = None,
    usage_metrics_fn: Callable[[list[dict]], dict] | None = None,
) -> dict:
    # 返回 {'summary': {...}, 'per_sample': [...]}，逐字段复刻 mainA.replay() 的输出
    # 若注入 extended_metrics_fn / usage_metrics_fn，则 summary 中对应字段用其结果；否则置 None（不耦合 scripts/）

# online_budget.py
def replay_online(
    strategy: str, target_budget: float, all_results: Sequence[dict], cached_lookup: Mapping[str, dict],
    budget_cfg: BudgetControllerConfig, *,  # 来自 l2w1.routing.budget
    run_id: str = '', prompt_version: str = 'prompt_v1.1',
    agent_b_label: str = 'Gemini 3 Flash Preview', eta: float = 0.5,
    extended_metrics_fn=None, usage_metrics_fn=None,
) -> dict:
    # 逐字段复刻 online_budget_check.run_online_routeronly() 的输出；使用 l2w1.routing.budget.OnlineBudgetController
```

**依赖方向：** `src/l2w1/replay` 只可依赖 `l2w1.io`、`l2w1.routing.budget`、`Levenshtein`、标准库；**不得 import `scripts.*` 或 `modules.*`**，也不得 import torch/transformers/paddle。summarizer 通过参数注入。

## 四、parity 测试策略（核心）

构造**合成缓存 fixtures**（在内存里造若干 dict，字段齐全：`sample_id, ocr_text/T_A, gt/T_GT, final_text_if_upgraded, vlm_raw_output, latency_ms, token_usage, error_type, mean_conf, min_conf, drop, conf, r_d, domain, split, image_path, has_professional_terms, professional_terms` 等）。

- **offline parity**：对每个 strategy×budget，分别跑
  - 旧：`scripts.paper1_mainA_runner.replay(strategy, budget, full, score_map, pv, run_id)`
  - 新：`l2w1.replay.offline.replay_offline(..., extended_metrics_fn=run_efficiency_frontier.summarize_extended_metrics, usage_metrics_fn=run_efficiency_frontier.summarize_latency_and_token_usage)`
  - 断言 `summary` 与 `per_sample` **逐字段完全相等**（score_map 用新 `router_score` 生成，并先断言它与旧 `score()` 对每行结果一致）。
- **online parity**：合成 `all_results` + `cached_lookup` + 同一 `BudgetControllerConfig`，比较
  - 旧：`scripts.paper1_online_budget_check.run_online_routeronly(...)`
  - 新：`l2w1.replay.online_budget.replay_online(...)`（注入同样的 summarizer）
  - 断言 `summary`/`per_sample`/`validation_logs` 逐字段相等。
- **cache 测试**：`load_cache_rows` 读 tmp jsonl 往返；`build_cached_result_lookup` 索引与覆盖语义（空 sample_id 跳过）。

> 注：parity 测试**可以** import `scripts.run_efficiency_frontier`（已验证 import 安全、纯函数）与两个回放脚本（模型导入都在 `main()` 内，模块级 import 安全）。这是**测试**对旧实现的引用，不违反"src 不依赖 scripts"——被测的新模块本身不依赖 scripts。

## 五、铁律

- ❌ 不改任何 `scripts/`、`modules/`；不触碰受保护路径；不调模型/API；不写实验结果。
- ❌ parity 测试只用合成内存数据，不读 `paper1_runs/` / `data/` 真实缓存。
- ✅ 通过 `pytest`（全绿不退化）、`ruff check src/ tests/`、`mypy src/l2w1`。

## 六、验收命令（Claude 独立执行）

```bash
python -m pytest tests/ -q
python -m ruff check src/ tests/
python -m mypy src/l2w1
```
全绿后分模块提交。
