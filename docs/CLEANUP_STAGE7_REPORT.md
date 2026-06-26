# Stage 7 报告 — 回放模块 `src/l2w1/replay/`

**日期：** 2026-06-26  
**范围：** Paper1 离线回放与在线预算回放的纯计算逻辑抽取

## 一、已完成内容

新增 `src/l2w1/replay/`：

- `paper1_cache.py`
  - `load_cache_rows()` 复用 `l2w1.io.jsonl.read_jsonl`
  - `build_cached_result_lookup()` 按非空 `sample_id` 建索引，后出现的同 ID 行覆盖前行，保持旧脚本行为
- `scoring.py`
  - `router_score()` 合并 `paper1_mainA_runner.score()` 与 `paper1_online_budget_check.build_router_score()` 的 GCR/WUR/DGCR/DWUR 逻辑
  - `normalize_format()` 使用与旧 `mainA.norm()`、`routing.backfill._FORMAT_EQUIVALENCE` 相同的全/半角标点映射表
- `offline.py`
  - `select_offline_upgrades()` 实现离线分数降序排序、预算截断与 1-based rank map
  - `replay_offline()` 复刻 `paper1_mainA_runner.replay()` 的 summary/per_sample 字段与数值逻辑
- `online_budget.py`
  - `replay_online()` 复刻 `paper1_online_budget_check.run_online_routeronly()` 的 summary/per_sample/validation_logs 字段与数值逻辑
  - 使用 `l2w1.routing.budget.OnlineBudgetController` 与 `BudgetControllerConfig`
- `__init__.py`
  - 导出 Stage 7 公共 API

新增测试：

- `tests/test_replay_cache.py`
- `tests/test_replay_offline_parity.py`
- `tests/test_replay_online_parity.py`

## 二、依赖注入与边界

`src/l2w1/replay` 不导入 `scripts.*`、`modules.*`，也不导入模型/API 相关依赖。当前依赖仅限：

- 标准库
- `Levenshtein`
- `l2w1.io.jsonl`
- `l2w1.routing.budget`
- 包内相对导入

扩展指标与用量指标通过 `extended_metrics_fn` / `usage_metrics_fn` 注入。未注入时，summary 中的 `BoundaryDeletionRecallAtB`、`SubstitutionCER`、`p95_latency_ms`、`avg_token_usage` 均为 `None`，避免新模块耦合旧脚本。

测试中允许导入旧脚本作 parity 对照；这是测试依赖，不进入 `src/l2w1/replay`。

## 三、行为一致性覆盖

- 离线 replay 与 `scripts.paper1_mainA_runner.replay()` 对多个 strategy/budget 组合逐字段比较 summary 与 per_sample。
- 在线 replay 与 `scripts.paper1_online_budget_check.run_online_routeronly()` 逐字段比较 summary、per_sample、validation_logs。
- `router_score()` 对 GCR/WUR/DGCR/DWUR 与旧实现逐值比较，并覆盖非法 strategy 抛 `ValueError`。
- 离线 CER/edit distance 保持旧逻辑：先做格式归一化。
- 在线 CER/edit distance 保持旧逻辑：使用原始字符串，不做格式归一化。
- cache 测试覆盖 jsonl 读取、空 `sample_id` 跳过、重复 ID 后者覆盖前者。

## 四、最终验证输出

```bash
$ source /home/coder/anaconda3/etc/profile.d/conda.sh
$ conda activate l2w1
$ python -m pytest tests/ -q
........................................................................ [ 59%]
..................................................                       [100%]
122 passed in 0.66s
```

```bash
$ python -m ruff check src/ tests/
All checks passed!
```

```bash
$ python -m mypy src/l2w1
Success: no issues found in 24 source files
```

额外边界检查：

```bash
$ rg -n "scripts\.|modules\.|import scripts|import modules|torch|transformers|paddle|tqdm|yaml|numpy" src/l2w1/replay || true
# no output
```

## 五、受保护路径

本阶段未修改 `scripts/`、`modules/`，未读取 `key.txt` 内容，未运行模型/API 代码，未写入实验结果目录。
