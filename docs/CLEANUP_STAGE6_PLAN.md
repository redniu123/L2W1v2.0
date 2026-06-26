# Stage 6 计划 — 路由模块 `src/l2w1/routing/`

**日期：** 2026-06-26
**负责：** Claude（接口规格 + review）/ Codex（实现）
**目标：** 把 `modules/router/` 中**纯逻辑、无模型依赖**的路由组件，干净抽取到 `src/l2w1/routing/`，并用 **parity tests**（新旧实现逐值对齐）保证不改变 Paper1 数值行为。

---

## 一、范围裁定（重要）

设计文档列了 5 个子模块（scores / ranking / budget / backfill / circuit）。实际审计 `modules/router/` 后：

| 子模块 | 抽取源 | 状态 |
|--------|--------|------|
| `circuit.py` | `circuit_breaker.CircuitBreaker` + `CircuitBreakerConfig` | ✅ 本阶段做（纯逻辑，零依赖） |
| `budget.py` | `uncertainty_router.OnlineBudgetController` + `BudgetControllerConfig` | ✅ 本阶段做（纯 PI 控制） |
| `scores.py` | `calibrated_scorer.CalibratedScorer` + `CalibratedScorerConfig` + `RuleOnlyScorer` | ✅ 本阶段做（纯 numpy） |
| `backfill.py` | `backfill.StrictBackfillController` + 配套 dataclass/enum + `apply_strict_backfill` | ✅ 本阶段做（依赖 `Levenshtein`，已装 0.27.3） |
| `ranking.py` | （无干净来源） | ⏸️ **暂缓**：`modules/router/` 中不存在独立的样本排序/分档逻辑（grep 无 percentile/argsort/rank/bin）。硬造逻辑违反"最小改动 / 不为重构而重构"。Stage 9 若发现脚本里有真实排序逻辑再回头补。 |

**不抽取**：`uncertainty_router.py` 里的 `CTCAligner` / `VisualEntropyCalculator` / `SemanticPPLCalculator` / `UncertaintyRouter`（涉及 logits / transformer，属于 Stage 8 VLM/OCR 抽象范畴），以及 `sh_da_router.SHDARouter`（编排层，依赖未抽取部分，留待后续）。

## 二、交付物

```
src/l2w1/routing/
├── __init__.py          # 导出四个模块的公开符号
├── circuit.py
├── budget.py
├── scores.py
└── backfill.py
tests/
├── conftest.py                      # 确保仓库根在 sys.path（供 parity 测试 import modules.router.*）
├── test_routing_circuit_parity.py
├── test_routing_budget_parity.py
├── test_routing_scores_parity.py
└── test_routing_backfill_parity.py
docs/CLEANUP_STAGE6_REPORT.md
```

## 三、抽取规范

1. **近似逐字复制 + 现代化清理**：逻辑/数值/默认值/字段名**完全不变**；仅做：
   - 类型注解现代化：`Dict`→`dict`、`Optional[X]`→`X | None`、`List`→`list`、`Tuple`→`tuple`。
   - 移除未使用 import；补 `from __future__ import annotations`。
   - 注释保持英文；docstring 可保留中文（与现有 src 模块一致即可，优先英文）。
   - 强制关键字参数：把原本的可选参数按需用 `*` 分隔（**但不得改变调用语义**——若旧代码按位置调用，保持兼容）。
2. **dataclass 默认值修复**：原代码有 `weights: Dict[str, float] = None` 这类「可变/None 默认 + `__post_init__`」写法。保持**行为不变**，但类型注解写成 `dict[str, float] | None = None`，让 mypy strict 通过。
3. **数值绝对不变**：sigmoid 的 clip 范围、`b_edge_L = 0.6*mean+0.4*peak`、ED 阈值=3、长度变化阈值=0.2、熔断 `min_samples`/`rejection_rate_threshold`/`cooldown_steps`、预算控制器的 λ 更新公式等，一律照搬。

## 四、parity 测试策略（核心质量保障）

每个 parity 测试同时 import **旧实现**（`modules.router.*`）与**新实现**（`l2w1.routing.*`），在一组**合成输入**上断言输出逐值相等：

- **circuit**：同一串 `observe(rejected=...)` 序列喂给新旧两个实例，断言每步 `get_stats()` 与 `triggered` 完全一致，含冷却倒计时与 `is_open`。
- **budget**：同一串 `q` 值序列 + 相同 config 喂给新旧 `OnlineBudgetController`，断言每步 `step(q)` 的 `(upgrade, details)`、`current_lambda`、`get_stats()` 一致。覆盖 warmup 与稳态。
- **scores**：随机/边界特征向量（mean_conf/min_conf/b_edge/drop/r_d）喂给新旧 `CalibratedScorer.compute_score_v51` 与 `compute_score`，断言 `s_b`/`logit` 相等（浮点用 `pytest.approx` 或 `==`，因同为 float32 路径应严格相等）；`RuleOnlyScorer` 同理。
- **backfill**：构造若干 `(T_A, T_cand)` 对（含 ED 超限、长度超限、纯格式改写如 `（）`→`()`、正常接受、空串等），断言新旧 `apply_strict_backfill` 的 `BackfillResult` 全字段一致（`T_final`/`is_rejected`/`rejection_reason`/`edit_distance`/`length_change_ratio`）。

`tests/conftest.py` 必须把仓库根目录插入 `sys.path`（幂等），否则 `import modules.router.*` 在 src-layout 下找不到。

## 五、铁律

- ❌ 不改 `modules/router/`（只读抽取，旧代码原地保留）。
- ❌ 不触碰任何受保护路径；不运行任何加载模型 / 调 API 的代码。
- ❌ parity 测试只用合成输入，不读真实数据/缓存。
- ✅ 通过 `pytest`（新增 + 现有全绿）、`ruff check src/ tests/`、`mypy src/l2w1`。

---

## 六、验收命令（Claude 独立执行）

```bash
python -m pytest tests/ -q
python -m ruff check src/ tests/
python -m mypy src/l2w1
```
全绿后分模块提交。
