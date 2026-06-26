# Stage 6 报告 — 路由纯逻辑模块抽取

**日期：** 2026-06-26  
**范围：** `modules/router/` 中无模型/API 依赖的路由纯逻辑组件

## 一、已完成内容

新增 `src/l2w1/routing/` 包，按 Stage 6 计划抽取并清理以下模块：

| 新模块 | 旧来源 | 内容 |
|--------|--------|------|
| `src/l2w1/routing/circuit.py` | `modules/router/circuit_breaker.py` | `CircuitBreakerConfig`, `CircuitBreaker` |
| `src/l2w1/routing/budget.py` | `modules/router/uncertainty_router.py` | `BudgetControllerConfig`, `OnlineBudgetController` |
| `src/l2w1/routing/scores.py` | `modules/router/calibrated_scorer.py` | `CalibratedScorerConfig`, `CalibratedScorer`, `RuleOnlyScorer`, `FEATURE_NAMES_V51`, `FEATURE_NAMES_V40` |
| `src/l2w1/routing/backfill.py` | `modules/router/backfill.py` | `StrictBackfillController`, `BackfillConfig`, `BackfillResult`, `RouteType`, `RejectionReason`, `apply_strict_backfill`, `_FORMAT_EQUIVALENCE` |
| `src/l2w1/routing/__init__.py` | 新增 | 统一导出 Stage 6 public symbols |

清理仅限类型与工程化：

- 添加 `from __future__ import annotations`
- `Dict/List/Optional/Tuple` 现代化为 `dict/list/X | None/tuple`
- 移除未使用 import
- 修复 dataclass `None` 默认值类型，使 `mypy --strict` 可通过
- 为返回值补充类型注解

数值行为保持不变：sigmoid clip 范围、预算控制器 lambda 更新公式、熔断阈值、回填 ED 阈值 3、长度变化阈值 0.2、纯格式等价表、字段名和返回字典 key 均与旧实现 parity 对齐。

## 二、测试

新增 `tests/conftest.py`，幂等插入仓库根目录到 `sys.path`，保证 src-layout 下 parity tests 可以 import `modules.router.*`。

新增 4 组 parity tests：

- `tests/test_routing_circuit_parity.py`
- `tests/test_routing_budget_parity.py`
- `tests/test_routing_scores_parity.py`
- `tests/test_routing_backfill_parity.py`

测试均使用合成输入，同时 import 旧实现和新实现，并逐值比较输出。

## 三、`ranking.py` 暂缓说明

本阶段未创建 `src/l2w1/routing/ranking.py`。原因与 `docs/CLEANUP_STAGE6_PLAN.md` 一致：`modules/router/` 中不存在独立、可近似逐字抽取的样本排序/分档纯逻辑。强行新增会引入新行为，违反本阶段“近似逐字复制、只做纯逻辑抽取”的约束。后续 Stage 9 若从脚本薄化中识别出真实排序逻辑，再回补该模块。

## 四、验证结果

以下命令在 `l2w1` conda 环境中执行。

```bash
$ python -m pytest tests/ -q
........................................................................ [ 75%]
.......................                                                  [100%]
95 passed in 0.63s
```

```bash
$ python -m ruff check src/ tests/
All checks passed!
```

```bash
$ python -m mypy src/l2w1
Success: no issues found in 19 source files
```

## 五、边界确认

- 未修改 `modules/router/`
- 未读取或修改 `key.txt`
- 未触碰 `configs/router_config.yaml`
- 未触碰 `paper1_runs/`, `data/`, `models/`, `results/`, `outputs/`, `cloud_result_sync/`
- 未运行任何模型/API 代码
- 未提交 git commit
