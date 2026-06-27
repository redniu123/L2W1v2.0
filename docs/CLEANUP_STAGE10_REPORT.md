# Stage 10 验收报告 — 工程化整理

**日期：** 2026-06-27
**分支：** `codex-stage4-infra`
**实现：** Codex（受 Claude 规格约束）/ 验收：Claude（独立执行）

---

## 一、做了什么

| 步骤 | commit | 内容 |
|------|--------|------|
| 顶层清理 | `1f4b360` | 合并 3 个 requirements → 保留权威 `requirements.txt`；旧中文 README → 工程化英文 README；删除过期根 md |
| scripts 分目录 | `8f0ed87` | 36 个平铺脚本 → `paper1/ experiments/ analysis/ tools/`；统一 path bootstrap；改全部 import |
| paper1 薄化 | `b35b855` | mainA/online runner 的 `main()` 改用 `l2w1.replay`，保留黄金参照 |

## 二、scripts/ 重组结果

```
scripts/
├── _common.py            # 统一 add_repo_root_to_path()（替代 21 处 sys.path hack，残留 0）
├── README.md
├── paper1/        (7)    mainA/mainC_runner, online_budget_check, upper_lower_bounds,
│                         merge_upper_lower_bounds, reproduce_from_frozen, verify_frozen_results
├── experiments/   (12)   efficiency_frontier, online_budget_control, all_frontiers,
│                         gemini_ceiling, full_budget_call, main_exp_b/c, phase_a_m5_budget_scan,
│                         mainline_nightly, formal_detached, stage2_execution, l2w1_pipeline
├── analysis/      (5)    batch1_analysis, gpt_case_analysis, phase_b_five_pool_analysis,
│                         top2_coverage, merge_mainc_runs
├── tools/         (11)   evaluate, prepare_calibration_data, train_calibrator, split_dataset,
│                         adapt_geology_data, augment_hard_cases, download_vlms, export_cloud_results,
│                         convert_old_gemini_to_mainc_v3, visualize_results, visualize_master_frontier
├── examples/             replay_offline_demo（薄入口范例）
├── dev/ (8)  legacy/ (2)
```

- **scripts/ 根仅剩** `_common.py` + `__init__.py`（原 36 个散脚本全部归类）。
- **37 个 git rename**（历史保留）；**21 处 sys.path hack → 0**，26 个脚本改用 `_common`。
- 全部交叉 import + 2 个 parity 测试的 import 同步更新到新路径。

## 三、paper1 薄化（去重，复用库）

- `paper1/mainA_runner.py main()`：score map 改用 `l2w1.replay.scoring.router_score`；`replay(...)` 调用改为 `l2w1.replay.offline.replay_offline(..., extended_metrics_fn=, usage_metrics_fn=)`。
- `paper1/online_budget_check.py main()`：`run_online_routeronly(...)` 改为 `l2w1.replay.online_budget.replay_online(...)`。
- **保留**本地 `score/norm/replay/build_router_score/run_online_routeronly` 函数体作为 parity 测试的**黄金参照**——`git diff` 确认这些函数签名零改动。
- **数值守护**：`tests/test_replay_{offline,online}_parity.py`（25 tests）断言库版 == 旧版，薄化后仍全绿，证明行为不变。
- **已知局限**：`main()` 需加载 PaddleOCR/调 Agent B，按铁律无法运行，故 `main()` 接线未端到端测试；正确性由"每个被替换函数均 parity 证明等价"保证。

## 四、独立验收（Claude，conda env `l2w1`）

```text
python -m pytest tests/ -q                         # 134 passed
python -m pytest tests/test_replay_*_parity.py -q   # 25 passed（黄金参照仍有效）
python -m ruff check src/ tests/ scripts            # All checks passed!
python -m mypy src/l2w1                              # Success: no issues (34 files)
python -c "import scripts.experiments.efficiency_frontier, scripts.paper1.mainA_runner, ..."  # imports OK
```

## 五、安全

- 全程未触碰受保护路径，未运行任何模型/API 脚本，git mv 保历史。
- 黄金参照函数零改动 → parity 测试的回归保护价值完整保留。
