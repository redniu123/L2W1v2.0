# Stage 9 计划 — 脚本层整理（薄入口 + 归类）`scripts/`

**日期：** 2026-06-26
**负责：** Claude（规格 + review）/ Codex（实现）
**目标：** 在**不破坏 Paper1 复现、不运行任何模型脚本**的前提下，给 `scripts/` 建立清晰结构：归类索引、隔离 dev/legacy、提供薄入口范例。**保守可逆**——只用 `git mv`、不删除、不改业务逻辑，移动后必须**全测试绿**。

---

## 一、绝对不可移动（keep-set，被依赖）

以下脚本被其它脚本或测试 import，**必须留在 `scripts/` 根**且保持可 import：

```
run_efficiency_frontier.py      （被 ~10 脚本 + 测试 import）
run_online_budget_control.py    （被 run_main_exp_b/c import）
prepare_calibration_data.py     （被 test_stage2_modules import）
paper1_mainA_runner.py          （被 test_replay_offline_parity import）
paper1_online_budget_check.py   （被 test_replay_online_parity import）
```

另外保持原位（Paper1 核心 / 复现 / 分析 / 数据工具，本阶段不动）：
`paper1_mainC_runner, paper1_upper_lower_bounds, paper1_merge_upper_lower_bounds, paper1_batch1_analysis, paper1_gpt_case_analysis, reproduce_paper1_from_frozen, verify_paper1_frozen_results, evaluate, merge_mainc_runs, convert_old_gemini_to_mainc_v3, export_cloud_results, visualize_results, visualize_master_frontier, adapt_geology_data, split_dataset, augment_hard_cases, check_top2_coverage, train_calibrator, download_vlms, run_phase_b_five_pool_analysis`。

## 二、本阶段执行的移动

### → `scripts/dev/`（开发/冒烟/API 测试脚本，非 Paper1 复现）
```
smoke_test_agent_b.py   smoke_test_multi_vlm.py
test_single_api.py      test_provider_pools.py
test_efficiency_100.py  test_stage2_modules.py
test_stage2_integration.py   check_server_environment.py
```

### → `scripts/legacy/`（仓库根的历史补丁脚本）
```
fix_frontier.py    fix_prompter.py     （位于仓库根目录）
```

**移动后必修**：这些文件用 `Path(__file__).resolve().parent.parent` 定位仓库根；下移一层后必须改为 `Path(__file__).resolve().parents[2]`（dev/ 与 legacy/ 均在 `scripts/<sub>/` 下，距根两级→需要 parents[2]；`fix_*.py` 原在根，移入 `scripts/legacy/` 后同样 parents[2]）。`test_single_api.py` 用 `sys.path.insert(0,'.')`（基于 cwd）不用改但保留。

**每个被移动文件**：在 shebang/编码声明之后插入头注释：
```python
# DEPRECATED (Stage 9): moved out of the active scripts/ root for organization.
# Kept for traceability only. Do NOT run during cleanup (loads models / calls APIs).
```

## 三、暂缓移动（写入 scripts/README.md 供 owner 复审）

以下实验 runner **本阶段不移动**：它们之间有 `subprocess` 互调、且 Stage 2 审计多标注 "inspect manually"，无法在禁跑约束下验证移动安全性。
```
run_main_exp_b, run_main_exp_c, run_full_budget_call, run_all_frontiers,
run_gemini_ceiling, run_formal_detached, run_mainline_nightly,
run_phase_a_m5_budget_scan, run_stage2_execution, run_l2w1_pipeline
```

## 四、新增交付物

- `scripts/README.md`：全部 44 个脚本的归类表（核心/复现/分析/可视化/数据工具/dev/legacy/暂缓）+ keep-set 标注 + 薄入口模式说明。
- `scripts/dev/README.md` + `scripts/dev/__init__.py`，`scripts/legacy/README.md` + `scripts/legacy/__init__.py`（说明：DEPRECATED，仅留痕，勿运行）。
- `scripts/examples/__init__.py` + `scripts/examples/replay_offline_demo.py`：**薄入口范例**——只用 `l2w1.config` + `l2w1.replay` + `l2w1.ocr/vlm` 的 cache_only/mock，在**合成内存数据**上跑离线回放，演示"业务逻辑全在 src/、脚本只做装配"的目标形态。**不加载模型、不读真实数据**。
- `tests/test_examples_replay_demo.py`：对该范例的可运行性做合成测试。
- `docs/CLEANUP_STAGE9_REPORT.md`：实际移动清单、暂缓清单及理由、最终验收输出。

## 五、铁律

- ✅ 只用 `git mv`（保留历史）；不删除、不改动业务逻辑（除必需的 `parents[2]` 路径修正与 DEPRECATED 头注释）。
- ❌ 不触碰 keep-set 与暂缓清单；不触碰受保护路径；不运行任何模型/API 脚本。
- ❌ 不移动任何被 import 的脚本（移动前对每个候选执行引用检查：`grep -rn "scripts.<name>\|scripts import <name>\|scripts/<name>.py" scripts tests .github *.py`，若有来自 keep 文件的引用则放弃移动该文件）。
- ✅ 移动后 `python -m pytest tests/ -q` 必须保持全绿（132+ 新增范例测试）；`ruff check src/ tests/` 与新增/移动文件、`mypy src/l2w1` 全通过。

## 六、验收命令（Claude 独立执行）

```bash
python -m pytest tests/ -q
python -m ruff check src/ tests/ scripts/examples
python -m mypy src/l2w1
python -c "import scripts.run_efficiency_frontier, scripts.paper1_mainA_runner, scripts.paper1_online_budget_check, scripts.prepare_calibration_data, scripts.run_online_budget_control; print('keep-set still importable')"
```
全绿后提交。
