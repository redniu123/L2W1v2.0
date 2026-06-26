# Stage 9 验收报告 — 脚本层整理（薄入口 + 归类）

**日期：** 2026-06-26
**分支：** `codex-stage4-infra`
**实现：** Codex（受 Claude 规格约束）/ 验收：Claude（独立执行）

---

## 一、实际执行的移动（git mv，保留历史）

### → `scripts/dev/`（开发/冒烟/API 测试脚本，非 Paper1 复现，DEPRECATED 头 + `parents[2]` 路径修正）
```
smoke_test_agent_b.py        smoke_test_multi_vlm.py
test_single_api.py           test_provider_pools.py
test_efficiency_100.py       test_stage2_modules.py
test_stage2_integration.py   check_server_environment.py
```

### → `scripts/legacy/`（仓库根历史补丁脚本，DEPRECATED 头）
```
fix_frontier.py    fix_prompter.py
```

每个被移动文件：
- 顶部加 `# DEPRECATED (Stage 9): ...` 头注释（仅留痕，勿运行）。
- 仓库根定位从 `Path(__file__).resolve().parent.parent` 修正为 `parents[2]`（下移一层），保证 `from modules...` / `from scripts...` 仍解析到仓库根。`test_single_api.py` 用 `sys.path.insert(0,'.')` 未改。

## 二、新增交付物

- `scripts/README.md`：全部脚本归类表 + keep-set（DO NOT MOVE）标注 + 薄入口模式说明。
- `scripts/dev/__init__.py` + `scripts/dev/README.md`，`scripts/legacy/__init__.py` + `scripts/legacy/README.md`（DEPRECATED 政策说明）。
- `scripts/examples/__init__.py` + `scripts/examples/replay_offline_demo.py`：**薄入口范例**——串联 Stage 8（`CacheOnlyOCREngine` / `CacheOnlyVLMExpert`）装配回放行 + Stage 7（`router_score` / `replay_offline`）跑离线回放，全程合成内存数据，**不加载模型、不读真实数据、不调 API**，可 `python scripts/examples/replay_offline_demo.py` 运行。
- `tests/test_examples_replay_demo.py`：对范例的合成可运行性测试。

## 三、暂缓移动（写入 scripts/README.md，待 owner 复审）

下列实验 runner **本阶段保持原位**——它们之间存在 `subprocess` 互调、且 Stage 2 审计多标注 "inspect manually"，在"禁止运行"约束下无法验证移动安全性：
```
run_main_exp_b, run_main_exp_c, run_full_budget_call, run_all_frontiers,
run_gemini_ceiling, run_formal_detached, run_mainline_nightly,
run_phase_a_m5_budget_scan, run_stage2_execution, run_l2w1_pipeline
```

## 四、绝对未触碰（keep-set，被 import，留在 scripts/ 根）

```
run_efficiency_frontier.py      run_online_budget_control.py
prepare_calibration_data.py     paper1_mainA_runner.py
paper1_online_budget_check.py
```

## 五、独立验收结果（Claude 在 conda env `l2w1` 执行）

```text
python -m pytest tests/ -q
# 134 passed in 1.06s   （含新增 2 个范例测试）

python -m ruff check src/ tests/ scripts/examples
# All checks passed!

python -m mypy src/l2w1
# Success: no issues found in 34 source files

python -c "import scripts.run_efficiency_frontier, scripts.paper1_mainA_runner, \
  scripts.paper1_online_budget_check, scripts.prepare_calibration_data, \
  scripts.run_online_budget_control; print('keep-set OK')"
# keep-set OK
```

## 六、安全性结论

- 仅用 `git mv`、未删除任何文件、未改动任何业务逻辑（仅路径修正 + DEPRECATED 头）。
- 未触碰受保护路径，未运行任何模型/API 脚本。
- keep-set 与 deferred 脚本零改动；全测试套件保持绿；Paper1 复现链路（paper1_* + run_efficiency_frontier + reproduce/verify）完整保留。
