# Stage 10 计划 — 工程化整理：scripts 分目录 + 脚本薄化 + 顶层清理

**日期：** 2026-06-27
**负责：** Claude（规格 + review）/ Codex（实现）
**目标：** 让 `scripts/` 从「36 个平铺脚本」变成成熟工程的「按功能分目录 + 薄入口 + 复用 `src/l2w1`」结构。参考 PaddleOCR / 典型论文开源库：**厚库（src/）+ 薄 CLI（scripts/）+ 清晰分组**。

铁律不变：不触碰受保护路径，不运行任何调模型/API 的脚本，git mv 保历史，每步测试全绿。

---

## 一、问题诊断（量化证据）

- 36 个脚本平铺在 `scripts/` 根，职责混杂（paper1 复现 / 实验 runner / 分析 / 可视化 / 数据工具 / 评估）。
- **21 个脚本各自 `sys.path.insert(parent.parent)`** —— 重复的路径 hack。
- **`router_score`/WUR 公式被复制到 6 个脚本**；**标点 `normalize_format` 映射复制到 7 个脚本** —— 但 `src/l2w1/replay/scoring.py` 已有权威实现。
- **3 个 requirements 文件**（`requirements.txt` / `_l2w1v2` / `_windows`）。
- 根目录散落中文设计/进度 md。

## 二、目标结构

```
scripts/
├── _common.py            # 统一 path bootstrap（替代 21 处 sys.path.insert hack）
├── README.md             # 分组索引 + 薄入口说明
├── paper1/               # Paper1 核心复现链路
│   ├── mainA_runner.py        (← paper1_mainA_runner.py)
│   ├── mainC_runner.py        (← paper1_mainC_runner.py)
│   ├── online_budget_check.py (← paper1_online_budget_check.py)
│   ├── upper_lower_bounds.py  (← paper1_upper_lower_bounds.py)
│   ├── merge_upper_lower_bounds.py (← paper1_merge_upper_lower_bounds.py)
│   ├── reproduce_from_frozen.py    (← reproduce_paper1_from_frozen.py)
│   └── verify_frozen_results.py    (← verify_paper1_frozen_results.py)
├── experiments/          # 实验 runner（efficiency frontier / 各 exp）
│   ├── efficiency_frontier.py (← run_efficiency_frontier.py)  ★枢纽，见 §四
│   ├── online_budget_control.py (← run_online_budget_control.py)
│   ├── all_frontiers.py / gemini_ceiling.py / full_budget_call.py
│   ├── main_exp_b.py / main_exp_c.py / phase_a_m5_budget_scan.py
│   ├── mainline_nightly.py / formal_detached.py / stage2_execution.py / l2w1_pipeline.py
├── analysis/             # 结果分析
│   ├── batch1_analysis.py / gpt_case_analysis.py (← paper1_*)
│   ├── phase_b_five_pool_analysis.py / top2_coverage.py / merge_mainc_runs.py
├── tools/                # 数据/校准/导出工具
│   ├── evaluate.py / prepare_calibration_data.py / train_calibrator.py
│   ├── split_dataset.py / adapt_geology_data.py / augment_hard_cases.py
│   ├── download_vlms.py / export_cloud_results.py / convert_old_gemini_to_mainc_v3.py
│   └── visualize_results.py / visualize_master_frontier.py
├── dev/      (已存在)
├── legacy/   (已存在)
└── examples/ (已存在)
```

> 重命名同时**去掉冗余前缀**（`paper1_mainA_runner` → `paper1/mainA_runner`，`run_efficiency_frontier` → `experiments/efficiency_frontier`），路径本身已表达归属。

## 三、统一 path bootstrap

新增 `scripts/_common.py`：
```python
"""Shared path bootstrap for script entry points."""
from __future__ import annotations
import sys
from pathlib import Path

def add_repo_root_to_path() -> Path:
    """Insert repo root + src/ into sys.path (idempotent). Returns repo root."""
    root = Path(__file__).resolve().parent.parent  # scripts/.. = repo root
    for p in (root, root / "src"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    return root
```
被移动脚本里的 `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` 一律替换为：
```python
from scripts._common import add_repo_root_to_path
add_repo_root_to_path()
```
（注意：脚本在子目录后，`_common` 仍在 `scripts/`，import 为 `from scripts._common import ...`；`_common.py` 自身用 `parent.parent` 因为它直接在 scripts/ 下。被移动脚本通过 import `_common` 间接获得正确根，不再各自算层级。）

## 四、枢纽模块 `run_efficiency_frontier`（被 13 处 import）

被以下 import：`merge_mainc_runs, paper1_mainA/C_runner, paper1_online_budget_check, run_full_budget_call, run_main_exp_b/c, run_phase_a_m5_budget_scan, run_online_budget_control, test_efficiency_100`，以及 **`tests/test_replay_offline_parity.py` / `test_replay_online_parity.py`**（我的 parity 测试）。

**策略（import-stable）：** 物理移动到 `scripts/experiments/efficiency_frontier.py`，但所有引用方一并更新为新路径 `from scripts.experiments.efficiency_frontier import ...`。**parity 测试的 import 也同步更新**——这是测试改 import，不影响被测的 `src/l2w1`，parity 关系不变。

逐个更新的引用文件清单（移动时必须全部改到）：
- `tests/test_replay_offline_parity.py`（`scripts.paper1_mainA_runner` → `scripts.paper1.mainA_runner`；`scripts.run_efficiency_frontier` → `scripts.experiments.efficiency_frontier`）
- `tests/test_replay_online_parity.py`（同理 online 版）
- `tests/test_stage2_modules.py`（在 dev/ 下，`scripts.prepare_calibration_data` → `scripts.tools.prepare_calibration_data`；该测试不在 testpaths 内，但仍修正以保持可运行）
- 所有 experiments/paper1 脚本内部的交叉 import。

## 五、脚本薄化（去重，复用 src/l2w1）

仅对 **paper1 复现链路**（有 parity 测试兜底的）做去重：
- `scripts/paper1/mainA_runner.py`、`mainC_runner.py`、`online_budget_check.py` 等里**复制粘贴的 `score()`/`norm()`** → 改为
  ```python
  from l2w1.replay.scoring import router_score, normalize_format
  ```
  删除本地重复定义，调用点改用 `router_score(strategy, row, eta=eta)` / `normalize_format(text)`。
- **数值守护：** `tests/test_replay_*_parity.py` 已断言 `router_score` == 旧 `score()`、`normalize_format` == 旧 `norm()`。薄化后这些测试必须仍全绿（证明行为不变）。
- **不动** experiments/ 里那些**无 parity 测试**的 runner 的内部逻辑（只移动 + 改 path/import），避免无测试保障的改动。

## 六、顶层清理

1. **requirements 合并**：以 `pyproject.toml` 为权威。保留 `requirements.txt`（核心运行期依赖），把 `requirements_l2w1v2.txt` / `requirements_windows.txt` 的差异并入或删除；在 README 注明「依赖以 pyproject 为准，requirements.txt 为便捷安装」。**先 diff 三者**，不丢任何真实依赖。
2. **根目录 docs 整理**：根目录散落的设计/进度 md（含中文名）→ git mv 进 `docs/`（已 gitignore 放行 `docs/**/*.md`）。仅移动文档，不改内容。

## 七、交付物

- `scripts/_common.py` + 重组后的 `scripts/{paper1,experiments,analysis,tools}/`（含各自 `__init__.py`）
- 更新后的 `scripts/README.md`（新分组索引）
- 薄化后的 paper1 runners
- 合并后的 requirements + 更新的根 README/pyproject 说明
- `docs/CLEANUP_STAGE10_REPORT.md`（移动清单、薄化 diff 摘要、验收输出）

## 八、铁律 & 验收

- ✅ 只 git mv（保历史）；不改 experiments 无测试脚本的逻辑；不碰受保护路径；不跑模型脚本。
- ✅ 每个子步骤后跑：`pytest tests/ -q` 全绿、`ruff check src/ tests/ scripts` 干净、`mypy src/l2w1` 干净。
- ✅ `import` 稳定性自检：
  ```bash
  python -c "import scripts.experiments.efficiency_frontier, scripts.paper1.mainA_runner, scripts.paper1.online_budget_check, scripts.tools.prepare_calibration_data, scripts.experiments.online_budget_control; print('imports OK')"
  ```
- ✅ parity 测试（offline/online）必须保持通过 —— 证明薄化未改数值。

## 九、执行顺序（每步独立验证 + commit）

1. 顶层清理（requirements 合并 + 根 docs 整理）→ 提交
2. `scripts/_common.py` + 建子目录 + git mv 全部脚本 + 修 path/import（**不薄化**）→ 测试全绿 → 提交
3. paper1 runners 薄化（复用 scoring）→ parity 测试全绿 → 提交
4. 更新 scripts/README + 写 Stage10 报告 → 提交
5. 最终全量验收 + 刷新 `clean-public` 分支
