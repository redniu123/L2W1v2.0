# Stage 4 工程基础设施报告

**日期：** 2026-06-26  
**分支：** `codex-stage4-infra`  
**范围：** `pyproject.toml`、`ruff`、`mypy`、`pytest` 配置、`pre-commit`、GitHub Actions CI

---

## 一、执行摘要

Stage 4 已完成工程基础设施落地：

- 新增 `pyproject.toml`，项目支持 `src` layout 和 `pip install -e ".[dev]"`。
- 新增 `ruff`、`mypy`、`pytest` 配置。
- 新增 `.pre-commit-config.yaml`。
- 新增 `.github/workflows/ci.yml`，CI 顺序为 `ruff check` -> `ruff format --check` -> `mypy` -> `pytest`。
- 使用 ruff 对 `src/l2w1` 和 `tests` 执行了格式化和 import 排序，以确保 Stage 4 质量门为绿色。

---

## 二、交付物

| 文件 | 内容 |
|------|------|
| `pyproject.toml` | Hatchling build backend、项目元数据、dev optional dependencies、pytest/ruff/mypy 配置 |
| `.pre-commit-config.yaml` | ruff lint/format、mypy、pytest hooks |
| `.github/workflows/ci.yml` | GitHub Actions CI，Python 3.10，editable dev install 后执行质量检查 |
| `docs/CLEANUP_STAGE4_REPORT.md` | 本报告 |

---

## 三、关键配置决策

### 3.1 pytest

`pyproject.toml` 中配置：

- `testpaths = ["tests"]`，避免裸跑 `pytest` 时收集 `scripts/test_*.py` 调试/API 脚本。
- `pythonpath = ["src"]`，让未安装包时也能导入 `l2w1`。
- `addopts = "-ra --strict-config --strict-markers"`，提高配置错误可见性。

### 3.2 ruff

当前启用规则：

- `E4`, `E7`, `E9`, `F`：基础 pycodestyle/pyflakes 错误。
- `I`：import 排序。
- `UP`：Python 3.10+ 语法升级建议。

本阶段暂未启用更激进规则，避免一次性把历史脚本和模型代码纳入整改范围。

### 3.3 mypy

`mypy` 仅检查 `src/l2w1`，并启用 `strict = true`。这是为了让新抽取的核心库先达到类型检查基线，不把 `modules/` 和 `scripts/` 的历史负担带入 Stage 4。

### 3.4 CI

CI 触发条件：

- push 到非 `main` 分支。
- PR 到 `main`。

CI 命令：

```bash
ruff check src tests
ruff format --check src tests
mypy src/l2w1
pytest
```

---

## 四、验证结果

在 `/home/coder/anaconda3/envs/l2w1` 环境中完成验证。

| 命令 | 结果 |
|------|------|
| `python -m pip install -e ".[dev]"` | 通过，成功安装 editable package 和 dev tools |
| `ruff check src tests` | 通过，`All checks passed!` |
| `ruff format --check src tests` | 通过，`18 files already formatted` |
| `mypy src/l2w1` | 通过，`Success: no issues found in 12 source files` |
| `pytest -q` | 通过，`78 passed` |
| `PRE_COMMIT_HOME=/tmp/pre-commit-cache pre-commit validate-config` | 通过 |

说明：

- 初次运行 `ruff` 时发现 `tests/test_metrics_parity_evaluate.py` 和 `tests/test_metrics_reliability.py` import 排序问题，已由 `ruff check --fix src tests` 修复。
- 初次运行 `ruff format --check` 时发现 13 个文件需要格式化，已由 `ruff format src tests` 修复。
- 默认 `pre-commit validate-config` 会尝试写入 `/home/coder/.cache/pre-commit`，该路径在当前沙箱中只读；使用 `PRE_COMMIT_HOME=/tmp/pre-commit-cache` 后验证通过。

---

## 五、受保护路径检查

本阶段未读取 `key.txt` 内容，未运行任何 VLM/API/模型脚本，未主动修改受保护路径。

执行前工作区已有脏状态，包含以下受保护路径或相关路径：

- `configs/router_config.yaml` 已修改。
- `key.txt` 已修改。
- `paper1_runs/mainC/...` 有已跟踪文件删除。
- `data/`、`logs/`、`cloud_result_sync/`、`paper1_runs/` 下存在大量未跟踪内容。

上述状态为 Stage 4 开始前已存在的基线，本阶段未对这些路径执行编辑或清理。

---

## 六、后续建议

Stage 5 开始前建议先明确两件事：

1. `key.txt` 当前是 Git tracked file，`.gitignore` 无法保护已跟踪文件；需要单独制定密钥移除/轮换策略。
2. 继续保持 CI 和 pre-commit 的检查范围先聚焦 `src/l2w1` 与 `tests`，等 Stage 6-9 迁移完成后再逐步扩大到 `modules/` 和薄化后的 `scripts/`。
