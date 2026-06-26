# Stage 5 计划 — 配置管理模块 `src/l2w1/config/`

**日期：** 2026-06-26
**负责：** Claude（接口规格 + review）/ Codex（实现）
**目标：** 消灭散落在脚本里的硬编码路径，提供统一、可被环境变量覆盖的配置入口。**绝不读取 `key.txt` 内容，只传递路径。**

---

## 一、交付物

```
src/l2w1/config/
├── __init__.py          # 导出 load_settings, Settings, L2W1Paths, L2W1Secrets
└── settings.py          # 全部实现
tests/
└── test_config_settings.py   # 仅用合成 fixtures（tmp_path / monkeypatch env）
docs/CLEANUP_STAGE5_REPORT.md  # 验收报告
```

## 二、配置优先级（从高到低）

```
环境变量 (os.environ)  >  .env 文件  >  YAML 配置文件  >  代码默认值
```

- **环境变量**：前缀 `L2W1_`，如 `L2W1_DATA_ROOT`。
- **.env 文件**：简单 `KEY=VALUE` 逐行解析（标准库实现，不引第三方）。支持 `#` 注释、空行、`KEY="quoted"`。
- **YAML**：**可选层**。`settings.py` 用 `try: import yaml` 守卫；未安装 PyYAML 时静默跳过 YAML 层，不报错（核心库保持 stdlib-only，`pyproject.toml` 不新增运行期依赖）。
- **默认值**：相对 `project_root` 的相对路径（见下表）。

## 三、数据类契约（必须严格遵守）

```python
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class L2W1Paths:
    project_root: Path
    data_root: Path       # 默认 project_root / "data" / "l2w1data"
    paper1_runs: Path     # 默认 project_root / "paper1_runs"
    model_root: Path      # 默认 project_root / "models"
    results_root: Path    # 默认 project_root / "results"
    configs_root: Path    # 默认 project_root / "configs"

@dataclass(frozen=True)
class L2W1Secrets:
    provider_key_path: Path   # 默认 project_root / "key.txt"
    # ⚠️ 仅存路径。settings.py 任何地方都不得 open()/read() 这个文件。

@dataclass(frozen=True)
class Settings:
    paths: L2W1Paths
    secrets: L2W1Secrets
```

## 四、公开 API

```python
def load_settings(
    *,
    project_root: Path | None = None,   # None → 自动探测（向上找含 pyproject.toml 的目录）
    env: Mapping[str, str] | None = None,    # None → os.environ；测试时注入
    dotenv_path: Path | None = None,         # None → project_root/".env"（存在才读）
    config_path: Path | None = None,         # None → 不读 YAML；显式传入才读
) -> Settings: ...
```

### 环境变量名映射

| 字段 | 环境变量 | .env / YAML 键 |
|------|----------|----------------|
| `data_root` | `L2W1_DATA_ROOT` | `data_root` |
| `paper1_runs` | `L2W1_PAPER1_RUNS` | `paper1_runs` |
| `model_root` | `L2W1_MODEL_ROOT` | `model_root` |
| `results_root` | `L2W1_RESULTS_ROOT` | `results_root` |
| `configs_root` | `L2W1_CONFIGS_ROOT` | `configs_root` |
| `provider_key_path` | `L2W1_KEY_FILE` | `provider_key_path` |

- 所有解析出的路径统一转成 `Path`；相对路径相对 `project_root` 解析为绝对路径（用 `(project_root / value)`，绝对路径原样保留）。
- `load_settings` **不创建任何目录、不校验路径是否存在、不读取 key 文件**。

## 五、铁律（实现时必须遵守）

- ❌ 不得 `open()` / `read_text()` / 读取 `key.txt`（哪怕默认路径指向它）。
- ❌ 不得修改 `configs/router_config.yaml`、不得写入任何受保护路径。
- ❌ `pyproject.toml` 的 `dependencies` 不新增运行期依赖（YAML 仅 optional dev/extra）。
- ✅ 全部类型注解，`X | Y` 风格，强制关键字参数用 `*`。
- ✅ 通过 `ruff check`、`mypy --strict`（mypy 配置已 strict）、`pytest`。

## 六、测试要求（`tests/test_config_settings.py`）

至少覆盖：
1. 纯默认值（不传任何东西，project_root 显式给 tmp_path）→ 各路径等于预期默认。
2. 环境变量覆盖单个字段（注入 `env={"L2W1_DATA_ROOT": "/abs/x"}`）。
3. .env 文件覆盖（写一个 tmp `.env`，含注释/空行/引号值）。
4. 优先级：env > .env（同字段两边都给，env 赢）。
5. 相对路径相对 project_root 解析为绝对。
6. 绝对路径原样保留。
7. `provider_key_path` 默认指向 `project_root/key.txt`，且**测试断言文件未被打开**（可用一个不存在的 key 路径，确认 load_settings 不抛错——证明没读它）。
8. 若 PyYAML 可用：YAML 层被 env/.env 覆盖的优先级测试（用 `pytest.importorskip("yaml")` 守卫）。

---

*实现完成后由 Claude 跑 `pytest -q && ruff check src/ tests/ && mypy src/l2w1/` 验收，全绿才提交。*
