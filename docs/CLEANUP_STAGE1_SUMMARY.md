# Cleanup Stage 1 Summary

生成日期：2026-06-01

## 本阶段目标

本阶段目标是为公司服务器迁移和安全复现做准备。范围仅限新增文档、只读检查脚本、只读验证脚本和服务器运行说明。

本阶段没有重构代码，没有删除文件，没有移动或重命名已有核心文件，也没有调用任何 VLM API。

## 新增文件

| 文件 | 用途 |
|---|---|
| `docs/SERVER_REPRODUCTION_GUIDE.md` | 公司服务器迁移和复现指南，说明环境、数据、模型、frozen results、运行顺序、API 风险和常见错误。 |
| `docs/SERVER_RERUN_PLAN.md` | 分 Level 0-5 的服务器重跑计划，明确每级目的、输入、输出、命令、API 风险和验收标准。 |
| `scripts/check_server_environment.py` | 只读环境检查脚本，检查 Python、依赖导入、GPU 可见性、数据/模型/frozen cache 路径和 test 行数。 |
| `scripts/verify_paper1_frozen_results.py` | 只读 frozen results 完整性验证脚本，输出 Markdown 报告到 `server_reproduction_runs/frozen_verification_report.md`。 |
| `scripts/reproduce_paper1_from_frozen.py` | 从 frozen CSV 生成最小复现包，输出到 `server_reproduction_runs/from_frozen/`。 |
| `scripts/README.md` | scripts 目录分类说明，标注 paper1 final、legacy、debug/test/smoke、服务器安全脚本。 |
| `docs/CLEANUP_STAGE1_SUMMARY.md` | 本阶段整理总结。 |

## 没有修改的核心逻辑

本阶段未修改以下核心实验脚本：

- `scripts/paper1_mainA_runner.py`
- `scripts/paper1_mainC_runner.py`
- `scripts/paper1_batch1_analysis.py`
- `scripts/paper1_gpt_case_analysis.py`
- `scripts/paper1_upper_lower_bounds.py`

本阶段未修改以下核心目录内已有文件：

- `paper1_workspace/02_frozen_results/`
- `paper1_workspace/03_figures_tables/`
- `data/`
- `models/`
- `modules/`
- `scripts/` 中的既有核心脚本

新增脚本只读取 frozen results 或检查路径，默认不覆盖已有结果。

## 公司服务器推荐运行顺序

从项目根目录运行：

```bash
python scripts/check_server_environment.py
python scripts/verify_paper1_frozen_results.py
python scripts/reproduce_paper1_from_frozen.py
```

预期新增输出：

```text
server_reproduction_runs/
├── frozen_verification_report.md
└── from_frozen/
    ├── copied_or_reproduced_tab_mainA_results.csv
    ├── copied_or_reproduced_tab_mainC_results.csv
    ├── copied_or_reproduced_budget_check_summary.csv
    └── reproduction_summary.md
```

这些命令不会调用 OCR，不会调用 VLM API，不会读取或打印 key 文件内容。

## 预计最可能出现的错误

| 错误 | 可能原因 | 处理方式 |
|---|---|---|
| `test.jsonl missing` | 数据目录未复制或当前工作目录不是项目根目录 | 确认 `data/l2w1data/test.jsonl` 存在，从项目根目录运行 |
| `expected 3424 non-empty lines` | test split 版本不对或文件传输不完整 | 使用 paper1 frozen 对应的 test split |
| `official Agent A cache missing` | `paper1_workspace/02_frozen_results/` 未完整复制 | 重新复制 frozen results |
| Main A full-call cache 行数不是 3424 | JSONL 被截断或换行损坏 | 重新传输并校验文件大小/行数 |
| `paddle import failed` | Paddle 未安装或 CUDA 版本不匹配 | 安装服务器 CUDA 对应的 PaddlePaddle |
| GPU 不可见 | 容器未挂 GPU、驱动问题、无 GPU 机器 | Level 0-3 可继续；Level 4-5 前需修复 |
| `cv2/yaml/Levenshtein/pandas import failed` | Python 依赖未安装 | 安装 `requirements.txt` 或公司环境对应依赖 |
| CSV 无法读取 | frozen CSV 文件损坏或编码问题 | 重新复制对应 CSV |

## 服务器运行失败时请发回的信息

请发回：

- `python scripts/check_server_environment.py` 的完整控制台输出；
- `server_reproduction_runs/frozen_verification_report.md`；
- `server_reproduction_runs/from_frozen/reproduction_summary.md`；
- `python --version`；
- `pip freeze`；
- 如涉及 GPU：`nvidia-smi` 输出；
- 失败命令的当前工作目录和完整命令行。

不要发回：

- `key.txt`
- `GPTkey.txt`
- `GPT.config`
- 任何包含 `sk-` token、Bearer token、API response secret 的日志

## 下一步建议

1. 先在公司服务器跑 Level 0-2。
2. 若 Level 0-2 全部通过，再新增或审核 Level 3 cache-only replay wrapper。
3. Level 4 小样本 Agent A OCR 只在 Paddle/GPU 环境稳定后运行。
4. Level 5 全量重跑需要单独审批 API 或本地 VLM 资源，且必须写入 `server_reproduction_runs/full_rerun/`。
