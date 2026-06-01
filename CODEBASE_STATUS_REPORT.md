# L2W1 代码现状报告

生成日期：2026-06-01  
审计范围：`G:\Code\PaddleOCR\L2W1`

> 说明：本报告基于只读命令完成目录、源码、配置、README、论文工作区、冻结实验结果、提交包、缓存和结果文件审计。图像、PDF、DOCX、模型权重等二进制文件未逐页/逐像素解析，但已按路径、体量、引用关系和文件类型纳入分类判断。报告只新增本文件，不修改、删除、移动、重命名任何已有项目文件。

---

## 一、项目总体概述

### 1.1 项目大概在做什么

该项目是一个面向中文单行手写 OCR 的选择性 VLM 纠错论文项目。核心问题不是让 VLM 处理所有样本，而是先用 OCR 模型生成初稿和置信度特征，再用 Router 在预算约束下选择一部分高风险样本交给 VLM 纠错，最后用离线 replay 评估不同预算和不同路由策略的 CER 改善。

从现有论文工作区、脚本和结果看，当前 paper1 的主任务可概括为：

> 在 geology / finance / medicine 三个领域的 L2W1 多领域中文手写 OCR 数据上，比较 GCR、WUR、DGCR、DWUR、MinConf、Random 等选择性纠错策略，并比较 Qwen、MiniCPM、Gemini、GPT 等 VLM correction 后端在相同 call-rate budget 下的行为差异。

### 1.2 核心研究流程

核心研究流程为：

1. 数据准备：读取 `data/l2w1data/train.jsonl`、`data/l2w1data/val.jsonl`、`data/l2w1data/test.jsonl` 和 `data/l2w1data/images/`。
2. Agent A OCR：用 PP-OCRv5 recognition 模型生成 `T_A`、置信度、top2、边界统计、领域风险等特征。
3. Router score：在 VLM 调用前计算 GCR / WUR / DGCR / DWUR priority score。
4. VLM full-call cache：对测试集样本进行一次 full-call correction，保存每个样本“如果升级”的 VLM 输出。
5. Offline dense-budget replay：按 router score 排序，对不同 budget 取 top-k 样本升级，其余样本保留 OCR 输出。
6. 指标计算：计算 CER、BoundaryDeletionRecallAtB、SubstitutionCER、AER、budget compliance、分领域指标、cross-model 指标、casebook。
7. 论文图表：从 `paper1_workspace/02_frozen_results/` 读取冻结结果，生成 `paper1_workspace/03_figures_tables/` 中的表格和主图。

### 1.3 与论文实验最相关的主流程

最相关、最应视为 paper1 正式主流程的是：

| 环节 | 核心文件/目录 | 状态判断 |
|---|---|---|
| 正式数据 | `data/l2w1data/train.jsonl`、`data/l2w1data/val.jsonl`、`data/l2w1data/test.jsonl`、`data/l2w1data/images/` | 论文复现核心数据 |
| 正式 OCR 起点 | `paper1_workspace/02_frozen_results/official_agent_a_cache/paper1_official_agent_a_cache.json` | 全论文唯一 OCR cache 起点 |
| Main A router replay | `paper1_workspace/02_frozen_results/mainA_final/` | Figure 2、Table 2、domain 结果核心来源 |
| Main C cross-model | `paper1_workspace/02_frozen_results/mainC_final/` | Figure 3、Table 3 核心来源 |
| Random / MinConf / domain / cost 补充 | `paper1_workspace/02_frozen_results/phase2_batch1_final/` | baseline、domain、cost-effectiveness 来源 |
| GPT 负优化案例 | `paper1_workspace/02_frozen_results/phase2_batch2_final/` | Appendix / mechanism analysis 来源 |
| A-only / B-only / A+B bounds | `paper1_workspace/02_frozen_results/upper_lower_bounds_final/` | Figure 4 / Table 4 来源 |
| 图表脚本 | `paper1_workspace/03_figures_tables/` | 论文图表生成入口 |
| 投稿材料 | `PRL_submission_package/` | 已打包论文、图、补充材料 |

### 1.4 项目形态判断

结论：当前项目是“已有冻结复现材料的论文项目 + 大量开发过程遗留代码集合”的混合状态。

有利于复现的部分：

- `paper1_workspace/README.md` 明确规定后续论文默认引用 `paper1_workspace/02_frozen_results/`。
- `paper1_workspace/01_registry/正式结果索引.md` 明确列出了正式结果的原始 run 和工作区副本。
- `paper1_workspace/03_figures_tables/fig_source_map.md` 和 `table_source_map.md` 明确锁定了主文 Figures / Tables 的数据来源。
- 冻结结果中保留了 `manifest.json`、`config_snapshot.yaml`、CSV summary、JSONL per-sample replay 文件。

不利于复现的部分：

- `scripts/` 同时包含早期 stage2、主线 v5.1、paper1 final、debug、smoke、test、转换、合并、云同步等多类脚本。
- `results/expriments/`、`paper1_runs/`、`paper1_runs_old/`、`cloud_result_sync/`、`paper1_workspace/02_frozen_results/`、`PRL_submission_package/` 存在多套相似结果。
- 若不阅读 `paper1_workspace/01_registry/`，很难判断哪套结果是论文最终引用版本。
- 配置、manifest 和脚本中存在 Windows / Linux 绝对路径、API key 文件、硬编码 token、旧默认路径。

---

## 二、目录结构分析

### 2.1 顶层目录体量

| 路径 | 文件数 | 体量 | 分类 | 用途判断 |
|---|---:|---:|---|---|
| `.git/` | 18942 | 约 1130 MB | 版本库 | Git 元数据，报告不分析为项目内容 |
| `.codex/` | 135 | 约 0.65 MB | 工具/技能 | Codex 本地技能，不属于论文代码 |
| `.codex_work/` | 302 | 约 3.19 MB | 临时工作区 | 文档/PDF 渲染缓存，非论文实验核心 |
| `archive/` | 10 | 约 0.02 MB | 归档/废弃候选 | 早期 API 测试与 module spec |
| `cloud_result_sync/` | 497 | 约 494 MB | 云端结果同步 | exB / phaseA 结果副本，追溯用 |
| `configs/` | 3 | 极小 | 配置 | 当前主要只有 `router_config.yaml` |
| `data/` | 15423 | 约 296 MB | 数据 | L2W1 数据和领域词典，论文复现核心 |
| `docs/` | 14 | 约 0.13 MB | 开发文档 | stage/spec/进度文档，多为开发过程记录 |
| `logs/` | 2 | 约 0.03 MB | 日志 | frontier / gemini ceiling 日志，非核心 |
| `models/` | 6 | 约 81 MB | 模型 | PP-OCRv5 inference model，VLM 权重目录占位/本地依赖 |
| `modules/` | 59 | 约 0.72 MB | 核心模块 | Paddle OCR、Router、VLM expert、utils |
| `outputs/` | 1 | 极小 | 旧输出 | 基本空，疑似早期输出目录 |
| `paper1_runs/` | 163 | 约 478 MB | 原始正式/中间 run | paper1 final 的原始 run 与中间版本 |
| `paper1_runs_old/` | 171 | 约 448 MB | 旧 run | upper/lower bounds 等旧版来源，追溯用 |
| `paper1_workspace/` | 509 | 约 929 MB | 论文工作区 | 当前 paper1 复现、图表、写作核心 |
| `ppocr/` | 11 | 约 0.14 MB | PaddleOCR 兼容代码 | 字典、postprocess、utility |
| `PRL_submission_package/` | 154 | 约 437 MB | 投稿包 | Manuscript、Figures、Supplementary |
| `results/` | 152 | 约 57 MB | 早期结果 | exA/exB/stage2，非 paper1 最终引用 |
| `scripts/` | 66 | 约 0.94 MB | 脚本入口 | 训练/评估/runner/分析/调试混合 |
| `tools/` | 4 | 约 0.02 MB | PaddleOCR infer 工具 | 检测/推理辅助 |

### 2.2 主要文件类型

| 类型 | 数量 | 判断 |
|---|---:|---|
| `.jpg` | 15436 | 数据图像主体，主要在 `data/l2w1data/images/` |
| `.jsonl` | 658 | 数据划分、per-sample replay、full-call cache |
| `.csv` | 455 | 表格结果、figure plotdata、case pool |
| `.md` | 211 | 文档、论文草稿、结果索引、进度记录 |
| `.json` | 181 | manifest、cache、配置快照、API/model metadata |
| `.py` | 116 | 源码、runner、图表脚本、文档生成脚本 |
| `.pdf` | 92 | 论文、图、参考文献、投稿材料 |
| `.yaml` | 42 | 配置快照和实验配置 |
| `.png` / `.svg` | 41 / 20 | 图表输出和视觉素材 |

### 2.3 主要目录用途与分类

| 路径 | 大致用途 | 分类建议 |
|---|---|---|
| `configs/router_config.yaml` | 主配置：数据路径、Agent B、prompt、formal budgets、router 参数 | 配置目录，强相关 |
| `data/l2w1data/` | 论文数据集：images + train/val/test JSONL | 数据目录，强相关 |
| `data/dicts/` | `Geology.txt`、`Finance.txt`、`Medicine.txt` 领域词典 | 数据/配置混合，强相关 |
| `models/agent_a_ppocr/PP-OCRv5_server_rec_infer/` | Agent A PP-OCRv5 inference model 与 `inference.yml/json` | 模型目录，强相关但环境相关 |
| `modules/paddle_engine/` | 修改后的 PaddleOCR recognition 推理与 logits/置信度导出 | 核心代码 |
| `modules/router/` | Router、budget controller、backfill、domain risk、circuit breaker | 核心代码 + 原型混合 |
| `modules/vlm_expert/` | Gemini/local VLM expert、provider pool、prompt | 核心代码 |
| `scripts/paper1_*.py` | paper1 final runner / 分析脚本 | 论文实验入口，强相关 |
| `scripts/run_*.py` | 早期或泛化实验 runner | 部分强相关，部分旧版 |
| `paper1_workspace/02_frozen_results/` | 正式冻结结果副本 | 结果目录，最强相关 |
| `paper1_workspace/03_figures_tables/` | 主图、主表、plotdata、生成脚本 | 图表目录，最强相关 |
| `paper1_workspace/04_drafts/` | 论文草稿、参考文献、修订脚本 | 写作目录 |
| `PRL_submission_package/` | PRL 投稿包、补充材料、结果副本 | 发布/归档目录，应保留 |
| `paper1_runs/` | final run 的原始目录和中间 run | 结果追溯目录 |
| `paper1_runs_old/` | 旧 run、online validation、upper/lower bounds 原始来源 | 旧结果追溯目录 |
| `results/expriments/` | exA/exB 早期主实验结果，目录名有 typo | 早期结果，归档候选 |
| `cloud_result_sync/` | 云端同步结果，exB 和 phaseA 大量 summary | 云同步缓存/追溯 |
| `archive/` | 早期 spec 与 API 测试 | 废弃/历史归档 |
| `logs/` | 运行日志 | 缓存/日志 |
| `__pycache__/` | Python 字节码缓存 | 缓存，应从论文材料排除 |

---

## 三、核心代码入口分析

### 3.1 最可能的主入口脚本

| 入口 | 角色 | 输入 | 输出 | 判断依据 |
|---|---|---|---|---|
| `scripts/paper1_mainA_runner.py` | paper1 Main A：RouterOnly dense budget replay | `configs/router_config.yaml`、`data/l2w1data/test.jsonl`、`data/l2w1data/images/`、PP-OCRv5 model、领域词典、可选 shared Agent A cache | `tab_mainA_results.csv`、`tab_mainA_domain_results.csv`、`tab_mainA_budget_check.csv`、`router_score_matrix.csv`、`shared_repmodel_full_call_cache.jsonl`、`offline_budget_*.jsonl` | 与 `paper1_workspace/02_frozen_results/mainA_final/` 完全对应 |
| `scripts/paper1_mainC_runner.py` | paper1 Main C：cross-model under same router | 同 Main A 的 Agent A cache；V1/V2/V3/V4 full-call cache 或 API/local VLM | `tab_mainC_results.csv`、`tab_mainC_budget_check.csv`、`V*_full_call_cache.jsonl`、`V*_offline_budget_*.jsonl` | 与 `mainC_final/`、Figure 3 / Table 3 对应 |
| `scripts/paper1_batch1_analysis.py` | Random / MinConf / domain / cost 补充分析 | Main A run、Main C run | `tab_random_baseline_results.csv`、`tab_minconf_baseline_results.csv`、`tab_domain_budget_curve.csv`、`tab_domain_router_ranking.csv`、`tab_cost_effectiveness.csv` | 与 `phase2_batch1_final/` 对应 |
| `scripts/paper1_gpt_case_analysis.py` | GPT 负优化和 casebook | Main A case pool、Main C V1-V4 budget replay | `gpt_error_bucket_stats.csv`、`model_error_style_comparison.csv`、`main_casebook.csv`、`figure_case_samples.md` | 与 `phase2_batch2_final/` 对应 |
| `scripts/paper1_upper_lower_bounds.py` | A-only / B-only / A+B bounds | Main A full-call cache、test jsonl、图像路径、Gemini/local VLM | `A_only_outputs.jsonl`、`B_only_recognition_outputs.jsonl`、`A_plus_B_correction_outputs.jsonl`、`tab_upper_lower_bounds.csv` | Figure 4 / Table 4 对应 |
| `scripts/paper1_merge_upper_lower_bounds.py` | 合并 bounds base / patch run | 两个 bounds run 目录 | merged `tab_upper_lower_bounds.csv` 等 | `upper_lower_bounds_final/manifest.json` 指向 merged 来源 |
| `paper1_workspace/03_figures_tables/*/make_*.py` | 主文图表生成 | `02_frozen_results/` 中 CSV/JSONL | PDF/SVG/PNG/CSV/TEX/MD | `fig_source_map.md`、`table_source_map.md` 锁定来源 |

### 3.2 训练、推理、OCR、VLM、Router、预算、评估、统计、画图脚本

| 类型 | 文件 | 现状 |
|---|---|---|
| OCR 推理 | `modules/paddle_engine/predict_rec_modified.py` | 修改版 PP-OCR recognition，负责 logits、softmax、置信度、边界统计等；文件长约 1406 行 |
| OCR cache 生成 | `scripts/paper1_mainA_runner.py`、`scripts/run_efficiency_frontier.py` | final paper 使用前者；后者是 v5.1 grand loop / 工具函数来源 |
| VLM 调用 | `modules/vlm_expert/gemini_expert.py`、`modules/vlm_expert/qwen_expert.py`、`minicpm_expert.py`、`internvl_expert.py`、`llava_expert.py`、`smolvlm_expert.py`、`paddleocr_vl_expert.py` | 多后端实现；paper1 final 主要依赖 Gemini API full-call cache 和 Main C V1-V4 cache |
| Provider / API key | `modules/vlm_expert/provider_pools.py` | 从 `key.txt` 解析 provider pool；复现有安全和环境风险 |
| Router score | `scripts/paper1_mainA_runner.py`、`scripts/paper1_mainC_runner.py`、`scripts/paper1_online_budget_check.py`、`modules/router/uncertainty_router.py` | final paper 的 GCR/WUR/DGCR/DWUR 公式主要写在 paper1 runner 内；模块中还有更完整 SH-DA++ 原型 |
| Budget replay | `scripts/paper1_mainA_runner.py`、`scripts/paper1_mainC_runner.py` | offline top-k replay；Main A dense budgets，Main C cross-model budgets |
| Online budget | `scripts/paper1_online_budget_check.py`、`modules/router/uncertainty_router.py` | 用缓存验证 online controller 合法性；不是主文核心结论 |
| Backfill | `modules/router/backfill.py` | 原型/兼容模块；paper1 RouterOnly 主流程中标记 `backfill_status=skipped` |
| Circuit breaker | `modules/router/circuit_breaker.py` | 原型功能；paper1 主图要求 no breaker |
| 训练/calibrator | `scripts/train_calibrator.py`、`scripts/prepare_calibration_data.py` | calibration/scorer 相关，final config 中 `calibrated_scorer.enabled: false` |
| 评估通用工具 | `scripts/evaluate.py` | 早期/通用 evaluator，包含 CER/OCR-R/CVR/AER/latency/budget stability；final paper runner 内部重复实现了部分指标 |
| 结果统计 | `scripts/paper1_batch1_analysis.py`、`scripts/paper1_gpt_case_analysis.py`、`scripts/paper1_online_budget_check.py` | paper1 二阶段分析强相关 |
| 画图 | `paper1_workspace/03_figures_tables/fig2/make_fig2_budget_frontier.py`、`fig3/make_fig3_cross_model_behavior.py`、`fig4/make_fig4_collaboration_bounds.py`、`fig5/make_fig5_domain_small_multiples.py`、`nature_figures/make_nature_figures.py` | 主图生成，读取冻结结果 |
| 表格 | `paper1_workspace/03_figures_tables/tables/Table 1/2/3/make_*.py` | 主表生成，读取数据划分和冻结结果 |

### 3.3 入口运行顺序建议

如果要复现 paper1 当前论文结果，应优先使用冻结结果，不建议直接重跑 API。若必须从头重跑，合理顺序是：

1. 准备环境和数据：`data/l2w1data/`、`models/agent_a_ppocr/PP-OCRv5_server_rec_infer/`、`ppocr/utils/ppocrv5_dict.txt`。
2. 生成或复用正式 Agent A cache：目标文件为 `paper1_workspace/02_frozen_results/official_agent_a_cache/paper1_official_agent_a_cache.json`。
3. 跑 Main A：`scripts/paper1_mainA_runner.py`，复用 official Agent A cache，生成 `mainA_final` 类文件。
4. 跑 Main C：`scripts/paper1_mainC_runner.py`，复用同一 Agent A cache 和 V1-V4 full-call cache，生成 cross-model replay。
5. 跑 baseline / domain / cost：`scripts/paper1_batch1_analysis.py`。
6. 跑 GPT/case 分析：`scripts/paper1_gpt_case_analysis.py`。
7. 跑 upper/lower bounds：`scripts/paper1_upper_lower_bounds.py`，必要时用 `paper1_merge_upper_lower_bounds.py` 合并。
8. 生成图表：`paper1_workspace/03_figures_tables/*/make_*.py`。
9. 打包补充材料：参考 `PRL_submission_package/03_Supplementary_material/`。

不确定点：

- `paper1_mainC_runner.py` 的 `--reuse_full_call_cache` 使用 `action='store_true', default=True`，命令行层面几乎总是复用缓存；如果用户期望强制新调用，需要改脚本或新增显式 `--no-reuse`，当前不确定是否为有意设计。
- `paper1_workspace/02_frozen_results/mainC_final/manifest.json` 显示 V1-V4 来源来自多个 Main C run，不是单次统一 run；这对最终结果有效性不一定有问题，但需要在复现说明中明确。

---

## 四、论文实验复现链路分析

### 4.1 从原始数据到论文结果的完整链路

```text
data/l2w1data/*.jsonl + images
    ↓
Agent A: PP-OCRv5 recognition
modules/paddle_engine/predict_rec_modified.py
    ↓
official Agent A cache
paper1_workspace/02_frozen_results/official_agent_a_cache/paper1_official_agent_a_cache.json
    ↓
Router pre-VLM score
GCR / WUR / DGCR / DWUR
    ↓
VLM full-call correction cache
mainA_final/shared_repmodel_full_call_cache.jsonl
mainC_final/V1..V4_full_call_cache.jsonl
    ↓
Offline dense-budget replay
offline_budget_*.jsonl
    ↓
Summary tables and case pools
tab_mainA_results.csv, tab_mainC_results.csv, tab_random_baseline_results.csv, ...
    ↓
Figures / Tables
paper1_workspace/03_figures_tables/
    ↓
Submission package
PRL_submission_package/
```

### 4.2 OCR draft 如何产生

Agent A 的核心实现位于：

- `modules/paddle_engine/predict_rec_modified.py`
- `ppocr/postprocess/rec_postprocess.py`
- `ppocr/utils/ppocrv5_dict.txt`
- `models/agent_a_ppocr/PP-OCRv5_server_rec_infer/inference.yml`

正式 OCR cache 位于：

- `paper1_workspace/02_frozen_results/official_agent_a_cache/paper1_official_agent_a_cache.json`

该 cache 的 manifest：

- `paper1_workspace/02_frozen_results/official_agent_a_cache/manifest.json`
- 记录 `n_samples = 3424`
- 原始路径为 `/home/coder/project/L2W1v2.0/paper1_runs/shared_agent_a_cache/official_20260421_run142031/paper1_official_agent_a_cache.json`

正式 cache 常用字段包括：

| 字段 | 含义 |
|---|---|
| `sample_id` | 样本唯一 ID |
| `source_image_id` | 原始图像/页面来源 |
| `domain` | `geology` / `finance` / `medicine` |
| `image_path` / `img_path` | 图像路径 |
| `T_A` | Agent A OCR draft |
| `T_GT` | ground truth |
| `conf`、`mean_conf`、`min_conf` | 样本/字符置信度 |
| `drop` | 边界 blank / 不稳定度相关信号 |
| `r_d` | 领域风险分数 |
| `top2_info`、`boundary_stats` | top-2 和边界统计 |

样本规模：

| split | 文件 | 行数 |
|---|---|---:|
| train | `data/l2w1data/train.jsonl` | 8522 |
| val | `data/l2w1data/val.jsonl` | 3470 |
| test | `data/l2w1data/test.jsonl` | 3424 |

### 4.3 VLM correction 如何调用或读取缓存

Main A：

- 调用入口：`scripts/paper1_mainA_runner.py`
- VLM 调用函数来自 `scripts/run_efficiency_frontier.py::build_agent_b_callable`
- Gemini API 实现：`modules/vlm_expert/gemini_expert.py`
- provider pool：`modules/vlm_expert/provider_pools.py`
- 正式 full-call cache：`paper1_workspace/02_frozen_results/mainA_final/shared_repmodel_full_call_cache.jsonl`
- 行数：3424

Main C：

- 调用入口：`scripts/paper1_mainC_runner.py`
- 模型映射写死在脚本变量 `M`：
  - `V1`: `Qwen3-VL-8B`
  - `V2`: `MiniCPM-V 4.5`
  - `V3`: `Gemini 3 Flash Preview`
  - `V4`: `gpt-5.4`
- 正式结果：`paper1_workspace/02_frozen_results/mainC_final/`
- `manifest.json` 显示 V1/V2/V3/V4 的 full-call cache 来自多个 run。

复现建议：

- 论文复现实验应优先读取 `*_full_call_cache.jsonl`，因为新跑 API 可能因模型版本、服务端行为、key 池、temperature、超时和解析逻辑导致不可重复。
- 若必须新跑，必须记录 provider、model_name、base_url、temperature、max_tokens、prompt_version、API 日期、失败重试策略和完整 raw output。

### 4.4 Router score 如何计算

paper1 final runner 内部直接计算 GCR/WUR/DGCR/DWUR：

位置：

- `scripts/paper1_mainA_runner.py`
- `scripts/paper1_mainC_runner.py`
- `scripts/paper1_online_budget_check.py`

核心公式：

| Router | 公式 |
|---|---|
| GCR | `1.0 - conf` |
| DGCR | `(1.0 - conf) + r_d` |
| WUR | `0.5*(1-mean_conf) + 0.3*(1-min_conf) + 0.2*drop + gate_bonus` |
| DWUR | `WUR + eta*r_d` |

关键常数：

| 参数 | 值 | 位置 |
|---|---:|---|
| `wur_mean_weight` | 0.5 | `configs/router_config.yaml` / runner 常量 |
| `wur_min_weight` | 0.3 | 同上 |
| `wur_drop_weight` | 0.2 | 同上 |
| `wur_min_conf_gate_threshold` | 0.35 | 同上 |
| `wur_drop_gate_threshold` | 0.20 | 同上 |
| `wur_gate_bonus` | 0.10 | 同上 |
| `eta` | 0.5 | `configs/router_config.yaml` 的 `sh_da_v4.rule_scorer.eta` |

最终 Main A 的 router 分数矩阵：

- `paper1_workspace/02_frozen_results/mainA_final/router_score_matrix.csv`
- 行数：3425，包括表头，即 3424 个样本。

### 4.5 budget threshold 如何设置

论文主流程是 offline replay，不是动态在线阈值：

- 对每个 router，按 score 从高到低排序。
- 对预算 `b`，选择 `round(N*b)` 个样本升级。
- `N = 3424`。
- 预算实际命中通过 `actual_call_rate` 和 `call_rate_valid` 记录。

Main A dense budgets：

`0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.80, 1.00`

Main C budgets：

`0.05, 0.10, 0.20, 0.30, 0.50, 0.80`

正式主文 budget points：

`0.10, 0.20, 0.30`

配置来源：

- `configs/router_config.yaml` 中 `mainline.formal_budgets`
- `scripts/paper1_mainA_runner.py` 中 `BUDGETS`
- `scripts/paper1_mainC_runner.py` 中 `B`

### 4.6 不同 budget 下如何 replay / evaluate

Main A replay 输出：

- `paper1_workspace/02_frozen_results/mainA_final/offline_budget_10_GCR.jsonl`
- `paper1_workspace/02_frozen_results/mainA_final/offline_budget_20_GCR.jsonl`
- `paper1_workspace/02_frozen_results/mainA_final/offline_budget_30_GCR.jsonl`
- 同理还有 WUR / DGCR / DWUR 和其他 dense budgets。

每条 replay 样本记录包含：

| 字段 | 含义 |
|---|---|
| `selected_for_upgrade` | 当前预算下是否调用 VLM |
| `replay_rank` | 按 router score 排序后的名次 |
| `final_text` | 若升级则取 VLM correction，否则取 OCR draft |
| `edit_distance_ocr` | OCR draft 到 GT 的编辑距离 |
| `edit_distance_final` | final text 到 GT 的编辑距离 |
| `backfill_status` | paper1 主流程中通常为 `skipped` |
| `backfill_reason` | paper1 主流程中为 `paper1_routeronly` |

Main A 汇总表：

- `tab_mainA_results.csv`：61 行，含 4 routers × 15 budgets + header。
- `tab_mainA_budget_check.csv`：13 行，含 4 routers × 3 formal budgets + header。
- `tab_mainA_domain_results.csv`：181 行，含 router × budget × domain + header。

Main C 汇总表：

- `tab_mainC_results.csv`：25 行，含 4 models × 6 budgets + header。
- `tab_mainC_budget_check.csv`：13 行，含 4 models × 3 formal budgets + header。

### 4.7 RER / CER 或其他指标如何计算

CER：

- 多个脚本使用 `Levenshtein.distance(normalized_prediction, normalized_gt) / len(normalized_gt)`。
- 归一化主要统一全半角括号、中文/英文标点等。
- 相关代码位置：
  - `scripts/paper1_mainA_runner.py`
  - `scripts/paper1_mainC_runner.py`
  - `scripts/paper1_batch1_analysis.py`
  - `scripts/paper1_upper_lower_bounds.py`
  - `scripts/run_efficiency_frontier.py`
  - `scripts/evaluate.py`

RER：

- Figure 2 / Table 2 中通常按 OCR baseline 计算 relative error reduction。
- baseline 在 `paper1_workspace/02_frozen_results/phase2_batch1_final/manifest.json` 中记录为 `base_ocr_cer = 0.088383`。
- Figure 4 的 bounds 中 `tab_upper_lower_bounds.csv` 给出：
  - A-Only: `CER = 0.088351`
  - B-Only Recognition: `CER = 0.082637`
  - A+B Correction: `CER = 0.0514`
- `fig_source_map.md` 明确主文统一使用 OCR baseline `0.088383`，bounds run 中 `0.088351` 的差异需附录说明。

其他指标：

| 指标 | 来源 |
|---|---|
| `BoundaryDeletionRecallAtB` | `summarize_extended_metrics`，识别边界删除样本是否被选中 |
| `SubstitutionCER` | 替换错误数量 / GT 总长度 |
| `AER` | 升级样本中 VLM 输出与 OCR 不同的比例，或 accepted edit ratio |
| `CVR` | 当前 paper1 表中多为 0 或占位 |
| `p95_latency_ms` | 升级样本 latency 的 95 分位 |
| `avg_token_usage` | token usage 平均值；当前 Gemini 结果多为 null，需谨慎解释 |

### 4.8 论文表格和图片结果来源

主文 Figures：

| Figure | 来源文件 | 生成脚本 |
|---|---|---|
| Figure 1 | `实验设计书.md`、`paper1_workspace/03_figures_tables/fig_source_map.md`、协议说明 | `paper1_workspace/03_figures_tables/fig1/nature_fig1/make_fig1_nature.py` |
| Figure 2 | `mainA_final/tab_mainA_results.csv`、`phase2_batch1_final/tab_random_baseline_results.csv`、`tab_minconf_baseline_results.csv` | `paper1_workspace/03_figures_tables/fig2/make_fig2_budget_frontier.py`、`nature_figures/make_nature_figures.py` |
| Figure 3 | `mainC_final/tab_mainC_results.csv`、`tab_mainC_budget_check.csv` | `paper1_workspace/03_figures_tables/fig3/make_fig3_cross_model_behavior.py` |
| Figure 4 | `upper_lower_bounds_final/tab_upper_lower_bounds.csv` | `paper1_workspace/03_figures_tables/fig4/make_fig4_collaboration_bounds.py` |
| Figure 5 | `mainA_final/tab_mainA_domain_results.csv`、`phase2_batch1_final/tab_domain_router_ranking.csv`、`tab_domain_budget_curve.csv` | `paper1_workspace/03_figures_tables/fig5/make_fig5_domain_small_multiples.py` |

主文 Tables：

| Table | 来源文件 | 生成脚本 |
|---|---|---|
| Table 1 | `data/l2w1data/train.jsonl`、`val.jsonl`、`test.jsonl` | `paper1_workspace/03_figures_tables/tables/Table 1/make_table1_dataset_protocol.py` |
| Table 2 | `mainA_final/tab_mainA_results.csv`、`tab_mainA_budget_check.csv`、Random / MinConf baseline | `paper1_workspace/03_figures_tables/tables/Table 2/make_table2_router_official_budget.py` |
| Table 3 | `mainC_final/tab_mainC_results.csv`、`tab_mainC_budget_check.csv` | `paper1_workspace/03_figures_tables/tables/Table 3/make_table3_cross_model_official_budget.py` |
| Table 4 | `upper_lower_bounds_final/tab_upper_lower_bounds.csv` | 当前未见单独 Table 4 脚本；Figure 4 脚本输出 `fig4_collaboration_bounds_summary.csv` |

不确定：

- Table 4 是否最终手工整理进 LaTeX，还是由 `make_fig4_collaboration_bounds.py` 的 summary CSV 间接生成。当前目录中 Table 1/2/3 有独立 make 脚本，Table 4 未发现对应 `tables/Table 4/`。

---

## 五、配置文件和参数分析

### 5.1 配置文件清单

| 文件 | 用途 | 与论文强相关程度 |
|---|---|---|
| `configs/router_config.yaml` | 主配置：数据、Agent B、prompt、formal budgets、router scorer、budget controller | 高 |
| `paper1_workspace/02_frozen_results/mainA_final/config_snapshot.yaml` | Main A 正式 run 快照 | 高 |
| `paper1_workspace/02_frozen_results/official_agent_a_cache/config_snapshot.yaml` | Agent A cache 快照 | 高 |
| `paper1_workspace/02_frozen_results/*/manifest.json` | 冻结结果来源记录 | 高 |
| `PRL_submission_package/03_Supplementary_material/02_Router_and_Budget_Configurations/router_config.yaml` | 投稿补充材料中的 router 配置 | 高 |
| `data/dicts/Geology.txt`、`Finance.txt`、`Medicine.txt` | 领域词典，影响 `r_d` / domain-aware router | 高 |
| `requirements.txt` | 手写精选依赖，锁定 CUDA/Paddle/Torch | 高 |
| `requirements_l2w1v2.txt` | pip freeze 风格完整依赖 | 中高 |
| `requirements_windows.txt` | Windows 最小依赖 | 中 |
| `models/agent_a_ppocr/PP-OCRv5_server_rec_infer/inference.yml` / `inference.json` | Paddle OCR 模型配置 | 高 |
| `scripts/_models.json`、`scripts/_test_model.json` | API/model 测试配置 | 低，疑似调试 |
| `GPT.config`、`GPTkey.txt`、`key.txt` | API key / provider 配置 | 高风险，不应进论文材料 |
| `.gitignore` | 忽略规则 | 中，当前规则与实际工作区状态不完全一致 |

### 5.2 论文强相关参数

| 参数 | 值 | 文件 |
|---|---|---|
| 数据根目录 | `data/l2w1data` | `configs/router_config.yaml` |
| test split | `data/l2w1data/test.jsonl` | 同上 |
| image root | `data/l2w1data/images` | 同上 |
| Agent A model | `./models/agent_a_ppocr/PP-OCRv5_server_rec_infer` | 同上 |
| char dict | `ppocr/utils/ppocrv5_dict.txt` | runner 默认参数 |
| prompt version | `prompt_v1.1` | `configs/router_config.yaml`、快照 |
| formal budgets | `[0.10, 0.20, 0.30]` | `configs/router_config.yaml` |
| Agent B backend | `gemini` | `configs/router_config.yaml` |
| Gemini model | `gemini-3-flash-preview` | `configs/router_config.yaml` |
| Gemini temperature | `0.1` | `configs/router_config.yaml` |
| Gemini max_tokens | `256` | `configs/router_config.yaml` |
| WUR weights | `0.5 / 0.3 / 0.2` | `configs/router_config.yaml` |
| WUR gates | `min_conf < 0.35`、`drop > 0.20`、bonus `0.10` | 同上 |
| domain weight eta | `0.5` | 同上 |
| Main A seed | `42` | `mainA_final/config_snapshot.yaml` |
| Random baseline seeds | `1,2,3,4,5` | `phase2_batch1_final/manifest.json` |

### 5.3 hard-coded path / 绝对路径 / API key 风险

高风险：

| 文件 | 问题 |
|---|---|
| `key.txt` | 存在大量 `sk-...` 形式 API keys；不应进入任何论文复现包或公开仓库 |
| `GPT.config` | 第 4 行附近存在 `apiKey` 字段 |
| `GPTkey.txt` | 从文件名和体量判断为 key 文件；需作为敏感文件处理 |
| `scripts/_test_api.py` | 第 17 行附近硬编码 `Authorization: Bearer sk-...`，这是最高优先级安全问题 |
| `scripts/_debug_agentb.py`、`scripts/_test_api.py` | 含 `sys.path.insert(0, 'G:/Code/PaddleOCR/L2W1')` 和个人机器路径 |
| `paper1_workspace/02_frozen_results/mainA_final/config_snapshot.yaml` | `output_dir`、`shared_agent_a_cache` 含 `/home/coder/project/L2W1v2.0/...` |
| `paper1_workspace/02_frozen_results/phase2_batch1_final/manifest.json` | 含 `G:\Code\PaddleOCR\L2W1\...` 绝对路径 |
| `paper1_workspace/02_frozen_results/upper_lower_bounds_final/manifest.json` | 指向旧相对路径 `paper1_runs/upper_lower_bounds/...`，当前正式来源实际在 `paper1_runs_old/` 注册文件中说明 |

中风险：

- `.gitignore` 忽略 `key.txt`，但根目录仍实际存在 `GPT.config`、`GPTkey.txt` 等敏感文件名，且 `scripts/_test_api.py` 有硬编码 token。
- 多个脚本默认路径仍指向旧目录，例如 `scripts/paper1_batch1_analysis.py` 默认 `paper1_runs/mainA/20260417_run112203`，而正式索引指向 `paper1_runs/mainA_fixed/20260421_run142428`。
- `configs/router_config.yaml` 中 `circuit_breaker.enabled: true`，但 paper1 主图来源映射要求 `No backfill, no breaker in this paper`；需要在复现文档中明确 paper1 runner 并未使用这些模块。

---

## 六、数据、缓存和结果文件分析

### 6.1 数据目录

| 路径 | 内容 | 建议 |
|---|---|---|
| `data/l2w1data/images/` | 15420 张单行图像，文件名如 `GeoP0199_L001.jpg`、`FinP0001_L001.jpg`、`MedP...jpg` | 论文复现必须保留 |
| `data/l2w1data/train.jsonl` | 8522 行 | 保留 |
| `data/l2w1data/val.jsonl` | 3470 行 | 保留 |
| `data/l2w1data/test.jsonl` | 3424 行 | 保留 |
| `data/dicts/Geology.txt` | 地质领域词典 | 保留 |
| `data/dicts/Finance.txt` | 金融领域词典 | 保留 |
| `data/dicts/Medicine.txt` | 医学领域词典 | 保留 |

### 6.2 缓存目录

| 路径 | 类型 | 判断 |
|---|---|---|
| `paper1_workspace/02_frozen_results/official_agent_a_cache/` | 正式 OCR cache | 必须保留 |
| `paper1_workspace/02_frozen_results/mainA_final/shared_repmodel_full_call_cache.jsonl` | Main A Gemini full-call cache | 必须保留 |
| `paper1_workspace/02_frozen_results/mainC_final/V*_full_call_cache.jsonl` | cross-model full-call cache | 必须保留 |
| `paper1_runs/shared_agent_a_cache/official_20260421_run142031/` | OCR cache 原始 run | 建议保留为追溯来源 |
| `results/stage2_v51/agent_a_cache.json` | stage2 旧 Agent A cache | 归档候选，不应作为 paper1 final 来源 |
| `.codex_work/` | 文档生成/LibreOffice profile/cache | 非论文复现材料，可排除 |
| `__pycache__/` | Python 缓存 | 可清理候选，但本次不删除 |

### 6.3 结果目录

| 路径 | 判断 |
|---|---|
| `paper1_workspace/02_frozen_results/` | paper1 final 结果唯一推荐引用目录 |
| `paper1_workspace/03_figures_tables/` | paper1 图表、plotdata、表格输出目录 |
| `PRL_submission_package/` | 投稿包，应作为发布归档保留 |
| `paper1_runs/` | 原始正式 run + 中间 run，保留追溯；不建议论文直接引用 |
| `paper1_runs_old/` | 旧 run；其中 `upper_lower_bounds/20260421_run102819_merged` 是正式 bounds 来源，应单独标注保留 |
| `results/expriments/` | exA/exB 早期结果；目录名 typo，建议归档为 stage/pre-paper experiments |
| `cloud_result_sync/` | 云同步结果，包含 exB_phaseA dense budget scan；保留追溯但不作为 final |
| `logs/` | 运行日志，低优先级保留 |
| `outputs/` | 基本空或旧输出，归档/清理候选 |

### 6.4 应保留为论文复现材料的文件

最小保留集建议：

- `README.md`
- `requirements.txt`、`requirements_l2w1v2.txt`、`requirements_windows.txt`
- `configs/router_config.yaml`
- `data/l2w1data/train.jsonl`
- `data/l2w1data/val.jsonl`
- `data/l2w1data/test.jsonl`
- `data/l2w1data/images/`
- `data/dicts/`
- `models/agent_a_ppocr/PP-OCRv5_server_rec_infer/`
- `modules/`
- `scripts/paper1_mainA_runner.py`
- `scripts/paper1_mainC_runner.py`
- `scripts/paper1_batch1_analysis.py`
- `scripts/paper1_gpt_case_analysis.py`
- `scripts/paper1_upper_lower_bounds.py`
- `scripts/paper1_merge_upper_lower_bounds.py`
- `scripts/paper1_online_budget_check.py`
- `scripts/merge_mainc_runs.py`
- `scripts/convert_old_gemini_to_mainc_v3.py`
- `paper1_workspace/01_registry/`
- `paper1_workspace/02_frozen_results/`
- `paper1_workspace/03_figures_tables/`
- `PRL_submission_package/03_Supplementary_material/`

### 6.5 可能是中间缓存、调试输出、重复结果或废弃结果

| 路径 | 分类建议 |
|---|---|
| `.codex_work/` | 工具临时文件，不纳入论文代码包 |
| `archive/api_test/` | API 调试归档 |
| `scripts/_debug_agentb.py`、`scripts/_test_api.py`、`scripts/_models.json`、`scripts/_test_model.json` | 调试/敏感候选 |
| `scripts/smoke_test_*.py`、`scripts/test_*.py` | smoke / integration test，需单独移入 tests 或 archive |
| `logs/frontier_test.log`、`logs/gemini_ceiling.log` | 运行日志 |
| `cloud_result_sync/exB_phaseA/` | 云端 dense scan 结果副本，追溯而非 final |
| `results/expriments/` | 早期 exA/exB 结果，归档候选 |
| `paper1_runs_old/online_budget_validation/` | online validation 旧 run，补充材料候选 |
| `paper1_workspace/03_figures_tables/fig1/例子/` | 图设计参考截图，非实验数据 |
| `paper1_workspace/04_drafts/*/_rewrite_work/` | 文稿生成脚本和中间 media，写作过程材料 |
| `nul` | 空文件，临时/误生成候选 |
| `*.pyc`、`__pycache__/` | 缓存 |
| `cloud_result_sync/exB/02_runs/*.pid` | 运行 pid 文件，临时 |

---

## 七、代码质量问题清单

### 7.1 高严重程度

| 问题 | 影响范围 | 涉及文件 | 为什么是问题 | 是否影响论文复现 | 后续建议 |
|---|---|---|---|---|---|
| API key 和硬编码 token 暴露 | 安全、合规、公开复现 | `key.txt`、`GPT.config`、`GPTkey.txt`、`scripts/_test_api.py` | 存在真实 key 形式字符串；公开或共享会导致账户风险 | 不影响已冻结结果读取，但影响代码发布 | 立即轮换密钥；从仓库和报告包排除；改用 `.env.example` |
| 正式结果依赖 VLM API cache，重新调用不可稳定复现 | Main A / Main C / bounds | `mainA_final/shared_repmodel_full_call_cache.jsonl`、`mainC_final/V*_full_call_cache.jsonl`、`modules/vlm_expert/gemini_expert.py` | 外部 API 模型可能变化，temperature、解析、超时、key 池均影响输出 | 影响“从头重跑”复现，不影响“读取冻结结果”复现 | 明确冻结 cache 是论文结果基准；补充 API 调用日期和 provider |
| 多套结果并存，final 来源必须靠人工读 registry 才能判断 | 所有论文图表 | `paper1_runs/`、`paper1_runs_old/`、`results/expriments/`、`cloud_result_sync/`、`paper1_workspace/02_frozen_results/` | 容易误引用旧 run 或中间 run | 高，可能导致论文数字不一致 | 维护 `RESULTS_INDEX.md`；脚本默认指向 frozen results |
| 脚本默认路径指向旧 run | batch / case / old scripts | `scripts/paper1_batch1_analysis.py`、`scripts/paper1_gpt_case_analysis.py` | 默认参数不是 `paper1_workspace/02_frozen_results`，重跑会得到旧版本 | 中高 | 将 final 命令写入 docs；新增 frozen wrapper |
| 配置启用原型模块但 paper1 声明 RouterOnly | 论文叙事一致性 | `configs/router_config.yaml`、`modules/router/backfill.py`、`modules/router/circuit_breaker.py`、`fig_source_map.md` | `circuit_breaker.enabled: true`、backfill 配置存在；paper1 结果中又标记 no backfill/no breaker | 若跑错入口会影响结果解释 | 为 paper1 单独建立 `configs/paper1_routeronly.yaml` |
| 绝对路径和个人机器路径混入正式快照 | 跨机器复现 | `mainA_final/config_snapshot.yaml`、`phase2_batch1_final/manifest.json`、`scripts/_debug_agentb.py` | Linux `/home/coder/...` 与 Windows `G:\...` 混用 | 中高 | 引入 path remap 说明；manifest 使用相对路径 |

### 7.2 中严重程度

| 问题 | 影响范围 | 涉及文件 | 为什么是问题 | 是否影响论文复现 | 后续建议 |
|---|---|---|---|---|---|
| CER / normalization / replay 逻辑重复实现 | 指标一致性 | `scripts/paper1_mainA_runner.py`、`paper1_mainC_runner.py`、`paper1_batch1_analysis.py`、`paper1_upper_lower_bounds.py`、`run_efficiency_frontier.py`、`evaluate.py` | 多处手写 `norm` 和 CER，未来改动容易不一致 | 当前冻结结果不受影响，重跑有风险 | 抽出 `src/metrics.py` 并加测试 |
| Router score 公式重复实现 | Router 一致性 | `paper1_mainA_runner.py`、`paper1_mainC_runner.py`、`paper1_online_budget_check.py`、`modules/router/uncertainty_router.py` | GCR/WUR/DGCR/DWUR 常数分散 | 影响重跑和第二篇扩展 | 抽出 `src/routing/scores.py` |
| 大型单文件职责过重 | 可维护性 | `modules/paddle_engine/predict_rec_modified.py`、`modules/router/uncertainty_router.py`、`scripts/evaluate.py`、`scripts/run_efficiency_frontier.py` | 文件 700-1450 行，包含多种职责 | 不直接影响冻结结果 | 阶段 3 模块化拆分 |
| 多套 requirements 不完全一致 | 环境复现 | `requirements.txt`、`requirements_l2w1v2.txt`、`requirements_windows.txt` | `Levenshtein`、`PyYAML`、`scikit-learn` 等版本存在差异 | 中 | 明确 Linux/CUDA、Windows/CPU、analysis-only 三个环境 |
| `.gitignore` 与项目实际内容不一致 | 版本管理 | `.gitignore`、根目录 key 文件、docs/results | `.gitignore` 忽略 docs/results/md，但工作区实际包含大量此类文件 | 中 | 制定论文归档清单，避免误提交/误排除 |
| Main C final 是多次 run 合并结果 | 结果追溯 | `mainC_final/manifest.json`、`scripts/merge_mainc_runs.py` | V1/V2/V3/V4 来源不同，需要说明合并规则 | 中 | 在 Supplementary 中记录合并 provenance |
| VLM 输出解析偏保守但没有系统测试 | 纠错输出质量 | `modules/vlm_expert/gemini_expert.py`、`agent_b_expert.py`、`constrained_prompter.py` | `_parse_output` 只取第一行、删首尾符号，可能影响符号类文本 | 可能影响重跑 | 建立 parse fixture tests |
| online budget validation 与 offline replay 同时存在 | 概念清晰度 | `paper1_online_budget_check.py`、`fig_source_map.md` | 容易让读者误解主结果来自 online controller | 中 | 文档明确主结果为 offline top-k replay |
| 随机种子记录不完整 | 随机复现 | `paper1_batch1_analysis.py`、Main A/C config | Random baseline seeds 有记录；API 并发顺序、Paddle/Python/CUDA 确定性未完整记录 | 中 | 加环境快照、CUDA/Paddle/Torch deterministic 说明 |

### 7.3 低严重程度

| 问题 | 影响范围 | 涉及文件 | 为什么是问题 | 是否影响论文复现 | 后续建议 |
|---|---|---|---|---|---|
| 目录名 typo | 可读性 | `results/expriments/` | 应为 `experiments` | 低 | 阶段 1 文档说明，不立即改名 |
| 中英文命名混杂 | 可读性 | 根目录中文 `.md`、英文脚本、中文子目录 | 对外发布不够统一 | 低 | docs 中保留中文，代码目录英文 |
| 图表输出多版本并存 | 图表选择 | `fig2_*`、`fig5_*_revised`、`nature_figures/` | 不清楚最终采用哪个版本时会误用 | 中低 | `fig_source_map.md` 已缓解；建议加 FINAL 标记 |
| 空文件 / pid / pyc / log | 清洁度 | `nul`、`*.pid`、`__pycache__/`、`logs/` | 临时文件干扰审计 | 低 | 阶段 1 归档/清理候选，不立即删除 |
| README 仍停留在早期结构 | 新用户上手 | `README.md` | 未说明 paper1 final 复现路径 | 中低 | 补充 “paper1 reproducibility quickstart” |

---

## 八、废弃代码和可疑文件候选清单

> 以下只是候选，不建议本报告阶段删除。

| 候选 | 怀疑原因 | 风险 | 后续处理建议 |
|---|---|---|---|
| `archive/` | 已命名为 archive，内容为早期 module spec 和 API test | 低 | 保留为历史归档；从主复现 README 中隐藏 |
| `archive/api_test/` | 与 `scripts/_debug_agentb.py`、`scripts/_test_api.py` 内容重复 | 中，含 API 测试模式 | 归档；检查是否含 key 后再公开 |
| `scripts/_test_api.py` | 硬编码 Bearer token 和个人路径 | 高 | 不公开；替换为安全 mock 示例 |
| `scripts/_debug_agentb.py` | debug 脚本，硬编码 `G:/Code/...` | 中 | 移入 `scripts/dev/` 或 `archive/debug/` |
| `scripts/_models.json`、`scripts/_test_model.json` | API/model 测试配置 | 中 | 归档或改为 `.sample` |
| `scripts/smoke_test_agent_b.py`、`scripts/smoke_test_multi_vlm.py` | smoke test，不是 final pipeline | 低 | 移入 `tests/smoke/` |
| `scripts/test_stage2_modules.py`、`scripts/test_stage2_integration.py`、`scripts/test_efficiency_100.py`、`scripts/test_provider_pools.py`、`scripts/test_single_api.py` | 测试/手动验证脚本混在 scripts | 中 | 迁移到 `tests/`，保留有用 fixture |
| `scripts/run_l2w1_pipeline.py` | 早期 pipeline，与 paper1 final runner 不同 | 中 | 标记 legacy，避免作为主入口 |
| `scripts/run_efficiency_frontier.py` | v5.1 grand loop，final paper runner 复用其工具函数但主入口不是它 | 中 | 拆出工具函数，runner 标为 legacy/prototype |
| `scripts/run_all_frontiers.py`、`run_main_exp_b.py`、`run_main_exp_c.py`、`run_phase_a_m5_budget_scan.py`、`run_phase_b_five_pool_analysis.py` | 与 exA/exB/phaseA 旧实验相关 | 中 | 归档到 `scripts/legacy_experiments/` |
| `results/expriments/` | 早期 exA/exB，目录名 typo，非 paper1 final | 中 | 保留为 historical results，迁移索引 |
| `cloud_result_sync/exB_phaseA/` | 云端 phaseA dense scan，体量大，非 final | 中 | 单独归档，写清与 paper1 final 的关系 |
| `paper1_runs_old/` | 多个旧 run，仅部分是 final bounds 来源 | 中 | 保留 `upper_lower_bounds/20260421_run102819_merged`，其余归档 |
| `paper1_runs/mainC_fixed/20260421_run150112`、`20260422_run052223`、`20260422_run062138` | Main C 中间来源，被 final manifest 引用 | 中 | 不删除；作为 provenance 保留 |
| `paper1_workspace/03_figures_tables/fig1/例子/` | PixPin 截图参考，不是实验数据 | 低 | 移入 `assets/design_references/` |
| `paper1_workspace/04_drafts/*/_rewrite_work/` | 文稿生成和修订中间脚本 | 低 | 写作归档，不纳入代码复现 |
| `logs/` | 运行日志 | 低 | 可压缩归档 |
| `nul` | 0 字节空文件 | 低 | 阶段 1 清单中标记为临时文件 |
| `__pycache__/`、`*.pyc` | Python 缓存 | 低 | 阶段 1 清理候选 |
| `.codex_work/` | Codex/LibreOffice 文档渲染缓存 | 低 | 不纳入论文项目结构 |

---

## 九、建议整理后的项目结构

建议目标结构：

```text
L2W1/
├── README.md
├── pyproject.toml                  # 或 environment.yml / requirements lock
├── configs/
│   ├── paper1_routeronly.yaml
│   ├── paper1_cross_model.yaml
│   └── env.example
├── data/
│   ├── l2w1data/
│   │   ├── train.jsonl
│   │   ├── val.jsonl
│   │   ├── test.jsonl
│   │   └── images/
│   └── dicts/
├── models/
│   └── agent_a_ppocr/
├── src/
│   ├── ocr/
│   ├── routing/
│   ├── vlm/
│   ├── replay/
│   ├── metrics/
│   └── io/
├── scripts/
│   ├── reproduce_paper1_mainA.py
│   ├── reproduce_paper1_mainC.py
│   ├── make_paper1_baselines.py
│   ├── make_paper1_bounds.py
│   └── make_paper1_figures_tables.py
├── cache/
│   ├── agent_a/
│   └── vlm_full_call/
├── results/
│   └── paper1_frozen/
├── figures/
│   ├── paper1/
│   └── appendix/
├── tables/
│   ├── paper1/
│   └── appendix/
├── docs/
│   ├── reproducibility.md
│   ├── result_index.md
│   ├── data_card.md
│   └── api_safety.md
├── supplementary/
│   ├── prompts/
│   ├── configs/
│   ├── dataset_splits/
│   └── frozen_results/
├── notebooks/
│   └── exploratory/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
└── archive/
    ├── legacy_stage2/
    ├── old_runs_index/
    └── manuscript_drafts/
```

目录职责：

| 目录 | 应放内容 |
|---|---|
| `src/ocr/` | `TextRecognizerWithLogits`、Agent A schema、置信度和边界统计 |
| `src/routing/` | GCR/WUR/DGCR/DWUR、domain risk、budget top-k/online controller |
| `src/vlm/` | Gemini/local VLM adapters、provider abstraction、prompt parser |
| `src/replay/` | full-call cache replay、budget sweep、case pool 生成 |
| `src/metrics/` | CER、RER、AER、BoundaryDeletionRecall、SubstitutionCER，统一实现 |
| `src/io/` | JSONL/CSV/manifest/schema helpers |
| `scripts/` | 薄入口，只解析参数并调用 `src/` |
| `configs/` | paper1 固定配置、第二篇论文配置、样例 env |
| `cache/` | 可复用但不一定公开的 agent/VLM cache |
| `results/paper1_frozen/` | 论文 frozen results，替代当前 `paper1_workspace/02_frozen_results` |
| `figures/`、`tables/` | 最终图表输出，不混入设计草图 |
| `docs/` | 复现说明、结果索引、数据说明、投稿材料说明 |
| `supplementary/` | 与论文补充材料一一对应的 prompt/config/split/result |
| `tests/` | 指标、router score、replay、parser 的可重复测试 |
| `archive/` | 老版本、临时脚本、草稿、旧 run 索引 |

---

## 十、分阶段整理路线

### 阶段 1：只归档和补文档，不改核心逻辑

目标：在不改变任何实验结果、不重构代码的前提下，让“论文最终结果在哪里、如何引用、哪些不要碰”变清楚。

具体任务：

| 任务 | 风险 | 验收标准 |
|---|---|---|
| 新增 `docs/reproducibility_paper1.md`，写清 final frozen results 入口 | 低 | 新用户能从文档定位 `paper1_workspace/02_frozen_results/` |
| 新增 `docs/result_provenance_index.md`，把 `paper1_workspace/01_registry/正式结果索引.md` 英文化/结构化 | 低 | 每个 Figure/Table 都能追溯到 CSV/JSONL |
| 新增敏感文件说明：`key.txt`、`GPT.config`、`GPTkey.txt` 不进入任何公开包 | 中 | 公开材料清单不含 key 文件 |
| 给 `paper1_runs/`、`paper1_runs_old/`、`results/expriments/`、`cloud_result_sync/` 写只读索引 | 低 | old/intermediate/final 分类明确 |
| 标注 debug/test/smoke 脚本，不移动、不删除 | 低 | `scripts/README.md` 能说明每类脚本用途 |
| 补 README 的 paper1 quickstart | 低 | README 明确“不确定时先读 frozen results，不要直接跑 API” |

阶段 1 不做：

- 不改 runner 逻辑。
- 不改结果文件。
- 不清理旧目录。
- 不重命名 `expriments` 等历史目录。

### 阶段 2：整理入口脚本和配置，不改变实验结果

目标：让复现入口清晰、配置可移植，同时用测试保证新入口读取冻结结果得到相同表格。

具体任务：

| 任务 | 风险 | 验收标准 |
|---|---|---|
| 新增 `configs/paper1_routeronly.yaml`，只包含 paper1 实际使用参数 | 中 | 不启用 backfill/breaker；formal budgets 明确 |
| 新增 `scripts/reproduce_paper1_from_frozen.py`，只读 frozen results 重新生成主表/主图 plotdata | 中 | 输出与现有 `03_figures_tables` plotdata 一致 |
| 把 `paper1_batch1_analysis.py`、`paper1_gpt_case_analysis.py` 默认路径改为参数化 wrapper；原脚本保持兼容 | 中 | 不传参时不误读旧 run，或明确报错要求路径 |
| 统一 metrics helper，但先只新增不替换旧逻辑 | 中 | 单元测试证明 CER/RER 与冻结表一致 |
| 新增 schema 检查脚本：检查 3424 样本、cache 字段、budget row count | 中 | CI/本地一键检查通过 |
| 用 `.env.example` 替代 key 文件示例 | 中 | 没有真实 key 的公开配置仍能说明如何配置 |

风险控制：

- 每一步只新增 wrapper 和测试，暂不删除旧脚本。
- 对比 `tab_mainA_results.csv`、`tab_mainC_results.csv`、`tab_upper_lower_bounds.csv` 的 hash 或关键数值。

### 阶段 3：模块化重构，为第二篇论文和实习项目展示做准备

目标：把当前“论文一次性脚本集合”整理为可扩展的研究代码库。

具体任务：

| 任务 | 风险 | 验收标准 |
|---|---|---|
| 抽出 `src/metrics.py`，统一 CER/RER/AER/BDR/SubstitutionCER | 中高 | 所有旧结果可用统一指标复算 |
| 抽出 `src/routing/scores.py`，统一 GCR/WUR/DGCR/DWUR | 中高 | router score matrix 与 frozen 结果一致 |
| 抽出 `src/replay/offline.py`，统一 top-k budget replay | 中高 | Main A / Main C replay 可共享 |
| 抽出 `src/vlm/cache.py` 和 `src/vlm/providers.py` | 中 | full-call cache 读写 schema 稳定 |
| 将 `modules/paddle_engine/predict_rec_modified.py` 拆分为 OCR runner、softmax/logits、feature extraction | 高 | Agent A cache 对 3424 test 样本输出一致 |
| 整理第二篇论文实验配置：新数据、新 router、新 VLM 后端单独配置 | 中 | paper1 frozen 与 paper2 dev 环境互不污染 |
| 建立最小测试集 fixtures | 中 | 不依赖 API、不依赖大模型即可测试 replay 和 metrics |

阶段 3 风险：

- OCR 推理和 PaddleOCR 修改代码复杂，重构时最容易引入数值差异。
- VLM API 调用不可稳定，测试应默认使用 frozen cache 或 mock。
- 旧论文结果必须 immutable，不应被重构脚本覆盖。

阶段 3 验收标准：

- `paper1_workspace/02_frozen_results/` 不被修改。
- 新模块能从 frozen cache 重新生成核心 CSV，关键 CER 与旧表完全一致。
- 所有真实 API 调用都必须显式传 `--enable_api_calls` 或类似开关。
- 第二篇论文开发使用新 `results/paper2_dev/`，不污染 paper1 final。

---

## 总结结论

1. 当前项目的论文核心已经比较清楚：`paper1_workspace/02_frozen_results/` 是 paper1 的正式冻结结果，`paper1_workspace/03_figures_tables/` 是最终图表来源。
2. 当前项目不是纯净的可复现实验仓库，而是一个在开发过程中逐步沉淀出“论文冻结层”的工作仓库。
3. 最重要的复现链路是：`data/l2w1data` → official Agent A cache → Main A/Main C full-call cache → offline replay → frozen summary CSV → figures/tables。
4. 最大风险不是算法代码本身，而是敏感 key 暴露、多版本结果混杂、脚本默认路径指向旧 run、绝对路径和 paper1 RouterOnly 叙事与 SH-DA++ 原型模块混在一起。
5. 最推荐的整理策略是先做文档和索引，不改核心逻辑；再做只读 wrapper 和 schema/metric 测试；最后才进入模块化重构。
