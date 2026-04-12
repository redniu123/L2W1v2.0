# L2W1 主线执行版 TODO

**来源**：`docs/代码落地差距清单.md`  
**目标**：把差距清单收敛为可直接执行的工程任务列表，优先保证主线论文实验可落地。  
**原则**：先收主线，再扩实验面。

---

## P0（本周必须完成）

### P0-1 统一主线命名
**目标**：让代码、结果文件、论文叙事统一到同一套名字。

- [ ] 把当前实验策略名收敛为：
  - `GCR`
  - `WUR`
  - `DGCR`
  - `DWUR`
  - `BAUR`
  - `DAR`
  - `Router-only`
  - `SH-DA++`
- [ ] 明确每个名字在代码中的逻辑定义
- [ ] 正式主表默认不再使用旧的 `BAUR-only` 作为系统名，统一切到 `Router-only`
- [ ] 在实验脚本输出 CSV / JSONL 中同步新名字

**验收标准**：
- 正式脚本跑出来的结果表中，主线路由名与设计书完全一致；
- 所有策略名都能直接对应设计书中的定义。

---

### P0-2 落地 WUR / DGCR / DWUR
**目标**：把设计书新增的固定规则路由正式接入代码并准备实验。

- [ ] `WUR`：实现固定规则分数
  - `0.5*(1-mean_conf) + 0.3*(1-min_conf) + 0.2*drop`
  - 默认启用 gated 版本
- [ ] `DGCR`：在 `GCR` 基础上加入 `eta * r_d`
- [ ] `DWUR`：在 `WUR` 基础上加入 `eta * r_d`
- [ ] 保留 `BAUR / DAR` 作为补充组，不删除
- [ ] 保证输出结果中可以明确区分上述 Router

**验收标准**：
- 可以独立运行 `GCR / WUR / DGCR / DWUR / BAUR / DAR`；
- 同一批样本上，这些 Router 输出可直接对比。

---

### P0-3 补 per-sample 结果保存协议
**目标**：让正式实验进入“可审查、可复现、可追溯”状态。

- [x] 每个预算点保存单独 JSONL
- [x] 每条 JSONL 至少保存：
  - `sample_id`
  - `image_path`
  - `gt`
  - `ocr_text`
  - `router_name`
  - `router_score`
  - `budget`
  - `selected_for_upgrade`
  - `vlm_model`
  - `prompt_version`
  - `vlm_raw_output`
  - `final_text`
  - `backfill_status`
  - `backfill_reason`
  - `run_id`
- [x] 每轮实验同时保存：
  - `metrics_summary.json`
  - `summary.csv`
  - `config_snapshot.yaml`
- [x] 第一轮补充分析归档：
  - `failure_cases.csv`
  - `domain_breakdown.csv`
  - `backfill_log.jsonl`

**验收标准**：
- 任意主表结果都能回溯到逐样本 JSONL；
- 没有 JSONL 的实验不再进入正式分析。

---

## P1（正式主实验前完成）

### P1-1 新增并冻结主线路由基线组
**目标**：形成设计书规定的主线路由集合。

- [ ] `GCR`：分数仅用全局置信度构造
- [ ] `WUR`：固定规则分数（mean / min / drop）
- [ ] `DGCR`：`GCR + eta * r_d`
- [ ] `DWUR`：`WUR + eta * r_d`
- [ ] `BAUR / DAR` 仅作为补充分析组保留

**验收标准**：
- 主线脚本默认可运行 `GCR / WUR / DGCR / DWUR`；
- `BAUR / DAR` 仍可运行，但不再混淆为唯一主线。

---

### P1-2 明确 Router-only 与 SH-DA++ 的系统对照
**目标**：形成系统层主实验对照。

- [ ] `Router-only`：只保留最优主线路由主链
- [ ] `SH-DA++`：最优 Router + 领域软提示 + 严格回填 + 熔断
- [ ] 写清楚两者唯一变量差异

**验收标准**：
- `Router-only` 和 `SH-DA++` 能在统一脚本中并列输出；
- 论文中可以直接引用这两个实验编号。

---

### P1-3 在线预算实验脚本正式化
**目标**：区分“在线预算实验”和“离线预算回放”。

- [x] 新建或改造脚本，使样本逐条经过 `OnlineBudgetController.step(q)`
- [x] 每条样本记录：
  - `lambda_current`
  - `selected_for_upgrade`
  - `actual_budget_window`
- [ ] 验证 `Actual Call Rate = B ± 0.5%`（执行准备已完成；小样本时需配合缩小 `window_size` 或使用全量样本验证）

**验收标准**：
- 在线实验结果真实满足预算协议；
- 不再把 top-K 静态截断结果误写成 online control 结果。

---

### P1-4 离线预算回放正式化
**目标**：把现有 100% full-call 与 top-K 筛选整理成正式 replay 协议。

- [ ] 保存 `full_budget_results.jsonl`
- [ ] 生成：
  - `offline_budget_05_results.jsonl`
  - `offline_budget_10_results.jsonl`
  - `offline_budget_20_results.jsonl`
  - `offline_budget_30_results.jsonl`
  - `offline_budget_50_results.jsonl`
  - `offline_budget_100_results.jsonl`
- [ ] 每个预算点都要有对应 summary 文件

**验收标准**：
- 一次 full-call 结果可以稳定复用；
- 曲线绘制和案例分析都基于 replay 文件完成。

---

### P1-5 Prompt 版本化
**目标**：让 Prompt 成为正式可追溯组件。

- [ ] 配置文件增加 `prompt_version`
- [ ] 每轮实验保存实际 Prompt 文本快照
- [ ] 每条 per-sample JSONL 写入 `prompt_version`

**验收标准**：
- 论文主表任意结果都能追溯到对应 Prompt 版本。

---

### P1-6 熔断器显式化
**目标**：让 circuit breaker 从“隐性容错”升级为“正式模块”。

- [ ] 新增独立 `CircuitBreaker` 模块
- [ ] 定义触发条件：
  - CVR 过高
  - timeout rate 过高
  - API error rate 过高
- [ ] 定义状态机：
  - `closed`
  - `open`
  - `half-open`
- [ ] 输出熔断日志

**验收标准**：
- 熔断触发是可记录、可统计、可解释的。

---

## P2（主线稳定后扩展）

### P2-1 多领域主线落地
**目标**：从 geology 单域原型扩展到设计书规定的三领域主线。

- [ ] 支持 `geology / finance / medicine` 三个 domain
- [ ] 为 finance / medicine 增加领域词典
- [ ] Prompt 根据 domain 切换
- [ ] 输出 `domain_breakdown.csv`

**验收标准**：
- 所有正式实验都能输出 per-domain 结果；
- 论文不再只有 geology 单域证据。

---

### P2-2 主文模型矩阵收敛
**目标**：把探索型模型池和主文模型池分开。

- [ ] 主文固定 2 local + 2 cloud：
  - Qwen3-VL-8B
  - InternVL3-8B
  - Gemini 3 Flash Preview
  - Claude Sonnet 4.6
- [ ] 其余模型转入 exploratory / appendix

**验收标准**：
- 主文表格不再混入探索型模型。

---

### P2-3 高级指标补齐
**目标**：补论文级辅助指标。

- [x] `Boundary Deletion Recall@B`
- [x] `Substitution CER`
- [ ] `p95 latency`（第一轮 summary 输出已完成，统计口径仍可继续严格化）
- [ ] `token usage`（第一轮 summary 输出已完成，真实覆盖率仍可继续提升）
- [x] `error_type` 错误桶（第一轮落盘）

**验收标准**：
- 主结果不只剩 CER/AER/CVR；
- 可支持更细粒度分析与审稿回复。

---

## 推荐执行顺序（实际开发顺序）

### 第 1 步
先改命名：
- `GCR / WUR / DGCR / DWUR / BAUR / DAR / Router-only / SH-DA++`

### 第 2 步
落地 `WUR / DGCR / DWUR`。

### 第 3 步
补 per-sample JSONL 与 run 级归档目录。

### 第 4 步
补正式 online budget 脚本。

### 第 5 步
把 offline replay 正式化。

### 第 6 步
再做 Prompt version / Circuit breaker / 多领域。

---

## 一句话执行策略

> **先收主线，后扩规模；先保证可追溯，后追求更大实验面。**

如果本周只能做三件事，就做：
1. 改命名；
2. 拆 BAUR / DAR；
3. 补 per-sample JSONL。
