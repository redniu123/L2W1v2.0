# Best Router Record for Main Experiment B

## 1. Router 名称
- best_router_name: GCR

## 2. 来源
- source_table: `results/expriments/exA/04_tables/tab_expA_main_results.csv`
- source_run_id: `20260413_run115631`
- source_date: 2026-04-13

## 3. 选择标准
- 正式在线预算点下的主结果优先
- 在当前正式结果中，`GCR@0.30` 取得全局最低 CER（0.078858）
- 未来计划扩展到 `0.50 / 0.80` 预算时，GCR 作为统一固定 Router 更稳、更便于跨预算叙事
- 虽然 `DGCR` 在 `0.10 / 0.20` 更优，但 GCR 更适合作为后续系统级实验的统一 BestRouter

## 4. 结论
- GCR 固定为主实验 B 的 BestRouter（统一预算扩展版）

## 5. 冻结信息
- freeze_date: 2026-04-14
- router_version: router_v5.1
- remarks: 后续预算扩展（如 0.50 / 0.80）统一继续使用 GCR
