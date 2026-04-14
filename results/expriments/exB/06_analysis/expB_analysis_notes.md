# 主实验 B 分析记录

## 当前状态
- 主实验 B 已完成协议层与材料骨架初始化。
- 当前采用 `GCR` 作为 BestRouter（统一预算扩展版）。
- 正式输出要求以 `docs/expB_output_spec.md` 为准。

## 关键提醒
- 现有脚本中的 `Router-only / SH-DA++` 默认仍偏向固定规则路由，不可直接等同于“DGCR 固定版 M5/M6”。
- 若直接运行当前脚本，会出现 BestRouter 定义与输出规范不完全一致的问题。
- 因此主实验 B 的下一步应是：补一个专用执行脚本，或对现有脚本做最小定向改造。

## 推荐实现方向
1. 复用 `run_online_budget_control.py` 的在线预算控制框架。
2. 新增主实验 B 专用 runner：
   - M5: DGCR + no soft prompt + no backfill + no breaker
   - M6: DGCR + soft prompt + backfill + breaker
3. 运行后直接输出 expB_output_spec 要求的目录与文件。
