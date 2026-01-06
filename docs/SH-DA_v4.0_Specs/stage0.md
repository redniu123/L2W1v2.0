A. Emission 矩阵拦截与归一化
在 TextRecognizerWithLogits.**call** 中，拦截 preds（即原始 logits）。
逻辑：应用 Softmax 算子得到 $E \in [0,1]^{T \times C}$ 。
Blank ID：从配置读取，默认通常为 0 。

B. 在线边界统计量计算（核心优化）
不落盘全量 $E$，直接在内存中根据 $\rho$ 计算边界指标 。
参数：$\rho = 0.1$（默认值） 。
计算公式：
$L = [1, \lfloor \rho T \rfloor], 
R = [\lceil (1-\rho)T \rceil, T]$ 。
blank_mean_L = mean(E[L, blank_id]) 7777。+2blank_peak_L = max(E[L, blank_id]) 。

C. Top-2 信号对齐
导出每位置的 Top-2 概率及其字符索引。若模型不支持，需标记 top2_status = 'missing'。

D. 日志底座与数据持久化
文件：router_features.jsonl。
最小字段集：id, img_path, blank_id, rho, T, N, char_conf[], blank_mean_L/R, blank_peak_L/R, top2_status, lat_router_ms 。
