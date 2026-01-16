# SH-DA++ v4.0 Stage 0/1 服务器执行命令清单

## 📋 前置条件检查

```bash
# 进入项目目录
cd L2W1

# 确认模型文件存在
ls -lh models/agent_a_ppocr/PP-OCRv5_server_rec_infer/
# 应该看到: inference.json, inference.pdiparams, inference.yml

# 确认配置文件存在
ls -lh configs/router_config.yaml

# 确认数据文件存在
ls -lh data/raw/HWDB_Benchmark/train_metadata.jsonl
ls -lh data/raw/HWDB_Benchmark/test_metadata.jsonl
```

---

## 🚀 执行步骤

### **第一步：环境重置与清洁**

```bash
# 删除所有旧结果，防止随机噪声污染
rm -rf results/*
mkdir -p results

echo "[✓] 环境已重置"
```

---

### **第二步：参数重新校准**

```bash
python scripts/calibrate_router.py \
    --config configs/router_config.yaml \
    --target_b 0.2 \
    --limit 1000 \
    --model_dir ./models/agent_a_ppocr/PP-OCRv5_server_rec_infer/
```

**观察重点**：
- ✅ 控制台应输出真实汉字识别文本（如"胡锦涛强调..."）
- ✅ `q` 分数直方图应显示 `min` 和 `max` 有明显差值（例如 0.2 到 0.9 之间分布）
- ✅ `[compute_softmax]` 信息仅在每 1000 个样本打印一次

**输出文件**：
- `results/calibration_stats.json` - 校准统计结果
- `results/ppocrv5_text.jsonl` - 所有识别文本（新增）

**验证命令**：
```bash
# 查看校准结果
cat results/calibration_stats.json | jq '.calibrated_params'

# 查看前 10 条识别文本
head -10 results/ppocrv5_text.jsonl | jq -r '.text'
```

---

### **第三步：正式数据采集**

```bash
python scripts/run_stage1_collection.py \
    --config configs/router_config.yaml \
    --model_dir ./models/agent_a_ppocr/PP-OCRv5_server_rec_infer/ \
    --output_dir ./results
```

**观察重点**：
- ✅ 每 500 个样本（前 5 个 + 每 500 个）输出 `ID | Text | q Score | conf | s_b | s_a`
- ✅ `avg_lat_router_ms` 应维持在 10ms 以内
- ✅ 进度更新频率为每 1000 个样本

**输出文件**：
- `results/router_features.jsonl` - 路由特征数据（包含 `agent_a_text`）

**验证命令**：
```bash
# 查看前 10 行数据
head -10 results/router_features.jsonl | jq '.'

# 检查是否包含识别文本
head -1 results/router_features.jsonl | jq '.agent_a_text'
```

---

### **第四步：稳定性压测**

```bash
python scripts/test_budget_stability.py \
    --config configs/router_config.yaml \
    --target_b 0.2 \
    --model_dir ./models/agent_a_ppocr/PP-OCRv5_server_rec_infer/
```

**观察重点**：
- ✅ `|B̄_total - B| ≤ 0.5%`（平均调用率误差）
- ✅ 最大震荡应降至 `±3%` 以内
- ✅ 生成的 PNG 图片应正确显示中文，分辨率 300 DPI

**输出文件**：
- `results/call_rate_over_time.png` - 调用率时间序列图
- `results/stability_report.json` - 稳定性报告

**验证命令**：
```bash
# 检查图片是否存在
ls -lh results/call_rate_over_time.png

# 查看稳定性报告
cat results/stability_report.json | jq '.'
```

---

### **第五步：最终审计**

```bash
python scripts/evaluate.py \
    --predictions results/router_features.jsonl
```

**观察重点**：
- ✅ `Boundary Deletion Recall@B` 必须 > 0.0
- ✅ `Overall CER Improvement` 应显示 System CER 相比 Agent A 有实质下降

**输出文件**：
- `metrics_summary.json` - 评估指标汇总

**验证命令**：
```bash
# 查看核心指标
cat metrics_summary.json | jq '.metrics.boundary_deletion_recall_at_b'
cat metrics_summary.json | jq '.metrics.overall_cer_improvement'
```

---

## 🔴 结项判定红线

如遇到以下情况，**立即停止并反馈**：

1. **红线 1**：`q` 分数的标准差 (`std`) 为 0 或接近 0
   ```bash
   # 检查命令
   cat results/calibration_stats.json | jq '.q.std'
   ```

2. **红线 2**：`Actual Call Rate` 无法收敛，或 `λ` 很快冲到上限 2.0
   ```bash
   # 检查命令
   cat results/stability_report.json | jq '.lambda_stats'
   ```

3. **红线 3**：生成的 PNG 图片无法显示中文或分辨率极低
   ```bash
   # 检查图片信息
   file results/call_rate_over_time.png
   ```

---

## 📤 提交数据清单

完成后，请提交以下文件：

1. ✅ `results/calibration_stats.json`
2. ✅ `results/ppocrv5_text.jsonl`（新增）
3. ✅ `results/router_features.jsonl`（前 10 行）
4. ✅ `results/call_rate_over_time.png`
5. ✅ `metrics_summary.json`

---

## 🔧 故障排查

### 问题 1：找不到模型文件
```bash
# 检查模型路径
ls -lh models/agent_a_ppocr/PP-OCRv5_server_rec_infer/

# 如果缺失，需要下载模型（根据实际情况调整）
```

### 问题 2：识别文本未显示
- 检查控制台输出，确认 `[compute_softmax]` 显示"检测到输入已是概率分布"
- 查看 `results/ppocrv5_text.jsonl` 是否包含 `text` 字段

### 问题 3：q 分数分布异常
- 查看 `results/calibration_stats.json` 中的 `q.std`
- 检查 `results/ppocrv5_text.jsonl` 中 `q` 值的分布范围

---

## 📝 执行日志示例

```bash
# 记录执行时间
echo "=== Stage 0/1 执行开始: $(date) ==="

# 执行各步骤...

# 记录结束时间
echo "=== Stage 0/1 执行完成: $(date) ==="
```

---

**PI，手册已就绪。请按顺序执行上述命令，完成 Stage 0/1 交付。**