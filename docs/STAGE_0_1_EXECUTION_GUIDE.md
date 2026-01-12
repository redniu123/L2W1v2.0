# SH-DA++ v4.0 Stage 0/1 交付执行指南

> **严格模式**：所有脚本已移除模拟模式，必须使用真实模型。任何错误都会直接报错退出。

---

## 📋 前置条件检查

在执行前，请确保以下条件满足：

### 1. 模型文件

```bash
# 检查 PP-OCRv5 模型是否存在
ls -la ./models/agent_a_ppocr/PP-OCRv5_server_rec_infer/

# 应该看到（PP-OCRv5 新格式）：
# - inference.pdiparams (必须)
# - inference.json (PP-OCRv5 使用 .json 而非 .pdmodel)
# - inference.yml (可选)
# - ppocr_keys.txt (可选)

# 或者旧格式：
# - inference.pdiparams
# - inference.pdmodel

# 如果不存在，请下载：
mkdir -p ./models
cd ./models
wget https://paddle-model-ecology.bj.bcebos.com/model/ocr/PP-OCRv5/ch_PP-OCRv5_rec_infer.tar
tar -xf ch_PP-OCRv5_rec_infer.tar
cd ..
```

### 2. 数据集结构

```bash
# 检查 HWDB 数据集
ls -la ./data/raw/HWDB_Benchmark/

# 应该包含：
# - train_metadata.jsonl  (用于校准)
# - test_metadata.jsonl   (用于正式采集)
# - test/ 或 images/      (图像文件目录)
```

### 3. 配置文件

```bash
# 检查配置文件是否存在
ls -la ./configs/router_config.yaml
```

---

## 🚀 执行流程（5 步）

### Step 0: 环境清理

```bash
cd L2W1

# 清除旧结果（重要！避免污染数据）
rm -rf results/*.jsonl results/*.json results/*.png results/debug/

# 创建输出目录
mkdir -p results/debug/edge_probe
mkdir -p results/debug/blank_curves
```

---

### Step 1: 参数校准（Train Set）

**目标**：在验证集上计算 `v_min`, `v_max`, `λ_0` 并更新配置文件

```bash
python scripts/calibrate_router.py \
    --metadata ./data/raw/HWDB_Benchmark/train_metadata.jsonl \
    --config ./configs/router_config.yaml \
    --model_dir ./models/ppocrv5_rec \
    --target_b 0.2 \
    --limit 2000 \
    --verbose
```

**参数说明**：

- `--metadata`: 训练集 metadata（用于校准）
- `--config`: 配置文件路径（会被自动更新）
- `--model_dir`: Agent A 模型目录（必须提供）
- `--target_b`: 目标调用率 B = 20%
- `--limit`: 限制样本数（可选，用于快速测试）
- `--verbose`: 打印详细日志

**预期输出**：

- `configs/router_config.yaml` 自动更新（`v_min`, `v_max`, `lambda_init`）
- `results/calibration_stats.json` 统计结果
- 终端打印分布直方图和校准参数

**验收标准**：

- ✅ `v_min > 0` 且 `v_max > v_min`
- ✅ `λ_0 ∈ [0.3, 0.8]` (合理范围)
- ✅ 配置文件已更新

---

### Step 2: 正式数据采集（Test Set）

**目标**：在测试集上运行完整流水线，生成 `router_features.jsonl`

```bash
python scripts/run_stage1_collection.py \
    --metadata ./data/raw/HWDB_Benchmark/test_metadata.jsonl \
    --config ./configs/router_config.yaml \
    --model_dir ./models/ppocrv5_rec \
    --output_dir ./results \
    --skip_agent_b \
    --verbose
```

**参数说明**：

- `--metadata`: 测试集 metadata
- `--config`: 配置文件路径（使用 Step 1 校准后的配置）
- `--model_dir`: Agent A 模型目录（必须提供）
- `--output_dir`: 输出目录
- `--skip_agent_b`: 跳过 Agent B（Stage 1 仅需 Agent A）
- `--verbose`: 打印详细日志

**预期输出**：

- `results/router_features.jsonl` （核心交付物）
- `results/stage1_collection_report.json` （采集统计报告）

**验收标准**：

- ✅ `router_features.jsonl` 包含所有测试样本
- ✅ 每行包含完整字段：`id`, `s_b`, `s_a`, `q`, `lambda_t`, `route_type`, `upgrade`, `blank_mean_L/R`, `top2_status`, `lat_router_ms`
- ✅ 处理成功率 > 95%

---

### Step 3: 预算稳定性测试

**目标**：验证 Actual Call Rate = B ± 0.5%

```bash
python scripts/test_budget_stability.py \
    --config ./configs/router_config.yaml \
    --metadata ./data/raw/HWDB_Benchmark/test_metadata.jsonl \
    --output ./results/call_rate_over_time.png \
    --verbose
```

**参数说明**：

- `--config`: 配置文件路径
- `--metadata`: 测试集 metadata
- `--output`: 可视化图表输出路径
- `--verbose`: 打印详细日志

**预期输出**：

- `results/call_rate_over_time.png` （λ 调整曲线图）
- 终端打印稳定性统计和硬约束检查结果

**验收标准**：

- ✅ `|Actual Call Rate - B| ≤ 0.5%`
- ✅ 95% 窗口 call rate 在 `B ± 3%`
- ✅ 无红色警告（震荡超过 `B ± 3%`）

---

### Step 4: 最终审计评估

**目标**：生成结项报表 `metrics_summary.json`

```bash
python scripts/evaluate.py \
    --predictions ./results/router_features.jsonl \
    --router_features ./results/router_features.jsonl \
    --output ./results/metrics_summary.json \
    --verbose
```

**参数说明**：

- `--predictions`: 预测结果文件（router_features.jsonl）
- `--router_features`: 路由特征文件（同上）
- `--output`: 输出 JSON 报告路径
- `--verbose`: 打印详细日志

**预期输出**：

- `results/metrics_summary.json` （核心指标汇总）
- 终端打印详细评估报表

**验收标准**：

- ✅ 包含所有核心指标：CER, Boundary Deletion Recall@B, Call Rate, CVR, AER, Latency P50/P95
- ✅ 报表格式符合 Stage 0/1 协议

---

## 📦 交付物清单

完成上述步骤后，`results/` 目录应包含以下文件：

| 文件                            | 描述             | 生成步骤 |
| ------------------------------- | ---------------- | -------- |
| `router_features.jsonl`         | 全量样本路由特征 | Step 2   |
| `metrics_summary.json`          | 核心指标汇总     | Step 4   |
| `stage1_collection_report.json` | 采集统计报告     | Step 2   |
| `calibration_stats.json`        | 校准统计结果     | Step 1   |
| `call_rate_over_time.png`       | λ 调整曲线       | Step 3   |

---

## 🔍 快速验证命令

```bash
# 检查交付物完整性
ls -lh results/*.jsonl results/*.json results/*.png

# 检查 router_features.jsonl 行数（应该等于测试集样本数）
wc -l results/router_features.jsonl

# 查看校准参数（从配置文件）
grep -A 10 "rule_scorer:" configs/router_config.yaml | grep -E "(v_min|v_max|lambda)"

# 查看最终指标摘要
cat results/metrics_summary.json | python -m json.tool | head -50
```

---

## ⚠️ 常见错误处理

### 错误 1: 模型文件不存在

```
[FATAL] 模型目录不存在: ./models/ppocrv5_rec
```

**解决**：下载模型（见前置条件检查）

### 错误 2: 图像路径解析失败

```
[Warning] 无法解析图像路径 (样本 X): test/sample_001.png
```

**解决**：检查 metadata 中的 `image_path` 格式，或使用 `--image_root` 参数

### 错误 3: 配置未校准

```
[Warning] 配置文件中未找到 'sh_da_v4' 部分
```

**解决**：先运行 Step 1（参数校准）

### 错误 4: Call Rate 不达标

```
✗ 失败 (误差: 0.8% > 0.5%)
```

**解决**：

- 检查 Step 1 的校准参数是否正确
- 增加 `calibrate_router.py` 的 `--limit` 样本数
- 检查数据分布是否异常

---

## 📊 预期性能指标

基于 HWDB 测试集的预期指标（参考值）：

- **Overall CER**: < 5%
- **Boundary Deletion Recall@B (B=20%)**: > 60%
- **Actual Call Rate**: 20.0% ± 0.5%
- **avg_lat_router_ms**: < 5ms
- **CVR**: < 30%
- **AER**: > 20%

---

## 📝 执行日志示例

成功的执行应该看到类似的输出：

```
======================================================================
  SH-DA++ v4.0 路由器参数校准
======================================================================
[Step 1] 加载 metadata...
  已加载 41780 个样本
[Step 2] 提取 v_edge 和计算 q 分数...
[✓] Agent A 初始化成功: ./models/ppocrv5_rec
  处理完成: 2000 个有效样本
  耗时: 245.32s (8.2 samples/s)
[Step 3] 计算分位数...
  校准结果:
  v_min (1% 分位数):   0.1234
  v_max (99% 分位数):  4.5678
  λ_0 (80% 分位数):    0.4321
[✓] 配置已更新: ./configs/router_config.yaml
```

---

## 🎯 验收检查清单

- [ ] Step 1 校准完成，配置文件已更新
- [ ] Step 2 生成完整的 `router_features.jsonl`
- [ ] Step 3 预算稳定性测试通过（误差 ≤ 0.5%）
- [ ] Step 4 生成完整的 `metrics_summary.json`
- [ ] 所有交付物文件存在且非空
- [ ] 日志中无 FATAL 错误
- [ ] 核心指标符合预期范围

---

**最后更新**: 2025-01-07  
**版本**: SH-DA++ v4.0 (严格模式，无模拟)
