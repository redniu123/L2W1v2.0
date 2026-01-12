# SH-DA++ v4.0 Stage 0/1 重启执行手册

> **重要提示**：由于之前系统错误地调用了模拟引擎（Mock Mode），所有校准参数和性能指标均基于无效的随机噪声。**重新校准与重新压测是绝对必要的**。

---

## 📋 前置条件检查

在执行前，请确认以下环境：

```bash
# 1. 检查模型目录
ls -la models/agent_a_ppocr/PP-OCRv5_server_rec_infer/
# 应看到：inference.pdiparams, inference.json, inference.yml, ppocr_keys.txt

# 2. 检查数据集
ls -la data/raw/HWDB_Benchmark/
# 应看到：train_metadata.jsonl, test_metadata.jsonl, test/ (图像目录)

# 3. 检查字典文件
ls -la ppocr/utils/ppocrv5_dict.txt
```

---

## 🚀 标准作业程序 (SOP)

### **第一步：环境重置与清洁**

```bash
cd L2W1

# 清理所有旧结果（防止随机噪声污染）
rm -rf results/*.jsonl results/*.json results/*.png results/debug/
mkdir -p results/debug

echo "[✓] 环境清洁完成"
```

**目的**：防止旧的随机噪声数据污染新生成的真实信号日志。

---

### **第二步：参数重新校准 (`calibrate_router.py`)**

**执行命令**：

```bash
python scripts/calibrate_router.py \
    --config configs/router_config.yaml \
    --metadata ./data/raw/HWDB_Benchmark/train_metadata.jsonl \
    --model_dir ./models/agent_a_ppocr/PP-OCRv5_server_rec_infer/ \
    --target_b 0.2 \
    --limit 1000
```

**观察重点**：

1. ✅ **控制台识别文本**：观察是否输出了真实的汉字（如"胡锦涛强调..."），确保 PP-OCRv5 已就位。
2. ✅ **q 分数分布**：检查 `q` 的直方图。如果 `min` 和 `max` 之间出现了明显的差值（例如在 0.2 到 0.9 之间分布），说明信号已正常。
3. ✅ **校准参数输出**：
   - `v_min` (v_edge 的 1% 分位数)
   - `v_max` (v_edge 的 99% 分位数)
   - `λ_0` (q 的 80% 分位数，作为初始阈值)

**提交数据**：

- `results/calibration_stats.json`
- 控制台打印的 `v_min`, `v_max`, `lambda_0` 数值

**预期输出示例**：

```
[Step 3] 计算分位数...
  校准结果:
  v_min (1% 分位数):   12.3456
  v_max (99% 分位数):  98.7654
  λ_0 (80% 分位数):    0.4321
[✓] 配置已更新: configs/router_config.yaml
```

---

### **第三步：正式数据采集 (`run_stage1_collection.py`)**

**执行命令**：

```bash
python scripts/run_stage1_collection.py \
    --metadata ./data/raw/HWDB_Benchmark/test_metadata.jsonl \
    --config ./configs/router_config.yaml \
    --model_dir ./models/agent_a_ppocr/PP-OCRv5_server_rec_infer/ \
    --output_dir ./results \
    --skip_agent_b
```

**观察重点**：

1. ✅ **实时状态**：观察每 500 个样本打印的 `[ID] '识别文本' | q=分数 | conf=置信度 | s_b=边界分 | s_a=歧义分`
   - 抽检 q 分数是否随文本难度变化（不应全部相同）
   - 识别文本应为真实汉字，而非随机字符串
2. ✅ **平均耗时**：观察 `avg_lat_router_ms` 是否维持在 10ms 以内
3. ✅ **数据备份**：如果存在旧 `router_features.jsonl`，会自动备份为 `router_features_backup_YYYYMMDD_HHMMSS.jsonl`

**提交数据**：

- `results/router_features.jsonl` 的前 10 行
- `results/stage1_collection_report.json`

**预期输出示例**：

```
[实时抽检模式] 每 500 个样本打印 1 条识别结果
  [sample_000001] '中国科学院计算技术研究所' | q=0.5234 | conf=87.5% | s_b=0.1234 | s_a=0.4567
  [sample_000501] '在时间的未尾' | q=0.6789 | conf=92.3% | s_b=0.2345 | s_a=0.5678
...
[✓] 数据采集完成: 10449 个样本
  - upgrade_rate: 0.20XX
  - avg_lat_router_ms: X.XX ms
```

---

### **第四步：稳定性压测 (`test_budget_stability.py`)**

**执行命令**：

```bash
python scripts/test_budget_stability.py \
    --config ./configs/router_config.yaml \
    --metadata ./data/raw/HWDB_Benchmark/test_metadata.jsonl \
    --model_dir ./models/agent_a_ppocr/PP-OCRv5_server_rec_infer/ \
    --output ./results/call_rate_over_time.png
```

**观察重点**：

1. ✅ **误差检查**：`|B̄_total - B|` 是否 ≤ 0.5%
2. ✅ **震荡检查**：最大震荡是否降至 ±3% 以内，标准差是否降至 0.01 以下
3. ✅ **图表质量**：
   - 中文显示正常（应显示"阈值 λ"、"调用率"等中文字符）
   - DPI 为 300（论文级质量）
   - 图表清晰，无模糊

**提交数据**：

- `results/call_rate_over_time.png`（检查中文显示与 300 DPI 清晰度）
- 控制台输出的稳定性评估结果

**预期输出示例**：

```
【硬约束检查】|Actual - B| ≤ 0.5%:
  ✓ 通过 (误差: 0.23%)

【震荡检查】是否超过 B ± 3%:
  ✓ 通过 (最大震荡: 2.1%)

[✓] 可视化图表已保存: ./results/call_rate_over_time.png (dpi=300, 论文级质量)
```

---

### **第五步：最终审计 (`evaluate.py`)**

**执行命令**：

```bash
python scripts/evaluate.py \
    --predictions ./results/router_features.jsonl \
    --router_features ./results/router_features.jsonl \
    --output ./results/metrics_summary.json
```

**观察重点**：

1. ✅ **`Boundary Deletion Recall@B`**：这是核心指标，必须大于 0.0（理想值 > 0.6）
2. ✅ **`Overall CER Improvement`**：查看 System CER 相比 Agent A 是否有实质下降
3. ✅ **`CVR` (Constraint Violation Rate)**：应 < 30%
4. ✅ **`AER` (Accepted Edit Rate)**：应 > 20%

**提交数据**：

- `results/metrics_summary.json`

**预期输出示例**：

```
======================================================================
  SH-DA++ v4.0 评估结果
======================================================================
Overall CER:
  Agent A:    4.23%
  System:     3.45%
  Improvement: 0.78%

Boundary Deletion Recall@B (B=20%): 0.65 (65.0%)

Reliability Metrics:
  CVR (Constraint Violation Rate): 0.25 (25.0%)
  AER (Accepted Edit Rate): 0.32 (32.0%)
...
```

---

## ⚠️ 结项判定红线

在执行过程中，如果遇到以下情况，请立即停止并反馈，这通常意味着代码逻辑仍有隐患：

### **红线 1：q 分数无梯度**

- **现象**：q 分数的标准差 (`std`) 仍然为 0 或接近 0
- **原因**：可能仍在使用 Mock 模式或 logits 温度参数设置错误
- **检查方法**：
  ```bash
  python -c "
  import json
  q_list = []
  with open('results/router_features.jsonl') as f:
      for line in f:
          q_list.append(json.loads(line)['q'])
  import numpy as np
  print(f'q std: {np.std(q_list):.6f}')
  print(f'q min: {min(q_list):.4f}, max: {max(q_list):.4f}')
  "
  ```
- **预期值**：`std > 0.1`, `max - min > 0.3`

### **红线 2：预算控制器无法收敛**

- **现象**：`Actual Call Rate` 依然无法收敛，或者 λ 很快冲到了上限 2.0
- **原因**：k 参数过大或窗口大小 W 过小
- **检查方法**：查看 `test_budget_stability.py` 输出的误差和震荡指标
- **预期值**：`|Actual - B| ≤ 0.5%`, `λ ∈ [0.3, 1.5]`

### **红线 3：图表中文显示异常或分辨率低**

- **现象**：生成的 PNG 图片无法显示中文或分辨率极低
- **原因**：字体配置错误或 DPI 设置错误
- **检查方法**：
  ```bash
  file results/call_rate_over_time.png  # 查看文件信息
  identify -verbose results/call_rate_over_time.png | grep Resolution  # 查看分辨率
  ```
- **预期值**：DPI = 300，中文正常显示

---

## 📊 完整执行命令（一键复制）

```bash
# ============================================================
# Step 0: 环境清理
# ============================================================
cd L2W1
rm -rf results/*.jsonl results/*.json results/*.png results/debug/
mkdir -p results/debug

# ============================================================
# Step 1: 参数校准 (Train Set, 1000 samples)
# ============================================================
python scripts/calibrate_router.py \
    --config configs/router_config.yaml \
    --metadata ./data/raw/HWDB_Benchmark/train_metadata.jsonl \
    --model_dir ./models/agent_a_ppocr/PP-OCRv5_server_rec_infer/ \
    --target_b 0.2 \
    --limit 1000

# ============================================================
# Step 2: 正式数据采集 (Test Set, 全部样本)
# ============================================================
python scripts/run_stage1_collection.py \
    --metadata ./data/raw/HWDB_Benchmark/test_metadata.jsonl \
    --config ./configs/router_config.yaml \
    --model_dir ./models/agent_a_ppocr/PP-OCRv5_server_rec_infer/ \
    --output_dir ./results \
    --skip_agent_b

# ============================================================
# Step 3: 预算稳定性测试
# ============================================================
python scripts/test_budget_stability.py \
    --config ./configs/router_config.yaml \
    --metadata ./data/raw/HWDB_Benchmark/test_metadata.jsonl \
    --model_dir ./models/agent_a_ppocr/PP-OCRv5_server_rec_infer/ \
    --output ./results/call_rate_over_time.png

# ============================================================
# Step 4: 最终审计评估
# ============================================================
python scripts/evaluate.py \
    --predictions ./results/router_features.jsonl \
    --router_features ./results/router_features.jsonl \
    --output ./results/metrics_summary.json

# ============================================================
# 快速验证
# ============================================================
echo "=== 交付物检查 ==="
ls -lh results/*.jsonl results/*.json results/*.png 2>/dev/null | head -10
wc -l results/router_features.jsonl 2>/dev/null
```

---

## 📝 关键配置参数确认

执行前请确认 `configs/router_config.yaml` 中的关键参数：

```yaml
agent_a:
  model_dir: "./models/agent_a_ppocr/PP-OCRv5_server_rec_infer/"
  rec_image_shape: "3, 48, 320"
  rec_char_dict_path: "./ppocr/utils/ppocrv5_dict.txt"

sh_da_v4:
  budget_controller:
    window_size: 500 # 增大窗口 (已优化)
    k: 0.01 # 减小比例系数 (已优化)
    target_budget: 0.2 # 目标调用率 20%

stage0:
  softmax_temperature: 0.1 # 温度参数 (关键修复)
```

---

**最后更新**: 2025-01-07  
**版本**: SH-DA++ v4.0 (真实模式，无 Mock)
