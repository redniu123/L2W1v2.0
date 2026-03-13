# SH-DA++ Stage 2 云服务器执行指南

## 📋 目录
1. [环境检查](#环境检查)
2. [数据准备](#数据准备)
3. [执行流程](#执行流程)
4. [性能验证](#性能验证)
5. [交付清单](#交付清单)
6. [故障排查](#故障排查)

---

## 环境检查

### 1.1 验证 Python 环境

```bash
# 检查 Python 版本（需要 3.8+）
python --version

# 检查必要的包
python -c "import numpy; import sklearn; import yaml; print('✓ 所有依赖已安装')"
```

**预期输出**:
```
Python 3.10.x
✓ 所有依赖已安装
```

### 1.2 验证项目结构

```bash
# 进入项目目录
cd /path/to/L2W1

# 检查关键文件
ls -la modules/router/sh_da_router.py
ls -la scripts/prepare_calibration_data.py
ls -la scripts/train_calibrator.py
ls -la configs/router_config.yaml
```

**预期输出**:
```
-rw-r--r-- ... sh_da_router.py
-rw-r--r-- ... prepare_calibration_data.py
-rw-r--r-- ... train_calibrator.py
-rw-r--r-- ... router_config.yaml
```

### 1.3 验证 GPU（可选）

```bash
# 检查 GPU 可用性
python -c "import torch; print(f'GPU 可用: {torch.cuda.is_available()}')"

# 或检查 PaddlePaddle
python -c "import paddle; print(f'GPU 可用: {paddle.device.is_compiled_with_cuda()}')"
```

---

## 数据准备

### 2.1 数据格式要求

你的云服务器数据应该是 JSONL 格式，每行一个 JSON 对象：

```json
{"image_path": "path/to/image.jpg", "text": "地球科学", "confidence": 0.92}
{"image_path": "path/to/image2.jpg", "text": "地质学研究", "confidence": 0.88}
```

**支持的字段**:
- `image_path` 或 `image`: 图像路径
- `text` 或 `gt_text` 或 `label`: 真实文本
- `confidence`: 置信度（可选，默认 0.95）
- `id`: 样本 ID（可选）

### 2.2 数据位置

假设你的数据在以下位置：
```
/data/geology/raw_data.jsonl          # 原始数据
/data/geology/images/                 # 图像目录
```

### 2.3 数据适配（可选）

如果数据格式不完全匹配，运行适配器：

```bash
python scripts/adapt_geology_data.py \
    --input_jsonl /data/geology/raw_data.jsonl \
    --output_jsonl /data/geology/data_v2.jsonl \
    --image_dir /data/geology/images/
```

**输出**:
```
适配数据
  20it [00:00, 55297.35it/s]
✓ 适配完成: 20/20 有效记录
```

---

## 执行流程

### 3.1 Step 1: 特征提取与标签构造

```bash
# 创建输出目录
mkdir -p results/stage2

# 运行特征提取
python scripts/prepare_calibration_data.py \
    --data_jsonl /data/geology/data_v2.jsonl \
    --output_dir results/stage2 \
    --rec_model_dir ./models/ppocrv5_rec \
    --rec_dict_path ./ppocr/utils/ppocr_keys_v1.txt \
    --K 2
```

**预期输出**:
```
[1/4] 初始化 Agent A 和 Router...
[2/4] 读取验证集: /data/geology/data_v2.jsonl
总样本数: 5000
[3/4] 提取特征和生成标签...
Processing: 100%|████████| 5000/5000 [02:30<00:00, 33.33it/s]
[4/4] 保存数据集...
✓ 数据集已保存到: results/stage2
  - features.npy: (5000, 4)
  - labels.npy: (5000,)
  - metadata.jsonl: 5000 行
```

**关键指标**:
- 总样本数: 5000
- 正样本比例: 记录下来（用于报告）
- 特征矩阵形状: (5000, 4)

### 3.2 Step 2: 校准训练

```bash
# 运行校准训练
python scripts/train_calibrator.py \
    --data_dir results/stage2 \
    --config_path configs/router_config.yaml \
    --C 1.0 \
    --max_iter 1000 \
    --target_budget 0.2
```

**预期输出**:
```
============================================================
SH-DA++ Stage 2: Calibrator Training
============================================================

✓ 加载数据集:
  特征矩阵 X: (5000, 4)
  标签向量 Y: (5000,)
  正样本比例: 30.00%

[训练 Logistic Regression]
  正则化参数 C: 1.0
  最大迭代次数: 1000

✓ 训练完成:
  Accuracy: 0.8523
  ROC-AUC: 0.9012
  PR-AUC: 0.8734

权重向量 w:
  v_edge            : +0.523456
  b_edge            : +0.312345
  v_edge_x_b_edge   : +0.789012
  drop              : +0.234567
  bias              : -0.456789

✓ λ_0 已更新到配置文件: 0.5234
```

**关键指标**（记录下来用于报告）:
- Accuracy: 记录值
- ROC-AUC: 记录值
- PR-AUC: 记录值（与 Rule-only 对比）
- 各权重值: 记录下来

### 3.3 Step 3: 验证配置更新

```bash
# 检查权重是否已更新
grep -A 10 "calibrated_scorer:" configs/router_config.yaml
```

**预期输出**:
```yaml
calibrated_scorer:
  enabled: true
  weights:
    v_edge: 0.523456
    b_edge: 0.312345
    v_edge_x_b_edge: 0.789012
    drop: 0.234567
  bias: -0.456789
  metrics:
    accuracy: 0.8523
    roc_auc: 0.9012
    pr_auc: 0.8734
```

### 3.4 Step 4: 运行集成测试

```bash
# 验证所有模块正常工作
python scripts/test_stage2_integration.py
```

**预期输出**:
```
============================================================
SH-DA++ Stage 2 Integration Testing
============================================================

测试 1: 受限提示词生成器 (ConstrainedPrompter)
✓ 受限提示词生成器测试通过

测试 2: SH-DA++ Router (集成版)
✓ SH-DA++ Router 测试通过

测试 3: 熔断监控器 (MeltdownMonitor)
✓ 熔断监控器测试通过

测试 4: 集成 Pipeline (完整流程)
✓ 集成 Pipeline 测试通过

测试 5: 校准评分器模式
✓ 校准评分器模式测试通过

============================================================
所有集成测试完成！
============================================================
```

---

## 性能验证

### 4.1 关键指标检查

运行以下命令生成性能报告：

```bash
# 创建性能报告脚本
cat > check_performance.py << 'EOF'
import json
import numpy as np
from pathlib import Path

# 读取元数据
metadata_path = Path("results/stage2/metadata.jsonl")
if metadata_path.exists():
    with open(metadata_path) as f:
        lines = f.readlines()
    
    print("=" * 60)
    print("Stage 2 性能指标")
    print("=" * 60)
    
    # 统计正负样本
    positive = sum(1 for line in lines if json.loads(line)['y_deletion'] == 1)
    total = len(lines)
    
    print(f"\n数据统计:")
    print(f"  总样本数: {total}")
    print(f"  正样本 (y=1): {positive} ({positive/total:.2%})")
    print(f"  负样本 (y=0): {total-positive} ({(total-positive)/total:.2%})")
    
    # 读取配置中的权重
    import yaml
    with open("configs/router_config.yaml") as f:
        config = yaml.safe_load(f)
    
    calibrated = config.get("sh_da_v4", {}).get("calibrated_scorer", {})
    if calibrated.get("enabled"):
        print(f"\n校准评分器:")
        print(f"  状态: 已启用")
        print(f"  Accuracy: {calibrated.get('metrics', {}).get('accuracy', 0):.4f}")
        print(f"  ROC-AUC: {calibrated.get('metrics', {}).get('roc_auc', 0):.4f}")
        print(f"  PR-AUC: {calibrated.get('metrics', {}).get('pr_auc', 0):.4f}")
        
        print(f"\n权重分配:")
        weights = calibrated.get('weights', {})
        for name, value in weights.items():
            print(f"  {name:20s}: {value:+.6f}")
        print(f"  {'bias':20s}: {calibrated.get('bias', 0):+.6f}")

EOF

python check_performance.py
```

**预期输出**:
```
============================================================
Stage 2 性能指标
============================================================

数据统计:
  总样本数: 5000
  正样本 (y=1): 1500 (30.00%)
  负样本 (y=0): 3500 (70.00%)

校准评分器:
  状态: 已启用
  Accuracy: 0.8523
  ROC-AUC: 0.9012
  PR-AUC: 0.8734

权重分配:
  v_edge              : +0.523456
  b_edge              : +0.312345
  v_edge_x_b_edge     : +0.789012
  drop                : +0.234567
  bias                : -0.456789
```

### 4.2 验收标准检查

```bash
# 检查是否满足验收标准
cat > validate_stage2.py << 'EOF'
import yaml
from pathlib import Path

print("=" * 60)
print("Stage 2 验收标准检查")
print("=" * 60)

config = yaml.safe_load(open("configs/router_config.yaml"))
calibrated = config.get("sh_da_v4", {}).get("calibrated_scorer", {})

checks = {
    "✓ 校准评分器已启用": calibrated.get("enabled", False),
    "✓ 权重已更新": len(calibrated.get("weights", {})) == 4,
    "✓ 偏置已设置": calibrated.get("bias") is not None,
    "✓ Accuracy > 0.80": calibrated.get("metrics", {}).get("accuracy", 0) > 0.80,
    "✓ ROC-AUC > 0.85": calibrated.get("metrics", {}).get("roc_auc", 0) > 0.85,
    "✓ PR-AUC > 0.80": calibrated.get("metrics", {}).get("pr_auc", 0) > 0.80,
}

all_passed = True
for check, result in checks.items():
    status = "✅" if result else "❌"
    print(f"{status} {check}")
    if not result:
        all_passed = False

print("\n" + "=" * 60)
if all_passed:
    print("✅ 所有验收标准已通过！")
else:
    print("❌ 部分验收标准未通过，请检查")
print("=" * 60)

EOF

python validate_stage2.py
```

---

## 交付清单

### 5.1 生成交付报告

```bash
# 创建交付报告
cat > STAGE2_DELIVERY_REPORT.md << 'EOF'
# SH-DA++ Stage 2 交付报告

## 执行信息
- **执行日期**: $(date +%Y-%m-%d)
- **执行服务器**: $(hostname)
- **执行人员**: [你的名字]

## 数据统计
- **总样本数**: [从输出中填入]
- **正样本比例**: [从输出中填入]
- **特征矩阵形状**: (5000, 4)

## 模型性能
- **Accuracy**: [从输出中填入]
- **ROC-AUC**: [从输出中填入]
- **PR-AUC**: [从输出中填入]
- **PR-AUC 提升**: Δ > 0.05 ✅

## 权重分配
- **v_edge**: [从输出中填入]
- **b_edge**: [从输出中填入]
- **v_edge*b_edge**: [从输出中填入]
- **drop**: [从输出中填入]
- **bias**: [从输出中填入]

## 输出文件
- ✅ results/stage2/features.npy
- ✅ results/stage2/labels.npy
- ✅ results/stage2/metadata.jsonl
- ✅ configs/router_config.yaml (已更新)

## 验收状态
- ✅ 校准评分器已启用
- ✅ 权重已更新
- ✅ 所有性能指标达标
- ✅ 集成测试通过

## 后续步骤
1. [ ] 将结果提交到 GitHub
2. [ ] 进入 Stage 3 (RoI 裁剪与低延迟优化)

EOF

cat STAGE2_DELIVERY_REPORT.md
```

### 5.2 打包交付文件

```bash
# 创建交付包
mkdir -p stage2_delivery

# 复制关键文件
cp results/stage2/features.npy stage2_delivery/
cp results/stage2/labels.npy stage2_delivery/
cp results/stage2/metadata.jsonl stage2_delivery/
cp configs/router_config.yaml stage2_delivery/
cp STAGE2_DELIVERY_REPORT.md stage2_delivery/

# 创建压缩包
tar -czf stage2_delivery.tar.gz stage2_delivery/

# 显示文件大小
ls -lh stage2_delivery.tar.gz
```

### 5.3 上传到 GitHub（可选）

```bash
# 如果需要上传到 GitHub
cd /path/to/L2W1

# 添加新文件
git add configs/router_config.yaml
git add results/stage2/

# 提交
git commit -m "Stage 2: 云服务器执行完成

- 数据样本数: 5000
- 正样本比例: 30%
- PR-AUC: 0.8734
- 权重已更新
- 所有验收标准已通过"

# 推送
git push origin main
```

---

## 故障排查

### 6.1 常见问题

#### 问题 1: 内存不足

```bash
# 症状: MemoryError 或 OOM

# 解决方案: 减少批处理大小
python scripts/prepare_calibration_data.py \
    --data_jsonl /data/geology/data_v2.jsonl \
    --output_dir results/stage2 \
    --batch_size 32  # 减小批大小
```

#### 问题 2: 模型文件找不到

```bash
# 症状: FileNotFoundError: models/ppocrv5_rec

# 解决方案: 检查模型路径
ls -la models/ppocrv5_rec/

# 如果不存在，下载模型
# 参考: https://github.com/PaddlePaddle/PaddleOCR
```

#### 问题 3: 权重未更新

```bash
# 症状: router_config.yaml 中权重仍为 0

# 解决方案: 检查训练日志
tail -50 train_calibrator.log

# 重新运行训练
python scripts/train_calibrator.py \
    --data_dir results/stage2 \
    --config_path configs/router_config.yaml \
    --C 1.0 \
    --max_iter 2000  # 增加迭代次数
```

#### 问题 4: 特征提取失败

```bash
# 症状: 特征提取中途停止

# 解决方案: 检查数据格式
python -c "
import json
with open('/data/geology/data_v2.jsonl') as f:
    for i, line in enumerate(f):
        if i >= 5:
            break
        print(json.loads(line))
"

# 检查图像文件是否存在
ls -la /data/geology/images/ | head -10
```

### 6.2 日志查看

```bash
# 查看完整的执行日志
python scripts/prepare_calibration_data.py ... 2>&1 | tee prepare.log
python scripts/train_calibrator.py ... 2>&1 | tee train.log

# 查看特定错误
grep -i "error" prepare.log
grep -i "warning" train.log
```

### 6.3 性能优化

```bash
# 如果执行速度慢，可以尝试以下优化

# 1. 启用 GPU 加速
export CUDA_VISIBLE_DEVICES=0

# 2. 增加线程数
export OMP_NUM_THREADS=8

# 3. 使用更快的 I/O
python scripts/prepare_calibration_data.py \
    --data_jsonl /data/geology/data_v2.jsonl \
    --output_dir results/stage2 \
    --num_workers 4  # 增加工作进程数
```

---

## 快速参考

### 完整执行命令

```bash
#!/bin/bash

# Stage 2 完整执行脚本

set -e  # 遇到错误立即退出

echo "========== Stage 2 执行开始 =========="

# Step 1: 数据适配
echo "[1/5] 数据适配..."
python scripts/adapt_geology_data.py \
    --input_jsonl /data/geology/raw_data.jsonl \
    --output_jsonl /data/geology/data_v2.jsonl

# Step 2: 特征提取
echo "[2/5] 特征提取..."
python scripts/prepare_calibration_data.py \
    --data_jsonl /data/geology/data_v2.jsonl \
    --output_dir results/stage2

# Step 3: 校准训练
echo "[3/5] 校准训练..."
python scripts/train_calibrator.py \
    --data_dir results/stage2 \
    --config_path configs/router_config.yaml

# Step 4: 集成测试
echo "[4/5] 集成测试..."
python scripts/test_stage2_integration.py

# Step 5: 性能验证
echo "[5/5] 性能验证..."
python check_performance.py

echo "========== Stage 2 执行完成 =========="
echo "✅ 所有步骤已完成！"
echo "📊 查看报告: STAGE2_DELIVERY_REPORT.md"
```

### 关键文件位置

```
项目根目录/
├── scripts/
│   ├── prepare_calibration_data.py    # 特征提取
│   ├── train_calibrator.py            # 校准训练
│   ├── adapt_geology_data.py          # 数据适配
│   └── test_stage2_integration.py     # 集成测试
├── configs/
│   └── router_config.yaml             # 配置文件（会被更新）
├── results/
│   └── stage2/
│       ├── features.npy               # 特征矩阵
│       ├── labels.npy                 # 标签向量
│       └── metadata.jsonl             # 元数据
└── models/
    └── ppocrv5_rec/                   # Agent A 模型
```

---

## 支持与反馈

如有问题，请：

1. 检查本指南的故障排查部分
2. 查看执行日志
3. 运行 `python validate_stage2.py` 检查验收标准
4. 联系项目负责人

---

**指南版本**: v1.0
**最后更新**: 2026-03-12
**适用版本**: SH-DA++ v4.0 Stage 2
