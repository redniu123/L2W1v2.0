# SH-DA++ v4.0 Stage 0 验证检查清单

## ✅ 检查项 1: 显存安全

### 问题
是否使用了 `batch_raw_logits[rno].copy()` 或 `deepcopy()` 防止 Paddle 显存复用导致数据被覆盖？

### 状态: ✅ **已修复**

**修复位置**: `predict_rec_modified.py` 第 1103 行

**修复前**:
```python
sample_logits = batch_raw_logits[rno]  # ❌ 直接引用，可能被覆盖
```

**修复后**:
```python
# 使用 .copy() 防止 Paddle 显存复用导致数据被覆盖
sample_logits = batch_raw_logits[rno].copy()  # ✅ 安全拷贝
```

**说明**:
- `batch_raw_logits` 本身已经在第 1061、1073、1076 行使用 `deepcopy()` 从 outputs 中提取
- 在第 1103 行提取单个样本时，额外使用 `.copy()` 确保数据安全
- 这样可以防止 PaddlePredictor 在后续推理中复用显存覆盖数据

---

## ✅ 检查项 2: 配置一致性

### 问题
`blank_id` 是否硬编码为 0？是否应从 args 或模型配置中动态获取？

### 状态: ✅ **已修复**

**修复位置**: `predict_rec_modified.py` 第 368-377 行

**修复前**:
```python
self.blank_id = 0  # ❌ 硬编码
```

**修复后**:
```python
# 从 postprocess_op 动态获取 blank_id (CTCLabelDecode 约定为 0)
self.blank_id = 0  # 默认值
if hasattr(self.postprocess_op, 'get_ignored_tokens'):
    ignored_tokens = self.postprocess_op.get_ignored_tokens()
    if ignored_tokens and len(ignored_tokens) > 0:
        self.blank_id = ignored_tokens[0]  # ✅ 动态获取
```

**说明**:
- `CTCLabelDecode.get_ignored_tokens()` 返回 `[0]`，符合 PaddleOCR 约定
- 如果未来模型使用不同的 blank_id，代码会自动适配
- 提供了默认值 `0` 作为回退

---

## ⚠️ 检查项 3: 计算准确性

### 问题
同一张图运行两次，`blank_mean_L` 是否二进制对齐（完全一致）？

### 状态: ✅ **理论上已保证，需运行时验证**

**保证措施**:

1. **显存安全**: 已使用 `.copy()` 防止数据被覆盖
2. **数值稳定性**: `compute_softmax()` 使用标准的 log-sum-exp trick
3. **确定性计算**: NumPy 操作在固定输入下应产生相同输出

**验证方法**:
```python
# 运行两次相同图像
result1 = recognizer([img])[0]
result2 = recognizer([img])[0]

# 检查二进制对齐
blank_mean_L_1 = result1['boundary_stats'][0]['blank_mean_L']
blank_mean_L_2 = result2['boundary_stats'][0]['blank_mean_L']

assert blank_mean_L_1 == blank_mean_L_2, "计算结果不一致！"
```

**潜在问题**:
- ⚠️ **浮点运算顺序**: 如果使用了并行计算，可能存在浮点运算顺序差异
- ⚠️ **随机性**: 如果模型推理包含随机性（如 dropout），结果会不同
- ✅ **Emission 矩阵**: `compute_softmax()` 是确定性操作，应保证一致性

**建议**:
- 在实际测试中验证同一图像的 `blank_mean_L` 是否完全一致
- 如果发现不一致，检查是否有其他随机性来源

---

## ✅ 检查项 4: 失败模式处理

### 问题
若 Top-2 提取失败，`top2_status` 是否正确落盘为 `'missing'`？

### 状态: ✅ **已修复并增强**

**修复位置**: `predict_rec_modified.py` 第 1117-1160 行

**修复前**:
```python
if self.character_list is not None:
    # ... 转换字符
    top2_status = 'available'
else:
    top2_status = 'missing'  # ❌ 仅在字符表缺失时标记
```

**修复后**:
```python
top2_status = 'available'  # 默认状态
top2_info = None

try:
    # Top-2 提取逻辑
    top2_indices = np.argsort(E, axis=-1)[:, -2:][:, ::-1]
    top2_probs = np.take_along_axis(E, top2_indices, axis=-1)
    # ... 计算统计量
    
    if self.character_list is not None and len(self.character_list) > 0:
        # 转换字符
        top2_status = 'available'
    else:
        top2_status = 'available_no_chars'  # ✅ 字符表不可用但提取成功
    
except Exception as e:
    # ✅ Top-2 提取失败，标记为 missing
    top2_status = 'missing'
    top2_info = {
        'top2_status': top2_status,
        'T': T,
        'C': C,
        'top1_conf_mean': 0.0,
        'top2_conf_mean': 0.0,
        'conf_gap_mean': 0.0,
        'error': str(e),  # ✅ 记录错误信息
    }
```

**状态码说明**:
- `'available'`: Top-2 提取成功，字符转换完成
- `'available_no_chars'`: Top-2 提取成功，但字符表不可用（仅统计量可用）
- `'missing'`: Top-2 提取失败（异常捕获）

**增强功能**:
1. ✅ 使用 `try-except` 捕获所有异常
2. ✅ 在失败时记录错误信息 (`error` 字段)
3. ✅ 区分"提取成功但无字符表"和"提取失败"两种情况

---

## 📊 总结

| 检查项 | 状态 | 备注 |
|--------|------|------|
| 1. 显存安全 | ✅ 已修复 | 使用 `.copy()` 确保数据安全 |
| 2. 配置一致性 | ✅ 已修复 | 从 `postprocess_op.get_ignored_tokens()` 动态获取 |
| 3. 计算准确性 | ⚠️ 需验证 | 理论上保证，建议运行时验证 |
| 4. 失败模式处理 | ✅ 已修复 | 完善的异常处理和状态码 |

---

## 🔍 推荐测试

```python
# 测试 1: 显存安全验证
# 连续运行两次相同 batch，检查结果是否一致

# 测试 2: 配置一致性验证
# 使用不同的模型配置，验证 blank_id 是否正确获取

# 测试 3: 计算准确性验证
# 同一张图运行两次，检查 blank_mean_L 是否二进制对齐

# 测试 4: 失败模式验证
# 模拟字符表缺失、E 矩阵异常等情况，验证 top2_status
```

---

**检查完成时间**: 2024-12-19  
**代码版本**: SH-DA++ v4.0 Stage 0

