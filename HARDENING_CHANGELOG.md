# L2W1 v5.0.1 代码加固记录

**日期**: 2025-12-21  
**版本**: v5.0.1 (Hardened)  

---

## 📋 加固清单

| # | 任务 | 文件 | 状态 |
|---|------|------|------|
| 1 | CTC 对齐回退策略 | `modules/router/uncertainty_router.py` | ✅ 完成 |
| 2 | 统一索引管理 | `modules/utils/indexing.py` (新建) | ✅ 完成 |
| 3 | 边界条件守护 | `modules/router/uncertainty_router.py` | ✅ 完成 |
| 4 | 负样本验证日志 | `scripts/sft_dataset.py` | ✅ 完成 |

---

## 🔧 详细修改

### 1. CTC 对齐回退策略 (核心修复)

**问题**: 当 CTC 解码字符数与文本长度不匹配时，直接使用均匀分布回退，导致 `suspicious_index` 偏移。

**解决方案**:

```python
# 新增三级对齐策略
TOLERANCE_WINDOW = 2          # 允许的字符数差异
EXTREME_MISMATCH_RATIO = 0.3  # 极端不匹配比例阈值

# 策略 1: 容错窗口 (±2 字符)
if length_diff <= TOLERANCE_WINDOW:
    return self._tolerant_align(...)

# 策略 2: 中等误差 - 贪婪映射 (基于熵权重)
if mismatch_ratio <= EXTREME_MISMATCH_RATIO:
    return self._greedy_align(...)

# 策略 3: 极端误差 (>30%) - 均匀回退 (最后手段)
return self._fallback_align(...)
```

**新增方法**:
- `_tolerant_align()`: 处理 ±2 字符误差，截断或填充
- `_greedy_align()`: 基于熵峰值的动态调整
- `get_alignment_stats()`: 获取对齐统计信息

---

### 2. 统一索引管理

**问题**: 0-indexed 与 1-indexed 转换分散在多个文件，维护风险高。

**解决方案**: 创建 `modules/utils/indexing.py`

```python
# 核心原则
# - 所有内部逻辑 (Router, Evaluator) 使用 0-indexed
# - 只在生成 Agent B Prompt 时转换为 1-indexed

from modules.utils import format_eip_index, to_display_index, validate_char_index

# 使用示例
prompt = f"第 {to_display_index(suspicious_idx)} 个字符..."

# 验证索引
is_valid, safe_idx = validate_char_index(idx, text)
```

**导出函数**:
- `to_display_index(zero_indexed)` → 1-indexed
- `from_display_index(one_indexed)` → 0-indexed
- `format_eip_index(idx, to_1_indexed=True)` → 格式化字符串
- `validate_char_index(idx, text)` → 边界检查
- `get_char_at_index(text, idx)` → 安全字符获取

---

### 3. 边界条件守护

**问题**: 空文本、单字符、极端长宽比未特殊处理。

**解决方案**: 在 `UncertaintyRouter.route()` 添加守护逻辑

```python
def route(self, logits, text, confidence=1.0, image_size=None):
    # Guard 1: 空文本 → CRITICAL
    if len(text) == 0:
        return RoutingResult(is_hard=True, risk_level="critical", ...)
    
    # Guard 2: 单字符 → 简化处理
    if len(text) == 1:
        return RoutingResult(suspicious_index=0, ...)
    
    # Guard 3: 极端长宽比 (>25:1) → 警告 + 降低置信度
    if image_size and width/height > 25.0:
        warnings.warn("极端长宽比检测...")
        confidence *= 0.8
```

---

### 4. 负样本验证日志

**问题**: 负样本提取逻辑可能导致图文不符。

**解决方案**: 在 `_add_negative_samples()` 添加调试日志

```python
# 初始化时打印 5 个随机负样本
logger.info("[负样本验证] 抽样检查:")
for sample in debug_samples[:5]:
    logger.info(f"  ID: {sample['neg_id']} <- {sample['source_id']}")
    logger.info(f"  Image: {sample['image']}")
    logger.info(f"  Text: '{sample['correct_text']}'")
```

---

## 📊 测试验证

### 边界条件测试

| 场景 | 输入 | 预期结果 | 实际结果 |
|------|------|----------|----------|
| 空文本 | `text=""` | `is_hard=True, risk=critical` | ✅ 通过 |
| 单字符 | `text="中"` | `is_hard=True/False, idx=0` | ✅ 通过 |
| 正常文本 | `text="中国科学院"` | 正常路由 | ✅ 通过 |

### 对齐策略测试

| 场景 | 长度差异 | 使用策略 |
|------|----------|----------|
| 完美匹配 | 0 | `perfect_match` |
| 小误差 | ≤2 | `tolerant_align` |
| 中等误差 | ≤30% | `greedy_align` |
| 极端误差 | >30% | `fallback_align` |

---

## 📁 新增/修改文件

### 新增文件
- `modules/utils/__init__.py`
- `modules/utils/indexing.py`

### 修改文件
- `modules/router/uncertainty_router.py` - CTC 对齐 + 边界守护
- `modules/vlm_expert/agent_b_expert.py` - 统一索引管理
- `scripts/data_pipeline.py` - 统一索引管理
- `scripts/sft_dataset.py` - 负样本验证日志

---

## 🚀 升级指南

1. 无需修改调用代码，API 保持兼容
2. 新增 `image_size` 参数为可选
3. 建议更新调用方使用统一索引工具

```python
# 推荐用法
from modules.utils import to_display_index, validate_char_index

# 在 Prompt 中使用
display_idx = to_display_index(router_result.suspicious_index)
prompt = f"第 {display_idx} 个字符..."

# 验证索引
is_valid, safe_idx = validate_char_index(idx, text)
```

---

**加固完成** ✅

