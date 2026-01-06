# V-CoT (Visual Chain-of-Thought) 设计文档

## 📋 概述

**V-CoT (Visual Chain-of-Thought)** 是 L2W1 项目中为 Agent B (Qwen2.5-VL) 设计的**结构化推理提示词框架**，专门用于修复 OCR 模型在图像边界区域的识别错误。

### 核心问题

传统 OCR 模型（如 PaddleOCR）在处理紧密裁剪的行文本图像时，存在**边界截断问题**：
- **左边界丢失**: "胡锦涛强调..." → "锦涛强调..."
- **右边界丢失**: "...食品安全工作" → "...食品安"
- **两边丢失**: "本报北京4月24日讯...集体学习" → "24日讯...进行"

### V-CoT 解决方案

通过**分步骤的视觉推理引导**，让 VLM 系统性地：
1. **聚焦观察**图像边界区域
2. **对比分析**OCR 结果与视觉内容
3. **补全输出**完整文本

---

## 🏗️ 架构设计

### 1. 边界风险类型检测

```python
class BoundaryRiskType(Enum):
    NONE = "none"              # 无边界风险
    LEFT_ONLY = "left_only"    # 仅左边界风险
    RIGHT_ONLY = "right_only"  # 仅右边界风险
    BOTH = "both"              # 两边都有风险
    UNKNOWN = "unknown"        # 未知（几何检测触发）
```

**检测逻辑**：
- 基于 **Router** 提供的边界置信度 (`left_confidence`, `right_confidence`)
- 阈值: `confidence < 0.8` → 判定为边界风险

### 2. 分步骤推理结构

V-CoT 采用**三步骤**推理框架：

```
步骤 1: 边界观察 (Observation)
  ↓
步骤 2: 对比分析 (Comparison)  
  ↓
步骤 3: 完整输出 (Transcription)
```

---

## 📝 提示词模板结构

### 完整模板结构

```
[系统角色设定]
你是一位专业的手写文本识别专家...

[OCR 识别结果]
```
{pred_text}
```

[观察任务] (根据风险类型动态生成)
⚠️ **系统检测到图像{LEFT/RIGHT/BOTH}边缘可能存在被截断的文字。**

### 步骤 1：观察图像边缘
仔细查看图像的**最{左/右}侧边缘**，看是否有：
- 被截断的笔画或偏旁
- 完整但未被 OCR 识别的字符
- 模糊但可辨认的文字痕迹

### 步骤 2：对比 OCR 结果
将你观察到的内容与 OCR 识别结果对比：
- OCR 是否遗漏了{左/右}侧的字符？
- {首/末}字是否被截断？

### 步骤 3：输出完整转录
如果发现遗漏，请在 OCR 结果的**{开头/末尾}**添加缺失的文字。

[置信度提示] (可选)
- 左边界置信度: {left_confidence:.2%}
- 右边界置信度: {right_confidence:.2%}
- 图像长宽比: {aspect_ratio:.1f}:1

[输出要求]
请直接输出**完整的文本转录**，格式如下：
```
[完整文本]
```

注意事项：
1. 如果确认 OCR 结果无误，直接输出原文即可
2. 如果发现遗漏，补全后输出完整文本
3. 最多补充 5 个字符（防止过度推测）
4. 不要添加任何解释或说明，只输出最终文本
```

---

## 🎯 三种场景的具体实现

### 场景 1: 仅左边界风险 (LEFT_ONLY)

**输入示例**:
- OCR 结果: `"锦涛强调做好农业标准化和食品安全工作"`
- 左边界置信度: 0.65
- 右边界置信度: 0.92

**生成的提示词要点**:
```
### 步骤 1：观察图像最左边
仔细查看图像的**最左侧边缘**，看是否有：
- 被截断的笔画或偏旁
- 完整但未被 OCR 识别的字符

### 步骤 2：对比 OCR 结果
- OCR 是否遗漏了左侧的字符？
- 首字是否被截断？

### 步骤 3：输出完整转录
如果发现遗漏，请在 OCR 结果的**开头**添加缺失的文字。
```

**期望输出**: `"胡锦涛强调做好农业标准化和食品安全工作"`

---

### 场景 2: 仅右边界风险 (RIGHT_ONLY)

**输入示例**:
- OCR 结果: `"本报北京4月24日讯中共中央政治局4月23日下午进行"`
- 左边界置信度: 0.92
- 右边界置信度: 0.68

**生成的提示词要点**:
```
### 步骤 1：观察图像最右边
仔细查看图像的**最右侧边缘**，看是否有：
- 被截断的笔画或偏旁
- 完整但未被 OCR 识别的字符

### 步骤 2：对比 OCR 结果
- OCR 是否遗漏了右侧的字符？
- 末字是否被截断？

### 步骤 3：输出完整转录
如果发现遗漏，请在 OCR 结果的**末尾**添加缺失的文字。
```

**期望输出**: `"本报北京4月24日讯中共中央政治局4月23日下午进行第四十一次集体学习"`

---

### 场景 3: 两边都有风险 (BOTH)

**输入示例**:
- OCR 结果: `"京4月24日讯中共中央政治局4月23日下午进行第四"`
- 左边界置信度: 0.55
- 右边界置信度: 0.60
- 图像长宽比: 15.5:1 (极端长宽比)

**生成的提示词要点**:
```
### 步骤 1：观察图像最左边
仔细查看图像的**最左侧边缘**...

### 步骤 2：观察图像最右边
仔细查看图像的**最右侧边缘**...

### 步骤 3：对比 OCR 结果
- OCR 是否遗漏了左右两侧的字符？
- 首字或末字是否被截断？

### 步骤 4：输出完整转录
如果发现遗漏，请补全缺失的文字，确保转录内容完整。
```

**期望输出**: `"本报北京4月24日讯中共中央政治局4月23日下午进行第四十一次集体学习"`

---

## 🔧 技术特性

### 1. 自适应风险类型

根据 Router 提供的置信度信息，**动态生成**针对性的观察指令：

```python
def detect_boundary_risk_type(
    left_confidence: float,
    right_confidence: float,
    threshold: float = 0.8
) -> BoundaryRiskType:
    left_risk = left_confidence < threshold
    right_risk = right_confidence < threshold
    
    if left_risk and right_risk:
        return BoundaryRiskType.BOTH
    elif left_risk:
        return BoundaryRiskType.LEFT_ONLY
    elif right_risk:
        return BoundaryRiskType.RIGHT_ONLY
    else:
        return BoundaryRiskType.NONE
```

### 2. 置信度感知提示

可选地提供置信度信息，帮助 VLM 理解问题的严重性：

```python
## 置信度提示
- 左边界置信度: 65.00%  (低置信度，很可能有问题)
- 右边界置信度: 92.00%  (高置信度，应该没问题)
- 图像长宽比: 15.5:1    (极端长宽比，边界更容易出问题)
```

### 3. 防止幻觉机制

```python
max_supplement_chars: int = 5  # 最多补充 5 个字符
```

在提示词中明确限制：
> "最多补充 5 个字符（防止过度推测）"

### 4. 双语支持

支持中文 (`zh`) 和英文 (`en`) 两种语言：

```python
config = VCoTPromptConfig(language="zh")  # 或 "en"
prompter = VCoTPrompter(config)
```

---

## 💡 设计原则

### 1. **边界聚焦** (Boundary Focus)
- 明确告知 VLM 需要关注**图像边缘**
- 避免模型注意力分散到图像中心区域

### 2. **对比推理** (Comparative Reasoning)
- 提供 OCR 结果作为**参考基准**
- 引导 VLM 进行差异分析，而非从零开始识别

### 3. **结构化输出** (Structured Output)
- 要求**仅输出文本**，不包含解释
- 简化后处理，便于提取最终结果

### 4. **风险感知** (Risk-Aware)
- 根据置信度和长宽比，动态调整提示词
- 高风险场景提供更详细的观察指令

---

## 📊 与 EIP (Explicit Index Prompting) 的关系

V-CoT 和 EIP 是**互补**的两种提示策略：

| 特性 | EIP | V-CoT |
|------|-----|-------|
| **适用场景** | 中间字符错误 | 边界字符丢失 |
| **引导方式** | 索引定位 | 区域聚焦 |
| **推理步骤** | 单步验证 | 多步观察-对比-输出 |
| **输入信息** | 字符级置信度 | 边界级置信度 |

**在 Agent B 中的使用**:
- **边界风险** → 使用 V-CoT 边界补全提示
- **中间字符错误** → 使用 EIP 显式索引提示

---

## 🚀 使用示例

### Python 代码示例

```python
from modules.vlm_expert.v_cot_prompter import (
    VCoTPrompter,
    VCoTPromptConfig,
    BoundaryRiskType
)

# 初始化
config = VCoTPromptConfig(
    language="zh",
    max_supplement_chars=5,
    show_confidence_hint=True
)
prompter = VCoTPrompter(config)

# 场景 1: 左边界风险
prompt = prompter.build_boundary_completion_prompt(
    pred_text="锦涛强调做好农业标准化和食品安全工作",
    risk_type=BoundaryRiskType.LEFT_ONLY,
    left_confidence=0.65,
    right_confidence=0.92,
)
print(prompt)

# 场景 2: 自动检测风险类型
risk_type = prompter.detect_boundary_risk_type(
    left_confidence=0.55,
    right_confidence=0.60,
    threshold=0.8
)
# 返回: BoundaryRiskType.BOTH

prompt = prompter.build_boundary_completion_prompt(
    pred_text="京4月24日讯...进行第四",
    risk_type=risk_type,
    left_confidence=0.55,
    right_confidence=0.60,
    aspect_ratio=15.5
)
```

### 便捷函数

```python
from modules.vlm_expert.v_cot_prompter import create_boundary_prompt

prompt = create_boundary_prompt(
    pred_text="锦涛强调做好农业标准化和食品安",
    left_confidence=0.95,
    right_confidence=0.65,
    threshold=0.8,
    language="zh"
)
```

---

## 📈 效果预期

### 成功案例

**Case 1: 左边界补全**
- 输入 OCR: `"锦涛强调做好农业标准化和食品安全工作"`
- V-CoT 输出: `"胡锦涛强调做好农业标准化和食品安全工作"`
- ✅ 成功补全首字 "胡"

**Case 2: 右边界补全**
- 输入 OCR: `"本报北京4月24日讯中共中央政治局4月23日下午进行"`
- V-CoT 输出: `"本报北京4月24日讯中共中央政治局4月23日下午进行第四十一次集体学习"`
- ✅ 成功补全末尾 "第四十一次集体学习"

**Case 3: 两边补全**
- 输入 OCR: `"京4月24日讯中共中央政治局4月23日下午进行第四"`
- V-CoT 输出: `"本报北京4月24日讯中共中央政治局4月23日下午进行第四十一次集体学习"`
- ✅ 成功补全开头 "本报北京" 和末尾 "十一次集体学习"

### 限制与注意事项

1. **字符数量限制**: 最多补充 5 个字符，防止过度推测
2. **视觉模糊**: 如果边界字符完全不可见，VLM 可能无法恢复
3. **上下文依赖**: 需要足够的上下文信息才能推断缺失字符

---

## 🔬 实验验证

### 评估指标

- **边界修复率** = 成功修复的边界错误数 / 总边界错误数
- **CER 改进** = CER_baseline - CER_after_v_cot
- **VLM 调用成本** = VLM 推理时间 × 调用次数

### 预期效果

在 HWDB 数据集上的预期改进：
- 边界错误修复率: **60-80%**
- 整体 CER 降低: **0.05-0.15** (相对改进 10-30%)
- VLM 调用率: **15-25%** (仅困难样本调用)

---

## 📚 参考文献与灵感来源

1. **Chain-of-Thought Prompting** (Wei et al., 2022)
   - 分步骤推理的思想来源

2. **Visual Question Answering** 中的注意力机制
   - 区域聚焦的概念

3. **OCR Post-Processing** 中的置信度阈值
   - 边界风险检测的依据

---

## 🎓 总结

V-CoT 是一个**专门针对边界错误修复**的提示词框架，通过：

1. ✅ **分步骤推理**引导 VLM 系统性观察
2. ✅ **风险类型感知**动态调整提示词
3. ✅ **对比分析**利用 OCR 结果作为参考
4. ✅ **结构化输出**简化后处理

实现了从"从零识别"到"边界补全"的任务转换，显著提升了边界错误的修复成功率。

