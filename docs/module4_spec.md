# Implementation Spec: Module 4 - Model Selection and Source Integration

## 1. 架构拓扑 (Architecture Topology)

[cite_start]本模块实现“宿主-插件 (Host-Plugin)”架构 。

- [cite_start]**Agent A (Host):** PP-OCRv5-Rec (Server Model)，负责全量扫描并暴露 Logits [cite: 7]。
- [cite_start]**Router (Gatekeeper):** 基于 Qwen2.5-0.5B (Semantic PPL) 和视觉熵 (Visual Entropy) 的轻量级仲裁模块 [cite: 7, 31]。
- [cite_start]**Agent B (Expert):** Qwen2.5-VL-3B (Int4)，执行显式索引引导下的行级视觉重写 [cite: 7, 42]。

## 2. 核心公式 (Core Formulas)

### 2.1 视觉不确定性 (Visual Entropy)

[cite_start]对于 Agent A 输出的原始 Logits $L \in \mathbb{R}^{B \times N \times V}$ [cite: 27, 33]：

1. **Softmax 归一化**:
   [cite_start]$$P = \text{softmax}(L, \text{dim}=-1)$$ [cite: 35]
2. **Shannon Entropy 计算**:
   [cite_start]$$H = -\sum (P \cdot \log(P + \epsilon))$$ [cite: 36]
3. **CTC 时间步对齐**:
   [cite_start]利用 CTC Decoder 逻辑，映射非空（Non-blank）且发生字符变更的时间步 $t$ [cite: 37][cite_start]，得到文本行对应的熵序列 $[h_1, h_2, ..., h_M]$ [cite: 38]。

### 2.2 语义困惑度 (Semantic PPL)

[cite_start]$$PPL = \exp\left(\frac{1}{M}\sum_{i=1}^{M} \text{CrossEntropy}(T_{ocr} \mid \text{Qwen2.5-0.5B})\right)$$ [cite: 41]

## 3. Tensor 形状与规格 (Tensor Shapes & Specs)

| 组件        | 数据对象        | Tensor Shape            | 说明                                                                             |
| :---------- | :-------------- | :---------------------- | :------------------------------------------------------------------------------- |
| **Agent A** | `raw_logits`    | $[B, N, V]$             | [cite_start]$B$: Batch, $N$: Seq Len (典型值 80), $V$: Vocab Size [cite: 27, 33] |
| **Router**  | `entropy_seq`   | $[B, M]$                | [cite_start]$M$: 识别出的文本长度 [cite: 38]                                     |
| **Agent B** | `image_patches` | $[H/14 \times W/14, D]$ | [cite_start]基于动态分辨率 (Naive Res) 的编码 Patch [cite: 181, 203]             |

## 4. 源码外科手术 (Surgical Hooking Logic)

[cite_start]**目标文件**: `PaddleOCR/tools/infer/predict_rec.py` [cite: 11]
**修改逻辑**:

1. [cite_start]**定位**: 找到 `self.predictor.run()` 调用点 [cite: 24]。
2. [cite_start]**拦截 (Hook)**: 在 `CTC Decode` 之前，对输出 Tensor 进行 `deep_copy` 并转为 Numpy 数组 [cite: 19, 26]。
3. [cite_start]**重构返回值**: 将函数返回类型从 `text, score` 修改为 `dict: {'text': rec_text, 'conf': rec_score, 'logits': raw_logits}` [cite: 29]。

## 5. Agent B 提示工程策略 (EIP Strategy)

[cite_start]**Explicit Index Prompting (EIP)**[cite: 49, 205]:

```python
prompt = f"""这是一张中文手写文档的行图片。
OCR识别结果："{ocr_text}"
系统检测到第 {suspicious_index + 1} 个字符（当前识别为“{suspicious_char}”）存在视觉不确定性。
任务：
1. 请结合整行的视觉上下文，重新审视该位置的字迹。
2. 如果原识别正确，请保持不变；如果错误，请修正。
3. 输出修正后的整行文本。""" # [cite: 51]
```
