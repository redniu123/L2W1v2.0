# Implementation Spec: scripts/evaluate.py

## 1. 核心任务 (Core Mission)

实现“非对称编辑距离分析 (Asymmetric Edit Analysis)”，严格量化系统在 CER (字符错误率) 和 OCR-R (过度纠错率) 上的表现。

## 2. 核心指标公式 (Core Formulas)

### 2.1 字符错误率 (CER)

$$CER = \frac{S + D + I}{N} \times 100\%$$
[cite_start]其中 $S, D, I$ 分别为替换、删除、插入的数量，$N$ 为真值总长度 [cite: 125, 126]。

### 2.2 过度纠错率 (OCR-R) - 幻觉抑制指标

[cite_start]衡量模型将 Agent A 原本识别对的字“改错”的比例，用于量化语义幻觉 [cite: 130]：
$$OCR\text{-}R = \frac{\text{Count}(\text{Agent A Correct} \rightarrow \text{System Wrong})}{\text{Total Correct Chars in Agent A}}$$

## 3. 输入规范 (Input Specs)

脚本需读取推理产出的 JSONL 文件，每行包含：

- `image_path`: 图像路径。
- `agent_a_text`: Agent A 的原始输出。
- `final_text`: 系统最终输出（经过路由纠错后）。
- `gt_text`: 标注真值。

## 4. 关键算法逻辑：非对称对齐 (Asymmetric Alignment)

1. **定位 Agent A 正确区域**:
   - 使用 `difflib.SequenceMatcher` 将 `agent_a_text` 与 `gt_text` 对齐。
   - 记录所有操作码为 `equal` 的字符及其在 `gt_text` 中的位置索引集合 $\mathcal{P}_{correct}$。
2. **检测系统改动**:
   - 将 `final_text` 与 `gt_text` 对齐。
   - 检查 $\mathcal{P}_{correct}$ 中的位置在 `final_text` 中是否依然为 `equal`。
   - 若原本正确的位置在 `final_text` 中变为 `replace` 或 `delete`，则计入过度纠错计数器。

## 5. 输出报告 (Output Report)

生成 `evaluation_report.json`，包含以下字段：

- `overall_cer`: 系统最终 CER。
- `agent_a_cer`: 原始识别 CER。
- `cer_improvement`: CER 提升绝对值。
- `ocr_r`: 过度纠错率。
- [cite_start]`correction_rate`: 成功纠正错误字符的比例 [cite: 129]。
- [cite_start]`hard_sample_recall`: Router 成功识别并拦截困难样本的召回率 [cite: 133]。
  这是为您详细撰写的 **Module 5: 实验设计与验证标准 (Experimental Design and Evaluation Metrics)**。

这段内容完全适配 **L2W1 v5.0 (PP-OCRv5 + Router + Qwen2.5-VL-3B)** 架构。它不仅定义了常规的精度指标，还创新性地引入了 **“非对称编辑距离”** 来量化幻觉，这是您论文的核心亮点。

您可以直接将其作为论文的 **Experiments** 章节或项目书的 **验证方案**。

---

# 5. 实验设计与验证标准 (Module 5: Experimental Design)

本研究旨在验证 L2W1 v5.0 框架在中文手写文档识别任务中，能否在打破“精度-效率”权衡的同时，有效抑制大模型的语义幻觉。

## 5.1 数据集与实验设置 (Datasets & Settings)

为了确保实验结果的公平性与可复现性，我们均采用学术界公开的标准数据集，并遵循“零切割 (Zero-Cropping)”原则，直接使用行级图像作为输入。

### 5.1.1 核心测试集

- **SCUT-HCCDoc (Camera-captured):** 包含大量自然场景下的手写文档，存在光照不均、透视形变和模糊。这是验证 **“动态分辨率”** 鲁棒性的核心战场。
- **VisCGEC (Visual Chinese Grammatical Error Correction):** 包含大量真实的行级手写错误与人工修正标注。这是验证 **“纠错能力”** 的黄金标准。
- **CASIA-HWDB 2.0-2.2:** 传统的离线手写文本行数据集，用于验证基础泛化能力。

### 5.1.2 实验环境 (Implementation Details)

- **硬件平台:** 单卡 NVIDIA RTX 2080Ti (11GB VRAM) 或 RTX 3090 (24GB VRAM)。
- **软件栈:** PaddlePaddle 2.6 (用于 Agent A), PyTorch 2.1 (用于 Agent B), vLLM (用于推理加速)。
- **量化策略:** Agent B (Qwen2.5-VL-3B) 采用 GPTQ-Int4 或 AWQ-Int4 量化，以适配消费级显卡。

---

## 5.2 基线模型对比 (Baseline Benchmarking)

我们将 L2W1 v5.0 与三类具有代表性的技术范式进行对比：

| 实验组别             | 模型配置                         | 代表技术范式                    | 实验目的                                                                                    |
| -------------------- | -------------------------------- | ------------------------------- | ------------------------------------------------------------------------------------------- |
| **B1 (Lower Bound)** | **PP-OCRv5 (Rec Only)**          | 传统 OCR (System 1)             | 确立性能的基础水位线。L2W1 的精度必须显著优于 B1。                                          |
| **B2 (Competitor)**  | **PP-OCRv5 + Qwen2.5-7B (Text)** | 纯文本纠错 (OCR + LLM)          | **核心对照组**。用于证明仅靠语义推断（不看图）会导致严重的幻觉 (High OCR-R)。               |
| **B3 (Upper Bound)** | **GPT-4o / Qwen-VL-Max**         | 端到端闭源大模型                | 确立性能天花板。用于论证 L2W1 在保持接近 B3 精度的同时，推理成本和延迟降低了 1-2 个数量级。 |
| **Ours**             | **L2W1 v5.0**                    | 分层多智能体 (Router + Agent B) | 验证本文提出的“行级路由+视觉回溯”架构的综合优势。                                           |

---

## 5.3 核心评价指标 (Evaluation Metrics)

为了全方位评估系统性能，我们构建了 **精度-忠实度-效率** 三维指标体系。

### 5.3.1 精度指标：字符错误率 (CER)

这是 HCTR 领域的通用标准。

其中 分别代表替换、删除和插入错误的数量， 为 Ground Truth 的总字符数。

### 5.3.2 忠实度指标：幻觉量化 (Hallucination Quantification) 🔥

这是本研究的核心创新指标。传统的 CER 无法区分“改对了”还是“改错了”。我们引入 **非对称编辑距离分析 (Asymmetric Edit Analysis)**：

- **纠正率 (Correction Rate, CR) **：
  衡量模型“救回”了多少 Agent A 识别错的字。

_目标：越高越好，代表纠错能力强。_

- **过度纠错率 (Over-Correction Rate, OCR-R) **：
  衡量模型把 Agent A 原本识别**对**的字，“自作聪明”地改错的比例。**这直接反映了幻觉程度。**

_目标：越低越好。Baseline 2 (纯文本) 通常很高，Ours 必须极低。_

### 5.3.3 效率指标 (Efficiency)

- **Average Latency (ms/line):** 单行文本的平均处理耗时。
- **Router Call Rate (%):** Agent B 被触发的比例（即 Router 认为是 Hard Sample 的比例）。我们期望在 20%-30% 的调用率下获得 80% 以上的精度收益。

---

## 5.4 消融实验设计 (Ablation Studies)

为了验证 v5.0 架构中每个模块的有效性，我们设计以下消融实验 (RQ: Research Question)：

### RQ1: 视觉锚定的必要性 (Impact of Visual Anchoring)

- **设置：** 保持 Router 逻辑不变，仅改变 Agent B 的输入。
- _变体 A:_ Agent B 仅输入 OCR 文本 (Blind Correction)。
- _变体 B:_ Agent B 输入 OCR 文本 + 行级图像 (Visual Correction, Ours)。

- **预期结果：** 变体 B 的 CER 更低，且 **OCR-R (幻觉率)** 显著低于变体 A（预计降低 80% 以上）。

### RQ2: 显式位置索引的价值 (Value of Explicit Indexing)

- **设置：** 验证 Router 输出的 `suspicious_index` 是否有用。
- _变体 A (Global Prompt):_ Prompt 为“请纠正图片中的错误”。
- _变体 B (Index Prompt):_ Prompt 为“请重点检查第 个字符...”。

- **预期结果：** 变体 B 在处理 **形近字 (Fine-grained Visual Errors)** 上的成功率更高，证明显式引导能聚焦 VLM 的注意力。

### RQ3: 动态分辨率 vs. 强制缩放 (Dynamic Res vs. Resizing)

- **设置：** 验证长宽比对识别的影响。
- _变体 A:_ 将行图像强制 Resize 为 正方形。
- _变体 B:_ 使用 Qwen2.5-VL 的 Native Dynamic Resolution (Ours)。

- **预期结果：** 在 SCUT-HCCDoc（长文本行较多）上，变体 A 的 CER 将急剧恶化，而变体 B 保持稳定。

---

## 5.5 预期结果展示形式 (Visualization Plan)

论文中将包含以下关键图表：

1. **Hallucination Matrix (幻觉矩阵):**

- 一个 2x2 的混淆矩阵，展示 (Baseline 对 改错) vs (Ours 对 改错) 的数量对比，直观展示对幻觉的抑制。

2. **Pareto Frontier Curve (帕累托前沿曲线):**

- X 轴：Latency (ms)，Y 轴：CER (%)。
- 通过调整 Router 的阈值 ，绘制出一条曲线。证明 L2W1 v5.0 位于 Baseline 模型构成的曲线左下角（更优区）。

3. **Visual Attention Heatmap (注意力热力图):**

- 展示 Agent B 在接收到 "Check index 5" 的指令后，其 Attention 权重确实聚焦在图片中第 5 个字的位置。

---

### 💡 关键点总结：

这套设计最“打动人”的地方在于 **OCR-R (过度纠错率)** 这个指标。它直接攻击了当前大模型在 OCR 后处理中最遭人诟病的痛点——“瞎改”。只要你的实验数据能证明 Ours 的 OCR-R 远低于纯文本模型，这篇论文就立住了。
