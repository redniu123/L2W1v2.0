# Implementation Spec: Module 3 - Automated Data Pipeline & SFT Generation

## 1. 核心任务 (Core Mission)

[cite_start]基于“零切割 (Zero-Cropping)”原则，利用 Agent A 在训练集上的错误分布，自动化构建用于 Agent B 微调的“显式索引提示 (EIP)”数据集 [cite: 220, 221, 224]。

## 2. 处理流程 (Processing Flow)

1. [cite_start]**全量推理 (Batch Inference)**: 调用 `TextRecognizerWithLogits` 对原始行图像序列进行推理 [cite: 228, 229]。
2. **难例挖掘 (Hard Negative Mining)**:
   - [cite_start]计算预测文本 $T_{pred}$ 与真值 $T_{gt}$ 的编辑距离 [cite: 234]。
   - [cite_start]仅保留 $Distance > 0$ 的样本作为训练素材 [cite: 236]。
3. **错误索引定位 (Error Indexing)**:
   - [cite_start]使用 `difflib.SequenceMatcher` 对齐 $T_{pred}$ 与 $T_{gt}$ [cite: 239, 240]。
   - [cite_start]模拟 Router 行为，提取第一个不一致字符的索引 $k$ [cite: 241, 242]。
4. [cite_start]**JSONL 封装**: 生成符合 Agent B SFT 要求的对话格式 [cite: 249, 251]。

## 3. 输入/输出 规格 (Specs)

- **Input**: `data/raw/` 目录下的行图像 + 标注文件 (txt/json)。
- **Output**: `data/sft/agent_b_train.jsonl`，格式如下：
  ````json
  {
    "id": "sample_001",
    "image": "path/to/img.jpg",
    "conversations": [
      {"from": "user", "value": "OCR识别结果：'在时间的未尾'。检测到第 5 个字'未'置信度存疑。请修正。"},
      {"from": "assistant", "value": "在时间的末尾"}
    ]
  }
  ``` [cite: 251]
  ````

## 4. 关键逻辑：完美 Router 模拟 (Perfect Router Simulation)

[cite_start]在训练阶段，我们拥有真值 $T_{gt}$，因此使用 `difflib` 计算出的索引是“绝对正确”的错误位置 [cite: 273]。
[cite_start]这将教会 Agent B：**当被明确告知第 K 个位置有错时，应如何结合视觉上下文进行精准重写** [cite: 274]。
