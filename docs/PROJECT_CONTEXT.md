# Project Context: L2W1 v5.0 - Holistic Line-level Handwriting Correction

## 1. 研究背景与核心愿景 (Research Vision)

[cite_start]本项目旨在解决中文手写识别 (HCTR) 的“不可能三角”：速度、精度与鲁棒性 [cite: 63, 65]。
[cite_start]传统方案因“字符切割”破坏拓扑结构，且纯文本纠错易产生语义幻觉 [cite: 71, 77]。
[cite_start]L2W1 v5.0 采用**“认知仿生 (Cognitive Bionics)”**架构，模拟人类“快速扫描 + 逻辑回视”的阅读过程 [cite: 165, 166]。

## 2. 系统架构 (Architecture Topology)

[cite_start]系统由三个核心智能体组成，通过异步流水线协作 [cite: 170, 171]：

### Phase 1: 宿主扫描与诊断 (Host Scan & Diagnose)

- [cite_start]**Agent A (PP-OCRv5-Rec)**: 作为 System 1（快思考），执行全量行级扫描 [cite: 7, 167]。
- [cite_start]**Router (仲裁器)**: 纯计算模块，执行双重门控（Dual Gating） [cite: 175, 193]：
  1. [cite_start]**视觉不确定性 ($U_{vis}$)**: 计算 Logits 的 Shannon Entropy [cite: 196]：
     $$H = -\sum (P \cdot \log(P + \epsilon))$$
  2. [cite_start]**语义困惑度 ($U_{sem}$)**: 使用 Qwen2.5-0.5B 计算 Perplexity (PPL) [cite: 197]。
- [cite_start]**分流机制**: 仅 ~20-30% 的困难样本（Hard Samples）会触发 Agent B [cite: 178]。

### Phase 2: 显式引导的视觉重写 (Guided Rewriting)

- [cite_start]**Agent B (Qwen2.5-VL-3B)**: 作为 System 2（慢思考），具备动态分辨率 (Naive Res) 特性 [cite: 7, 168, 203]。
- [cite_start]**核心创新 (EIP)**: 采用显式索引提示 (Explicit Index Prompting)，告知 Agent B “第 K 个字存疑”，将生成任务转化为验证式纠错 [cite: 205]。

## 3. 技术实施规范 (Implementation Specs)

### 3.1 Agent A 源码拦截 (Logits Hooking)

- [cite_start]**目标**: 修改 `PaddleOCR/tools/infer/predict_rec.py` [cite: 11]。
- [cite_start]**逻辑**: 在 `self.predictor.run()` 之后拦截原始 Logits Tensor，形状为 $[Batch, Seq\_Len, Vocab\_Size]$ [cite: 15, 27]。
- [cite_start]**位置**: 代码应集成在 `modules/paddle_engine/predict_rec_modified.py` [cite: 16]。

### 3.2 零切割数据工程 (Zero-Cropping Pipeline)

- [cite_start]**原则**: 严禁字符级切割，直接输入原始长宽比的行图像（支持 20:1 等极端比例） [cite: 221, 225]。
- [cite_start]**数据挖掘**: 通过对比 Agent A 预测结果与真值，利用 Levenshtein 距离自动化构建“错题集” [cite: 232, 234]。

### 3.3 评估指标 (Evaluation Metrics)

- [cite_start]**CER (Character Error Rate)**: 基础精度指标 [cite: 123]。
- [cite_start]**OCR-R (Over-Correction Rate)**: **核心创新指标**。衡量模型将正确识别结果改错的比例，用于量化“幻觉” [cite: 127, 130]：
  $$OCR\text{-}R = \frac{\text{Count}(\text{Agent A Correct} \rightarrow \text{System Wrong})}{\text{Total Correct Chars in Agent A}}$$

## 4. 目录开发约定

- `modules/`: 存放模型核心逻辑。
- `scripts/`: 存放实验流水线，需支持命令行参数驱动。
- `configs/`: 统一管理 `entropy_threshold` 和 `ppl_threshold`。
