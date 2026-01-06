# Implementation Spec: Module 3.3 - Agent B Supervised Fine-Tuning (SFT)

## 1. 训练目标

教会 Agent B 听懂显式索引指令，并结合行级视觉特征修正 Agent A 的识别错误，同时抑制语义幻觉。

## 2. 核心训练策略

- [cite_start]**技术路径**: QLoRA (4-bit LoRA) [cite: 117]。
- **目标模块**: Qwen2.5-VL 的 `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`。
- **数据策略**:
  - [cite_start]使用 `agent_b_train.jsonl` 中的 EIP 提示对 [cite: 248, 249]。
  - 混合 10%-20% 的“负样本”（即识别结果正确，Prompt 告知无需修正），以维持模型的稳定性。

## 3. 损失函数与优化 (Loss & Optimization)

- **Loss**: Standard Cross-Entropy on completion tokens.
- **Optimizer**: AdamW with cosine learning rate schedule.
- [cite_start]**Batch Size**: 累积梯度实现 Effective BS = 128 [cite: 58]。

## 4. 验证指标 (Validation Metrics)

- [cite_start]**OCR-R (过度纠错率)**: 必须在验证集上实时监控，防止模型“瞎改” 。
- [cite_start]**CER (字符错误率)**: 验证纠错后的精度收益 [cite: 123]。
