# Implementation Spec: Agent B - Dynamic Visual Expert

## 1. 核心技术栈 (Tech Stack)

- [cite_start]**Model**: Qwen2.5-VL-3B-Instruct[cite: 201].
- [cite_start]**Quantization**: 4-bit (via bitsandbytes) 以适配 11GB 显存 (RTX 2080Ti)[cite: 54, 117].
- [cite_start]**Acceleration**: Flash Attention 2 (可选).

## 2. 核心特性配置 (Core Features)

### 2.1 动态分辨率 (Native Resolution)

[cite_start]严禁将长条状的手写行图像 Resize 为正方形 [cite: 48, 148, 203]。
[cite_start]通过 `AutoProcessor` 配置 Patch 序列，保留原始长宽比（最高支持 20:1） [cite: 203, 225]。

- [cite_start]`min_pixels`: $256 \times 28 \times 28$[cite: 46].
- [cite_start]`max_pixels`: $1280 \times 28 \times 28$[cite: 47].

### 2.2 显式索引提示 (Explicit Index Prompting, EIP)

[cite_start]将任务从“开放式 OCR”降维为“验证式纠错” [cite: 205]。
[cite_start]使用 Router 提供的 `suspicious_index` 构建结构化提示 [cite: 50, 51]。

## 3. Tensor 规格与输入 (Input Specs)

- [cite_start]**Image**: 原始行级图像[cite: 224].
- [cite_start]**Prompt**: 包含识别结果、错误位置索引及该位置字符的模版[cite: 51].
- [cite_start]**Output**: 修正后的完整字符序列 $S_{final}$[cite: 51, 90].

## 4. 关键函数伪代码

```python
def correct_line(image, ocr_text, suspicious_idx, suspicious_char):
    # 1. 构造 EIP Prompt
    prompt = f"OCR识别结果：'{ocr_text}'。第 {suspicious_idx+1} 个字'{suspicious_char}'存疑。请修正。"

    # 2. 动态分辨率编码 (Native Res)
    # 3. 4-bit 推理
    # 4. 返回修正文本
```
