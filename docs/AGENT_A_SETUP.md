# Agent A (PP-OCRv5) 配置指南

## 🎉 L2W1 独立部署版

L2W1 现已支持 **完全独立部署**，无需下载完整的 PaddleOCR 代码库！

我们已将所需的 PaddleOCR 模块精简并集成到 L2W1 项目中：

```
L2W1/
├── tools/                      # 精简版推理工具
│   └── infer/
│       └── utility.py          # 推理工具函数
├── ppocr/                      # 精简版 PPOCR 模块
│   ├── postprocess/
│   │   ├── __init__.py
│   │   └── rec_postprocess.py  # CTCLabelDecode
│   └── utils/
│       ├── logging.py          # 日志模块
│       ├── utility.py          # 工具函数
│       └── ppocr_keys_v1.txt   # 字符字典
└── models/
    └── agent_a_ppocr/          # 放置模型文件
```

---

## 快速配置步骤

### 步骤 1: 解压模型

你已经下载了 `PP-OCRv5_server_rec_infer.tar`，现在解压它：

```bash
cd L2W1/models/agent_a_ppocr
tar -xvf PP-OCRv5_server_rec_infer.tar

# 解压后应该有以下文件:
# - inference.pdmodel (或 model.pdmodel)
# - inference.pdiparams (或 model.pdiparams)
# - inference.yml (可选)
```

### 步骤 2: 运行数据流水线

```bash
cd L2W1

python scripts/data_pipeline.py \
    --data_dir ./data/raw/viscgec \
    --split train \
    --rec_model_dir ./models/agent_a_ppocr \
    --max_cer 0.3
```

### 步骤 3: 验证成功

**成功标志**：

```
[INFO] Agent A (TextRecognizerWithLogits) 初始化成功
[1/4] 加载数据集...
...
```

**不再出现**：

```
[WARNING] 无法初始化 Agent A: No module named 'tools'
```

---

## 模型目录结构

解压后，`models/agent_a_ppocr/` 目录应包含：

```
models/agent_a_ppocr/
├── inference.pdmodel      # 或 model.pdmodel
├── inference.pdiparams    # 或 model.pdiparams
└── inference.yml          # 可选配置文件
```

如果模型文件在子目录中，需要指定完整路径：

```bash
--rec_model_dir ./models/agent_a_ppocr/PP-OCRv5_server_rec_infer
```

---

## 常见问题

### Q1: 仍然看到 "No module named 'tools'" 错误？

**检查**：

1. 确保 `L2W1/tools/infer/utility.py` 存在
2. 确保从 L2W1 目录运行命令（`cd L2W1`）
3. 检查 Python 路径：`python -c "import tools.infer.utility; print('OK')"`

### Q2: 模型文件找不到？

**检查模型路径**：

```bash
ls -la models/agent_a_ppocr/
# 应该看到 .pdmodel 和 .pdiparams 文件
```

**如果模型在子目录**：

```bash
--rec_model_dir ./models/agent_a_ppocr/子目录名
```

### Q3: 字符字典找不到？

**默认路径**：`./ppocr/utils/ppocr_keys_v1.txt`

**如果需要自定义**：

```bash
--rec_char_dict_path ./path/to/your/dict.txt
```

---

## 技术说明

### 精简版模块说明

| 模块                                   | 功能               | 来源           |
| -------------------------------------- | ------------------ | -------------- |
| `tools/infer/utility.py`               | 创建 Paddle 预测器 | PaddleOCR 精简 |
| `ppocr/postprocess/rec_postprocess.py` | CTC 解码           | PaddleOCR 精简 |
| `ppocr/utils/logging.py`               | 日志记录           | PaddleOCR 精简 |
| `ppocr/utils/utility.py`               | 图像加载工具       | PaddleOCR 精简 |

### 与完整 PaddleOCR 的区别

- ✅ **保留**: 文本识别 (Rec) 推理核心功能
- ✅ **保留**: CTC 解码和 Logits 拦截
- ❌ **移除**: 文本检测 (Det)
- ❌ **移除**: 表格识别、版面分析
- ❌ **移除**: VQA、Layout 相关功能

这使得 L2W1 可以作为独立项目部署，无需完整的 PaddleOCR 代码库。
