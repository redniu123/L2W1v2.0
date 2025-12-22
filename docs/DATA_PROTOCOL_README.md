# L2W1 Master Data Protocol v2.0

本文档定义了 L2W1 项目中数据交换的标准格式。

## 核心原则

1. **suspicious_index 始终为 0-indexed**：系统内部所有逻辑统一使用 0-indexed
2. **嵌套结构**：按照 Agent A → Router → Agent B → Metadata 的逻辑分层
3. **物理隔离**：Train/Val/Test 数据集严格分离，使用独立的 JSONL 文件
4. **向后兼容**：评估模块同时支持嵌套格式和扁平格式

---

## 0. 物理存储结构 (v2.0 新增)

### 目录结构

```
data/raw/[dataset_name]/
├── images/                    # 图像文件目录
│   ├── train/                 # 训练集图像
│   │   ├── 001.png
│   │   └── ...
│   ├── val/                   # 验证集图像
│   └── test/                  # 测试集图像
├── train.jsonl                # 训练集元数据 (必需)
├── val.jsonl                  # 验证集元数据 (可选)
└── test.jsonl                 # 测试集元数据 (可选)
```

### JSONL 文件格式

每行一个 JSON 对象：

```jsonl
{"id": "viscgec_train_001", "image": "images/train/001.png", "gt_text": "真值文本", "source": "viscgec"}
{"id": "viscgec_train_002", "image": "images/train/002.png", "gt_text": "另一行文本", "source": "viscgec"}
```

### 加载示例

```bash
# 加载训练集
python scripts/data_pipeline.py --data_dir ./data/raw/viscgec --split train

# 加载测试集
python scripts/data_pipeline.py --data_dir ./data/raw/viscgec --split test

# 加载所有划分
python scripts/data_pipeline.py --data_dir ./data/raw/viscgec --split all
```

---

## 1. Pipeline 输出格式

`modules/pipeline.py` 中 `PipelineResult.to_dict()` 的输出格式：

```json
{
  "id": "viscgec_train_001",
  "image": "data/raw/viscgec/images/train/2016_8.png",
  "gt_text": "才能在瞬息万变的比赛中占据上风。",
  
  "agent_a": {
    "text": "才能瞬息万变的比赛中占据上风。",
    "confidence": 0.9245,
    "suspicious_index": 2,
    "suspicious_char": "瞬",
    "raw_logits_shape": [80, 6625]
  },
  
  "router": {
    "is_hard": true,
    "visual_entropy": 3.4567,
    "semantic_ppl": 125.4,
    "risk_level": "high",
    "decision": "route_to_agent_b"
  },
  
  "agent_b": {
    "text": "才能在瞬息万变的比赛中占据上风。",
    "is_corrected": true,
    "refinement_strategy": "explicit_indexing_prompt"
  },
  
  "metadata": {
    "source": "viscgec",
    "split": "test",                    // [v2.0 新增] train, val, test
    "difficulty": "hard",
    "error_type": "grammar_omission",
    "gt_char_len": 17,
    "processing_time_ms": 1240,
    "environment": "RTX2080Ti_4bit_quant"
  }
}
```

---

## 2. 数据集输入格式 (metadata.jsonl)

`scripts/data_pipeline.py` 支持的输入格式：

```jsonl
{"id": "viscgec_001", "image": "images/001.png", "gt_text": "真值文本", "source": "viscgec", "error_type": "similar_char"}
{"id": "viscgec_002", "image": "images/002.png", "gt_text": "另一行文本", "source": "viscgec", "error_type": "grammar_omission"}
```

### 字段说明

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `id` | string | 是 | 样本唯一标识符 |
| `image` | string | 是 | 图像路径（相对或绝对） |
| `gt_text` | string | 是 | 真值文本 (Ground Truth) |
| `source` | string | 否 | 数据来源 (viscgec, scut, casia) |
| `error_type` | string | 否 | 错误类型分类 |
| `difficulty` | string | 否 | 难度 (easy, normal, hard) |

### 支持的 error_type 值

- `similar_char`: 形近字替换 (如 "未" → "末")
- `grammar_omission`: 语法遗漏 (如 漏字)
- `extra_char`: 多余字符
- `missing_char`: 缺失字符
- `unknown`: 未知类型

---

## 3. SFT 训练数据格式

`scripts/data_pipeline.py` 生成的 SFT 数据格式：

```json
{
  "id": "viscgec_001",
  "image": "path/to/image.png",
  "gt_text": "真值文本",
  
  "agent_a": {
    "text": "预测文本",
    "suspicious_index": 2,
    "suspicious_char": "字"
  },
  
  "conversations": [
    {"from": "user", "value": "OCR识别结果为：'预测文本'。\n系统检测到第 3 个字..."},
    {"from": "assistant", "value": "真值文本"}
  ],
  
  "metadata": {
    "source": "viscgec",
    "error_type": "similar_char",
    "difficulty": "hard",
    "gt_char_len": 4
  }
}
```

---

## 4. 评估结果格式

`scripts/evaluate.py` 生成的评估报告格式：

```json
{
  "overall_cer": 0.0512,
  "agent_a_cer": 0.0834,
  "cer_improvement": 0.0322,
  "accuracy": 0.7823,
  
  "ocr_r": 0.0234,
  "correction_rate": 0.7156,
  
  "hard_sample_recall": 0.8912,
  "router_call_rate": 0.2345,
  
  "edit_operations": {
    "substitutions": 156,
    "deletions": 23,
    "insertions": 45
  },
  
  "sample_stats": {
    "total": 1000,
    "exact_match": 782,
    "overcorrected": 23,
    "corrected": 156
  },
  
  "error_type_breakdown": {
    "similar_char": {"count": 450, "cer": 0.0423, "ocr_r": 0.0189, "correction_rate": 0.7823},
    "grammar_omission": {"count": 230, "cer": 0.0612, "ocr_r": 0.0312, "correction_rate": 0.6912},
    "unknown": {"count": 320, "cer": 0.0534, "ocr_r": 0.0245, "correction_rate": 0.7012}
  },
  
  "timestamp": "2025-12-22T10:30:00",
  "version": "L2W1 v5.0"
}
```

---

## 5. 索引约定

### suspicious_index

- **内部存储**: 始终为 **0-indexed** (第一个字符索引为 0)
- **显示给用户**: 转换为 **1-indexed** (使用 `to_display_index()`)

```python
# 内部逻辑
suspicious_index = 2  # 指向第 3 个字符 (0-indexed)

# 显示给用户
from modules.utils.indexing import to_display_index
display_index = to_display_index(suspicious_index)  # 返回 3
prompt = f"系统检测到第 {display_index} 个字符存疑"  # "第 3 个字符"
```

---

## 6. 文件路径约定

### 图像路径解析

`HCTRDatasetLoader` 支持以下路径格式：

```python
# 相对路径 (相对于 data_dir)
"images/001.png"       → data_dir/images/001.png
"train/001.png"        → data_dir/train/001.png

# 绝对路径
"/data/images/001.png" → /data/images/001.png

# 自动补全扩展名
"images/001"           → data_dir/images/001.jpg (尝试 .jpg, .png 等)
```

---

## 7. 使用示例

### 加载数据集

```python
from scripts.data_pipeline import HCTRDatasetLoader

# 自动检测格式（优先使用 metadata.jsonl）
loader = HCTRDatasetLoader("./data/raw/viscgec", format="auto")

for sample in loader.load():
    print(f"ID: {sample.id}")
    print(f"Image: {sample.image_path}")
    print(f"GT: {sample.ground_truth}")
    print(f"Source: {sample.source}")
    print(f"Error Type: {sample.error_type}")
```

### 运行 Pipeline

```python
from modules import L2W1Pipeline, PipelineConfig

config = PipelineConfig(use_mock=True)
pipeline = L2W1Pipeline(config)

result = pipeline.process(
    image="./images/line_001.jpg",
    sample_id="sample_001",
    gt_text="真值文本",
    source="viscgec",
    error_type="similar_char"
)

# 输出为 Data Protocol v1.0 格式
output = result.to_dict()
```

### 评估结果

```python
from scripts.evaluate import load_predictions, evaluate_batch

# 加载嵌套格式的结果
final_texts, gt_texts, agent_a_texts, is_hard_samples, metadata_list = \
    load_predictions("./outputs/inference_results.jsonl")

# 评估（包含 error_type 聚合）
result = evaluate_batch(
    predictions=final_texts,
    references=gt_texts,
    agent_a_texts=agent_a_texts,
    metadata_list=metadata_list
)

# 打印包含 error_type 分组的报告
result.print_summary()
```

---

## 版本历史

- **v1.0** (2025-12-22): 初始版本，定义嵌套结构和 error_type 聚合


