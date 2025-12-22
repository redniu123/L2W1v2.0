# L2W1 v5.0 真实数据运行指南

本指南将帮助您使用真实数据完整运行 L2W1 v5.0 项目。

---

## 📋 目录

1. [模型下载说明](#模型下载说明)
2. [数据准备](#数据准备)
3. [完整运行流程](#完整运行流程)
4. [分步骤运行](#分步骤运行)
5. [常见问题](#常见问题)

---

## 📦 模型下载说明

### 自动下载的模型

以下模型会在首次使用时**自动从 HuggingFace 下载**：

| 模型 | 来源 | 大小 | 下载位置 |
|------|------|------|----------|
| **Agent B (Qwen2.5-VL-3B)** | HuggingFace | ~6GB | `~/.cache/huggingface/hub/` |
| **Router 语言模型 (Qwen2.5-0.5B)** | HuggingFace | ~1GB | `~/.cache/huggingface/hub/` |

**说明**：
- 使用 `transformers` 库的 `from_pretrained()` 会自动下载
- 首次运行时会显示下载进度
- 下载后会自动缓存，后续运行无需重新下载
- 如果网络较慢，可以预先下载（见下方"手动下载"）

### 需要手动下载的模型

| 模型 | 来源 | 大小 | 下载位置 |
|------|------|------|----------|
| **Agent A (PP-OCRv5-Rec)** | PaddleOCR | ~10MB | `models/agent_a_ppocr/` |

**下载方法**：

```bash
# 方法 1: 使用 PaddleOCR 官方工具
pip install paddleocr
python -c "
from paddleocr import PaddleOCR
ocr = PaddleOCR(use_angle_cls=False, lang='ch')
# 模型会自动下载到 ~/.paddleocr/whl/rec/ch/
"

# 方法 2: 手动下载（推荐）
mkdir -p models/agent_a_ppocr
cd models/agent_a_ppocr

# 下载 PP-OCRv5 识别模型
wget https://paddleocr.bj.bcebos.com/PP-OCRv5/chinese/ch_PP-OCRv5_rec_infer.tar
tar -xf ch_PP-OCRv5_rec_infer.tar
mv inference/* .
rm -rf inference ch_PP-OCRv5_rec_infer.tar

# 验证文件
ls -lh
# 应该看到: inference.pdmodel, inference.pdiparams 等文件
```

---

## 📁 数据准备

### 数据格式要求

L2W1 需要以下格式的数据：

```
data/
├── raw/                    # 原始数据目录
│   ├── images/            # 图像文件目录
│   │   ├── line_001.jpg
│   │   ├── line_002.jpg
│   │   └── ...
│   └── labels.txt         # 标注文件（格式：image_path\tground_truth）
└── test/                  # 测试数据（可选）
    ├── images/
    └── labels.txt
```

### labels.txt 格式

每行格式：`图像路径\t真值文本`

```txt
images/line_001.jpg	中国科学院计算技术研究所
images/line_002.jpg	在时间的未尾，我们相遇
images/line_003.jpg	手写文本识别系统
```

### 准备测试数据

```bash
# 1. 创建数据目录
mkdir -p data/raw/images
mkdir -p data/test/images

# 2. 将图像文件放入 images/ 目录
# cp your_images/*.jpg data/raw/images/

# 3. 创建标注文件
cat > data/raw/labels.txt << 'EOF'
images/line_001.jpg	真值文本1
images/line_002.jpg	真值文本2
EOF

# 4. 验证数据
head -n 5 data/raw/labels.txt
ls data/raw/images/ | head -n 5
```

---

## 🚀 完整运行流程

### 方式 1: 端到端 Pipeline（推荐）

```bash
# 激活环境
conda activate l2w1v2

# 创建运行脚本
cat > run_pipeline.py << 'EOF'
#!/usr/bin/env python3
"""L2W1 端到端运行脚本"""
import sys
from pathlib import Path
from modules import L2W1Pipeline, PipelineConfig

# 配置
config = PipelineConfig(
    # Agent A 配置
    agent_a_model_dir="./models/agent_a_ppocr",
    
    # Agent B 配置（会自动下载）
    agent_b_model_path="Qwen/Qwen2.5-VL-3B-Instruct",
    agent_b_use_4bit=True,
    
    # Router 配置
    entropy_threshold_low=2.0,
    entropy_threshold_high=4.0,
    
    # 其他
    verbose=True
)

# 创建 Pipeline
print("正在初始化 L2W1 Pipeline...")
pipeline = L2W1Pipeline(config)
print("Pipeline 初始化完成!\n")

# 处理单张图像
image_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/images/line_001.jpg"

if not Path(image_path).exists():
    print(f"错误: 图像文件不存在: {image_path}")
    sys.exit(1)

print(f"处理图像: {image_path}")
result = pipeline.process(image_path)

# 输出结果
print("\n" + "="*60)
print("L2W1 推理结果")
print("="*60)
print(f"Agent A 识别: {result.agent_a_text}")
print(f"是否困难样本: {result.is_hard}")
if result.is_hard:
    print(f"风险等级: {result.risk_level}")
    print(f"存疑字符索引: {result.suspicious_index} (字符: '{result.suspicious_char}')")
    print(f"Agent B 修正: {result.agent_b_text}")
print(f"最终输出: {result.final_text}")
print("="*60)
EOF

# 运行
python run_pipeline.py data/raw/images/line_001.jpg
```

### 方式 2: 批量处理

```bash
cat > run_batch.py << 'EOF'
#!/usr/bin/env python3
"""批量处理脚本"""
import json
from pathlib import Path
from modules import L2W1Pipeline, PipelineConfig

# 配置
config = PipelineConfig(
    agent_a_model_dir="./models/agent_a_ppocr",
    agent_b_model_path="Qwen/Qwen2.5-VL-3B-Instruct",
    agent_b_use_4bit=True,
    verbose=False
)

# 创建 Pipeline
pipeline = L2W1Pipeline(config)

# 读取标注文件
labels_file = Path("data/raw/labels.txt")
results = []

with open(labels_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if not line or '\t' not in line:
            continue
        
        image_path, gt_text = line.split('\t', 1)
        full_path = Path("data/raw") / image_path
        
        if not full_path.exists():
            print(f"跳过: {full_path} 不存在")
            continue
        
        print(f"处理: {image_path}")
        result = pipeline.process(str(full_path))
        
        results.append({
            'image': image_path,
            'agent_a_text': result.agent_a_text,
            'final_text': result.final_text,
            'gt_text': gt_text,
            'is_hard': result.is_hard,
            'routed_to_agent_b': result.routed_to_agent_b
        })

# 保存结果
output_file = Path("data/test/inference_results.jsonl")
with open(output_file, 'w', encoding='utf-8') as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')

print(f"\n处理完成! 结果保存到: {output_file}")
print(f"共处理 {len(results)} 个样本")
EOF

python run_batch.py
```

---

## 📝 分步骤运行

### 步骤 1: 生成 SFT 数据集

```bash
# 运行数据管道
python scripts/data_pipeline.py \
    --data_dir ./data/raw \
    --output_path ./data/sft/agent_b_train.jsonl \
    --batch_size 32 \
    --max_cer 0.3

# 检查输出
head -n 3 ./data/sft/agent_b_train.jsonl
wc -l ./data/sft/agent_b_train.jsonl
```

**预期输出**：
- `data/sft/agent_b_train.jsonl`: 训练数据集
- 控制台显示处理进度和统计信息

### 步骤 2: 训练 Agent B（可选）

```bash
# 训练 Agent B
python scripts/train_agent_b.py \
    --data_path ./data/sft/agent_b_train.jsonl \
    --output_dir ./models/agent_b_vlm/lora_checkpoints \
    --model_path Qwen/Qwen2.5-VL-3B-Instruct \
    --use_4bit \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 3 \
    --save_steps 500

# 检查检查点
ls -lh ./models/agent_b_vlm/lora_checkpoints/
```

### 步骤 3: 运行推理

```bash
# 使用 Pipeline 处理图像
python run_pipeline.py data/raw/images/line_001.jpg

# 或批量处理
python run_batch.py
```

### 步骤 4: 评估结果

```bash
# 运行评估
python scripts/evaluate.py \
    --predictions ./data/test/inference_results.jsonl \
    --output_dir ./data/test \
    --save_report

# 查看评估报告
cat ./data/test/evaluation_report.json
```

### 步骤 5: 可视化（可选）

```bash
# 生成可视化图表
python scripts/visualize_results.py \
    --eval_report ./data/test/evaluation_report.json \
    --output_dir ./outputs/figures
```

---

## 🔧 模型下载详细说明

### Agent B 模型（自动下载）

**首次运行时会自动下载**，您也可以预先下载：

```bash
# 预先下载 Qwen2.5-VL-3B
python -c "
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
print('正在下载 Qwen2.5-VL-3B...')
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    'Qwen/Qwen2.5-VL-3B-Instruct',
    trust_remote_code=True
)
processor = AutoProcessor.from_pretrained(
    'Qwen/Qwen2.5-VL-3B-Instruct',
    trust_remote_code=True
)
print('下载完成!')
"

# 模型会下载到: ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/
```

### Agent A 模型（手动下载）

```bash
# 创建模型目录
mkdir -p models/agent_a_ppocr
cd models/agent_a_ppocr

# 下载 PP-OCRv5 中文识别模型
wget https://paddleocr.bj.bcebos.com/PP-OCRv5/chinese/ch_PP-OCRv5_rec_infer.tar

# 解压
tar -xf ch_PP-OCRv5_rec_infer.tar

# 整理文件
mv inference/* .
rmdir inference
rm ch_PP-OCRv5_rec_infer.tar

# 验证
ls -lh
# 应该看到:
# - inference.pdmodel
# - inference.pdiparams
# - inference.yml (可选)

cd ../..
```

### Router 语言模型（自动下载，可选）

Router 的语言模型会在首次计算语义 PPL 时自动下载。如果需要预先下载：

```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('正在下载 Qwen2.5-0.5B...')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')
print('下载完成!')
"
```

---

## ⚙️ 配置说明

### Pipeline 配置

```python
config = PipelineConfig(
    # Agent A 配置
    agent_a_model_dir="./models/agent_a_ppocr",  # PP-OCRv5 模型路径
    
    # Agent B 配置
    agent_b_model_path="Qwen/Qwen2.5-VL-3B-Instruct",  # HuggingFace 模型名或本地路径
    agent_b_use_4bit=True,  # 使用 4-bit 量化（节省显存）
    
    # Router 配置
    entropy_threshold_low=2.0,   # 视觉熵低阈值
    entropy_threshold_high=4.0,  # 视觉熵高阈值
    ppl_threshold_low=50.0,      # 语义 PPL 低阈值
    ppl_threshold_high=200.0,    # 语义 PPL 高阈值
    
    # 其他
    verbose=True  # 打印详细日志
)
```

### 使用本地模型

如果已经下载了模型到本地：

```python
config = PipelineConfig(
    agent_b_model_path="./models/qwen2.5-vl-3b",  # 本地路径
    # ...
)
```

---

## ❓ 常见问题

### Q1: Agent B 模型下载很慢

**解决方案**：
1. 使用镜像源（如果在中国）：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   ```
2. 预先下载（见上方"模型下载详细说明"）
3. 使用代理

### Q2: Agent A 模型找不到

**错误**: `not find model file path`

**解决**:
```bash
# 检查模型路径
ls -lh models/agent_a_ppocr/

# 确保有以下文件:
# - inference.pdmodel
# - inference.pdiparams

# 如果缺少，重新下载（见上方"Agent A 模型下载"）
```

### Q3: 显存不足

**错误**: CUDA OOM

**解决**:
```python
# 确保使用 4-bit 量化
config = PipelineConfig(
    agent_b_use_4bit=True,  # 必须为 True
    # ...
)
```

### Q4: 网络问题导致模型下载失败

**解决**:
1. 手动下载模型到本地
2. 使用本地路径：
   ```python
   config.agent_b_model_path = "./models/qwen2.5-vl-3b"
   ```

---

## 📊 快速验证

运行以下命令验证所有组件：

```bash
# 1. 检查 Agent A 模型
ls -lh models/agent_a_ppocr/inference.*

# 2. 测试 Pipeline（会自动下载 Agent B）
python -c "
from modules import L2W1Pipeline, PipelineConfig
config = PipelineConfig(agent_a_model_dir='./models/agent_a_ppocr')
pipeline = L2W1Pipeline(config)
print('✓ Pipeline 初始化成功')
"

# 3. 检查数据
head -n 3 data/raw/labels.txt
ls data/raw/images/ | head -n 3
```

---

## 🎯 完整示例

```bash
# 1. 准备数据
mkdir -p data/raw/images
# 将图像放入 data/raw/images/
# 创建 data/raw/labels.txt

# 2. 下载 Agent A 模型
mkdir -p models/agent_a_ppocr
cd models/agent_a_ppocr
wget https://paddleocr.bj.bcebos.com/PP-OCRv5/chinese/ch_PP-OCRv5_rec_infer.tar
tar -xf ch_PP-OCRv5_rec_infer.tar && mv inference/* . && rm -rf inference *.tar
cd ../..

# 3. 运行 Pipeline（Agent B 会自动下载）
python run_pipeline.py data/raw/images/line_001.jpg

# 4. 批量处理
python run_batch.py

# 5. 评估
python scripts/evaluate.py --predictions ./data/test/inference_results.jsonl
```

---

**准备好数据后，按照上述步骤运行即可！** 🚀

