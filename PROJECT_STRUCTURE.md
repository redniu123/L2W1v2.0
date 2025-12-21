# L2W1 v5.0 项目文件结构清单

本文档详细说明 L2W1 v5.0 项目中每个文件夹和文件的作用、使用方法以及对应的模块。

---

## 📁 项目根目录结构

```
L2W1/
├── README.md                    # 项目主文档
├── requirements.txt             # Python 依赖包列表
├── PROJECT_STRUCTURE.md         # 本文件：项目结构说明
│
├── configs/                     # 配置文件目录
├── data/                        # 数据目录
├── docs/                        # 文档目录
├── models/                     # 模型权重目录
├── modules/                     # 核心模块代码
├── notebooks/                    # Jupyter 笔记本（可选）
├── outputs/                     # 输出结果目录
└── scripts/                     # 脚本工具目录
```

---

## 📂 1. 配置文件 (configs/)

### 1.1 `configs/__init__.py`
- **作用**: Python 包初始化文件，导出配置加载函数
- **模块**: 配置管理模块
- **使用**: 通过 `from configs import load_config` 导入

### 1.2 `configs/router_config.yaml`
- **作用**: Router 不确定性阈值配置文件
- **模块**: Module 2 - Router 模块
- **内容**:
  ```yaml
  visual:
    entropy_threshold_low: 2.0
    entropy_threshold_high: 4.0
  semantic:
    ppl_threshold_low: 50.0
    ppl_threshold_high: 200.0
  ```
- **使用**: 
  - 修改阈值以调整 Router 的敏感度
  - 通过 `modules/router/uncertainty_router.py` 自动加载

---

## 📂 2. 数据目录 (data/)

### 2.1 `data/raw/`
- **作用**: 存放原始数据集（如 SCUT-HCCDoc, VisCGEC）
- **模块**: 数据预处理模块
- **使用**: 放置原始图像和标注文件

### 2.2 `data/processed/`
- **作用**: 存放处理后的数据（如裁剪后的行级图像）
- **模块**: 数据预处理模块
- **使用**: `scripts/data_pipeline.py` 的输出目录

### 2.3 `data/sft/agent_b_train.jsonl`
- **作用**: Agent B 的 SFT 训练数据集（JSONL 格式）
- **模块**: Module 3 - SFT 数据生成模块
- **格式**:
  ```json
  {
    "id": "sample_001",
    "image": "./images/img_001.jpg",
    "conversations": [
      {"from": "user", "value": "OCR识别结果为：'...'。\n系统检测到第 X 个字..."},
      {"from": "assistant", "value": "修正后的文本"}
    ]
  }
  ```
- **使用**: 
  - 由 `scripts/data_pipeline.py` 自动生成
  - 作为 `scripts/train_agent_b.py` 的输入

### 2.4 `data/test/inference_results.jsonl`
- **作用**: 推理结果测试文件（用于评估）
- **模块**: Module 5 - 评估模块
- **格式**:
  ```json
  {
    "image_path": "./images/001.jpg",
    "agent_a_text": "原始识别结果",
    "final_text": "系统最终输出",
    "gt_text": "真值文本",
    "is_hard": true
  }
  ```
- **使用**: 作为 `scripts/evaluate.py` 的输入

### 2.5 `data/test/evaluation_report.json`
- **作用**: 评估报告输出文件
- **模块**: Module 5 - 评估模块
- **生成**: 由 `scripts/evaluate.py` 自动生成
- **使用**: 作为 `scripts/visualize_results.py` 的输入

---

## 📂 3. 文档目录 (docs/)

### 3.1 `docs/PROJECT_CONTEXT.md`
- **作用**: 项目背景和整体架构说明
- **模块**: 项目总览
- **内容**: L2W1 v5.0 的设计理念、技术路线、核心创新点

### 3.2 `docs/module3_pipeline_spec.md`
- **作用**: Module 3 - 数据流水线规范
- **模块**: Module 3 - SFT 数据生成
- **内容**: 
  - 数据加载规范
  - 错误索引对齐算法
  - SFT Prompt 模板定义
- **对应代码**: `scripts/data_pipeline.py`

### 3.3 `docs/module3_sft_spec.md`
- **作用**: Module 3 - SFT 训练规范
- **模块**: Module 3 - Agent B 微调
- **内容**: 
  - 负样本生成策略
  - Label Masking 实现
  - 训练超参数配置
- **对应代码**: `scripts/train_agent_b.py`, `scripts/sft_dataset.py`

### 3.4 `docs/module4_agent_b_spec.md`
- **作用**: Module 4 - Agent B 推理规范
- **模块**: Module 4 - VLM 专家模块
- **内容**: 
  - 4-bit 量化配置
  - 动态分辨率处理
  - EIP (显式索引提示) 策略
- **对应代码**: `modules/vlm_expert/agent_b_expert.py`

### 3.5 `docs/module4_spec.md`
- **作用**: Module 4 - Agent B 完整规范（补充）
- **模块**: Module 4 - VLM 专家模块

### 3.6 `docs/module5_evaluate_spec.md`
- **作用**: Module 5 - 评估指标规范
- **模块**: Module 5 - 评估模块
- **内容**: 
  - CER 计算公式
  - OCR-R (过度纠错率) 定义
  - 非对称编辑距离分析算法
- **对应代码**: `scripts/evaluate.py`

---

## 📂 4. 模型目录 (models/)

### 4.1 `models/agent_a_ppocr/`
- **作用**: Agent A (PP-OCRv5) 模型权重目录
- **模块**: Module 1 - Agent A 模块
- **使用**: 存放 PaddleOCR 识别模型权重
- **来源**: 从 PaddleOCR 官方下载

### 4.2 `models/agent_b_vlm/lora_checkpoints/`
- **作用**: Agent B (Qwen2.5-VL-3B) LoRA 微调检查点
- **模块**: Module 3 - Agent B 训练模块
- **生成**: 由 `scripts/train_agent_b.py` 训练后保存
- **使用**: 推理时加载 LoRA 权重

### 4.3 `models/router_qwen/`
- **作用**: Router (Qwen2.5-0.5B) 模型权重目录
- **模块**: Module 2 - Router 模块
- **使用**: 存放轻量级语言模型（用于计算语义困惑度）

---

## 📂 5. 核心模块代码 (modules/)

### 5.1 `modules/__init__.py`
- **作用**: 模块包初始化，导出主要接口
- **导出**: 
  - `L2W1Pipeline`: 端到端推理管道
  - `PipelineConfig`: 管道配置类
- **使用**: `from modules import L2W1Pipeline`

### 5.2 `modules/pipeline.py`
- **作用**: **核心管道模块** - 整合 Agent A, Router, Agent B
- **模块**: 系统集成模块
- **主要类**:
  - `L2W1Pipeline`: 端到端推理管道
  - `PipelineConfig`: 管道配置
- **使用示例**:
  ```python
  from modules import L2W1Pipeline, PipelineConfig
  
  config = PipelineConfig(
      agent_a_model_path="./models/agent_a_ppocr",
      agent_b_model_path="Qwen/Qwen2.5-VL-3B-Instruct",
      router_config_path="./configs/router_config.yaml"
  )
  
  pipeline = L2W1Pipeline(config)
  result = pipeline.process(image_path, return_details=True)
  ```

---

### 5.3 `modules/paddle_engine/` - Agent A 模块

#### 5.3.1 `modules/paddle_engine/__init__.py`
- **作用**: 导出 `TextRecognizerWithLogits`
- **模块**: Module 1 - Agent A 模块

#### 5.3.2 `modules/paddle_engine/predict_rec_modified.py`
- **作用**: **修改后的 PaddleOCR 推理逻辑** - 拦截原始 Logits
- **模块**: Module 1 - Agent A 模块
- **核心功能**:
  - 在 CTC Decode 之前拦截 `raw_logits` (形状: `[B, Seq_Len, Vocab_Size]`)
  - 使用 `deepcopy` 防止内存复用覆盖
  - 返回字典: `{'text': rec_text, 'conf': rec_score, 'logits': raw_logits}`
- **关键代码位置**: `__call__` 方法中的 Logits 拦截逻辑
- **使用**: 由 `modules/pipeline.py` 内部调用

#### 5.3.3 `modules/paddle_engine/demo_logits_hook.py`
- **作用**: Logits 拦截功能演示脚本
- **模块**: Module 1 - Agent A 模块
- **使用**: 
  ```bash
  python modules/paddle_engine/demo_logits_hook.py
  ```
- **输出**: 打印 Logits 的形状和统计信息

---

### 5.4 `modules/router/` - Router 模块

#### 5.4.1 `modules/router/__init__.py`
- **作用**: 导出 Router 相关类和函数
- **导出**: 
  - `UncertaintyRouter`: 不确定性路由器
  - `calculate_visual_entropy`: 视觉熵计算函数

#### 5.4.2 `modules/router/uncertainty_router.py`
- **作用**: **Router 核心实现** - 计算视觉熵和语义困惑度
- **模块**: Module 2 - Router 模块
- **主要类**:
  - `CTCAligner`: CTC 时间步对齐器（将固定长度 logits 映射到字符位置）
  - `VisualEntropyCalculator`: 视觉熵计算器
  - `SemanticPPLCalculator`: 语义困惑度计算器
  - `UncertaintyRouter`: 路由器主类
- **核心算法**:
  - **CTC 对齐**: 将 80 长度的 logits 序列对齐到 M 长度的文本
  - **视觉不确定性**: $U_{vis} = H(p_t)$ (Shannon Entropy)
  - **语义不确定性**: $U_{sem} = PPL(text)$ (Perplexity)
  - **路由决策**: 双门控机制（视觉 OR 语义超过阈值）
- **输出格式**:
  ```python
  {
      "is_hard": bool,
      "suspicious_index": int,      # 0-based 字符索引
      "suspicious_char": str,
      "risk_level": str,            # "high", "medium", "low"
      "visual_entropy": float,
      "semantic_ppl": float
  }
  ```
- **使用**: 由 `modules/pipeline.py` 内部调用

---

### 5.5 `modules/vlm_expert/` - Agent B 模块

#### 5.5.1 `modules/vlm_expert/__init__.py`
- **作用**: 导出 Agent B 相关类和配置
- **导出**: 
  - `AgentBExpert`: Agent B 推理类
  - `AgentBExpertMock`: Mock 类（用于测试）
  - `AgentBConfig`: Agent B 配置类

#### 5.5.2 `modules/vlm_expert/agent_b_expert.py`
- **作用**: **Agent B 核心实现** - Qwen2.5-VL-3B 视觉专家
- **模块**: Module 4 - Agent B 模块
- **核心功能**:
  - **4-bit 量化**: 使用 `bitsandbytes` NF4 量化（适配 11GB VRAM）
  - **动态分辨率**: `min_pixels=200704, max_pixels=1003520`（支持极端长宽比）
  - **EIP 策略**: 显式索引提示（"请重点检查第 X 个字符"）
- **主要类**:
  - `AgentBExpert`: 主推理类
  - `AgentBConfig`: 配置类
  - `EIPPromptTemplate`: EIP 提示模板
- **使用示例**:
  ```python
  from modules.vlm_expert import AgentBExpert, AgentBConfig
  
  config = AgentBConfig(
      model_path="Qwen/Qwen2.5-VL-3B-Instruct",
      use_4bit=True,
      min_pixels=200704,
      max_pixels=1003520
  )
  
  expert = AgentBExpert(config)
  result = expert.process_hard_sample(
      image_path="./images/line_001.jpg",
      manifest={"suspicious_index": 5, "ocr_text": "..."}
  )
  ```
- **使用**: 由 `modules/pipeline.py` 内部调用

---

## 📂 6. 脚本工具目录 (scripts/)

### 6.1 `scripts/__init__.py`
- **作用**: 脚本包初始化，导出主要工具类
- **导出**: 
  - `DataPipeline`: 数据流水线
  - `AgentBTrainer`: Agent B 训练器
  - `evaluate_metrics`: 评估指标计算

---

### 6.2 `scripts/data_pipeline.py`
- **作用**: **SFT 数据生成流水线** - 自动生成 Agent B 训练数据
- **模块**: Module 3 - SFT 数据生成模块
- **主要类**:
  - `HCTRDatasetLoader`: 数据集加载器
  - `ErrorAnalyzer`: 错误分析器（使用 `difflib.SequenceMatcher`）
  - `SFTGenerator`: SFT 样本生成器
  - `DataPipeline`: 主流水线类
- **核心功能**:
  - 加载 HCTR 数据集（图像 + 真值）
  - 使用 Agent A 进行推理，获取 `pred_text`
  - 使用 `difflib` 对齐 `pred_text` 和 `gt_text`，找到第一个错误位置
  - 生成 EIP Prompt: "系统检测到第 X 个字（即'Y'）置信度存疑..."
  - 输出 JSONL 格式的 SFT 数据集
- **使用**:
  ```bash
  python scripts/data_pipeline.py \
      --data_dir ./data/raw/scut_hccdoc \
      --output_path ./data/sft/agent_b_train.jsonl \
      --batch_size 32
  ```
- **输出**: `data/sft/agent_b_train.jsonl`

---

### 6.3 `scripts/sft_dataset.py`
- **作用**: **SFT 数据集类** - 用于 Agent B 训练
- **模块**: Module 3 - SFT 训练模块
- **主要类**:
  - `AgentBSFTDataset`: PyTorch Dataset 类
  - `NegativeSampleTemplates`: 负样本模板生成器
  - `create_data_collator`: 数据整理器工厂函数
- **核心功能**:
  - **负样本生成**: 自动混合 15% 的负样本（识别正确但要求确认）
  - **Label Masking**: 只在 assistant 回复部分计算 Loss（user prompt 部分 mask 为 -100）
  - **动态分辨率**: 与推理配置保持一致（`min_pixels=200704, max_pixels=1003520`）
- **使用**: 由 `scripts/train_agent_b.py` 内部调用

---

### 6.4 `scripts/train_agent_b.py`
- **作用**: **Agent B QLoRA 训练脚本**
- **模块**: Module 3 - Agent B 训练模块
- **主要类**:
  - `TrainingConfig`: 训练配置类
  - `AgentBTrainer`: 训练器类
  - `MetricsComputer`: 指标计算器
- **核心功能**:
  - **QLoRA 配置**: r=16, alpha=32, dropout=0.05
  - **4-bit 量化**: NF4 + Double Quantization
  - **内存优化**: Gradient Checkpointing + Gradient Accumulation
  - **有效批次**: 4 × 32 = 128
  - **学习率**: 2e-4 + Cosine 调度
- **使用**:
  ```bash
  # 测试模式
  python scripts/train_agent_b.py --mock
  
  # 真实训练
  python scripts/train_agent_b.py \
      --data_path ./data/sft/agent_b_train.jsonl \
      --output_dir ./models/agent_b_vlm/lora_checkpoints \
      --model_path Qwen/Qwen2.5-VL-3B-Instruct \
      --num_epochs 3 \
      --batch_size 4 \
      --gradient_accumulation_steps 32 \
      --learning_rate 2e-4
  ```
- **输出**: LoRA 权重保存在 `models/agent_b_vlm/lora_checkpoints/`

---

### 6.5 `scripts/evaluate.py`
- **作用**: **评估指标计算脚本** - 非对称编辑距离分析
- **模块**: Module 5 - 评估模块
- **核心功能**:
  - **CER 计算**: $(S + D + I) / N$，详细统计 S, D, I
  - **OCR-R 计算**: 过度纠错率（核心创新指标）
    - 公式: $OCR\text{-}R = \frac{Count(A_{correct} \rightarrow S_{wrong})}{Total_{correct}^A}$
    - 使用非对称对齐算法
  - **Correction Rate**: 纠正率
  - **Hard Sample Recall**: Router 召回率
- **主要类**:
  - `EvaluationResult`: 评估结果数据类
  - `MetricsComputer`: 指标计算器（静态方法）
- **使用**:
  ```bash
  # 测试模式
  python scripts/evaluate.py --test
  
  # 评估推理结果
  python scripts/evaluate.py \
      --predictions ./data/test/inference_results.jsonl \
      --verbose
  
  # 保存评估报告
  python scripts/evaluate.py \
      --predictions ./data/test/inference_results.jsonl \
      --output ./data/test/evaluation_report.json
  ```
- **输出**: `evaluation_report.json`（包含 CER, OCR-R, Correction Rate 等）

---

### 6.6 `scripts/visualize_results.py`
- **作用**: **论文图表生成脚本** - 生成出版级图表
- **模块**: Module 5 - 可视化模块
- **核心功能**:
  - **幻觉矩阵**: 2x2 混淆矩阵（展示过度纠错 vs 成功纠正）
  - **帕累托前沿**: CER vs Latency 性能边界曲线
  - **Router 阈值分析**: 阈值 τ 对 CER/OCR-R 的影响
  - **注意力热力图**: Agent B 视觉聚焦分布
  - **案例对比网格**: 多案例前后对比
- **使用**:
  ```bash
  # 生成示例图表
  python scripts/visualize_results.py --demo
  
  # 从评估报告生成图表
  python scripts/visualize_results.py \
      --eval_report ./data/test/evaluation_report.json \
      --output_dir ./outputs/figures
  ```
- **输出**: 
  - `hallucination_matrix.png`
  - `pareto_frontier.png`
  - `threshold_analysis.png`
  - `attention_heatmap.png`
  - `comparison_grid.png`

---

## 📂 7. 输出目录 (outputs/)

### 7.1 `outputs/figures/`
- **作用**: 存放生成的论文图表
- **模块**: Module 5 - 可视化模块
- **生成**: 由 `scripts/visualize_results.py` 生成
- **文件**: PNG 格式，300 DPI，适合论文出版

---

## 📂 8. 根目录文件

### 8.1 `README.md`
- **作用**: 项目主文档
- **内容**: 
  - 项目简介
  - 快速开始指南
  - 安装说明
  - 使用示例

### 8.2 `requirements.txt`
- **作用**: Python 依赖包列表
- **主要依赖**:
  - `paddlepaddle`: PaddleOCR 引擎
  - `torch`: PyTorch（Agent B 训练）
  - `transformers`: HuggingFace Transformers（Agent B）
  - `peft`: LoRA 微调库
  - `bitsandbytes`: 4-bit 量化
  - `matplotlib`, `seaborn`: 可视化
- **安装**:
  ```bash
  pip install -r requirements.txt
  ```

---

## 🔄 模块依赖关系图

```
Module 1 (Agent A)
    ↓
Module 2 (Router) ← configs/router_config.yaml
    ↓
Module 3 (Data Pipeline) → data/sft/agent_b_train.jsonl
    ↓
Module 3 (Train Agent B) → models/agent_b_vlm/lora_checkpoints/
    ↓
Module 4 (Agent B) ← models/agent_b_vlm/lora_checkpoints/
    ↓
Module 5 (Evaluate) → data/test/evaluation_report.json
    ↓
Module 5 (Visualize) → outputs/figures/
```

---

## 📋 典型工作流程

### 流程 1: 端到端推理
```python
from modules import L2W1Pipeline, PipelineConfig

config = PipelineConfig(...)
pipeline = L2W1Pipeline(config)
result = pipeline.process(image_path)
```

### 流程 2: 生成训练数据
```bash
python scripts/data_pipeline.py \
    --data_dir ./data/raw \
    --output_path ./data/sft/agent_b_train.jsonl
```

### 流程 3: 训练 Agent B
```bash
python scripts/train_agent_b.py \
    --data_path ./data/sft/agent_b_train.jsonl \
    --output_dir ./models/agent_b_vlm/lora_checkpoints
```

### 流程 4: 评估系统
```bash
# 1. 运行推理生成结果
# 2. 评估
python scripts/evaluate.py --predictions ./inference_results.jsonl
# 3. 可视化
python scripts/visualize_results.py --eval_report ./evaluation_report.json
```

---

## 🎯 关键文件速查表

| 文件 | 作用 | 模块 | 何时使用 |
|------|------|------|----------|
| `modules/pipeline.py` | 端到端推理管道 | 系统集成 | 推理时 |
| `modules/paddle_engine/predict_rec_modified.py` | Agent A Logits 拦截 | Module 1 | 推理时（内部） |
| `modules/router/uncertainty_router.py` | Router 路由决策 | Module 2 | 推理时（内部） |
| `modules/vlm_expert/agent_b_expert.py` | Agent B 视觉专家 | Module 4 | 推理时（内部） |
| `scripts/data_pipeline.py` | 生成 SFT 数据 | Module 3 | 训练前 |
| `scripts/train_agent_b.py` | 训练 Agent B | Module 3 | 训练时 |
| `scripts/evaluate.py` | 评估指标计算 | Module 5 | 评估时 |
| `scripts/visualize_results.py` | 生成论文图表 | Module 5 | 论文撰写时 |
| `configs/router_config.yaml` | Router 阈值配置 | Module 2 | 调参时 |

---

## 📝 注意事项

1. **路径依赖**: 所有相对路径基于项目根目录 `L2W1/`
2. **模型权重**: Agent A 和 Router 模型需要单独下载
3. **数据格式**: SFT 数据必须符合 JSONL 格式规范
4. **显存要求**: Agent B 训练需要 11GB+ VRAM（RTX 2080Ti）
5. **中文支持**: 可视化脚本需要中文字体（SimHei/Microsoft YaHei）

---

**最后更新**: 2025-12-20  
**版本**: L2W1 v5.0

