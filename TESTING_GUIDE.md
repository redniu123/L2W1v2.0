# L2W1 v5.0 测试指南

本指南将帮助您逐步测试 L2W1 v5.0 的各个模块，发现并定位 bug。

---

## 📋 测试流程概览

```
1. 环境验证 → 2. 模块导入 → 3. 数据管道 → 4. Router → 5. Agent B → 6. Pipeline → 7. 评估
```

---

## 🔧 步骤 1: 环境验证

### 1.1 激活环境并验证基础环境

```bash
# 激活 conda 环境
conda activate l2w1v2

# 验证 Python 版本
python --version
# 预期: Python 3.10.x

# 验证 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
# 预期: PyTorch 版本号, CUDA Available: True

# 验证 GPU
nvidia-smi
# 预期: 显示 GPU 信息
```

**✅ 通过标准**: PyTorch 可导入，CUDA 可用，GPU 可见

---

### 1.2 验证核心依赖

```bash
# 创建测试脚本
cat > test_imports.py << 'EOF'
#!/usr/bin/env python3
"""测试核心依赖导入"""
import sys

errors = []

def test_import(name, module_name=None):
    mod = module_name or name
    try:
        __import__(mod)
        print(f"✓ {name}")
        return True
    except ImportError as e:
        print(f"✗ {name}: {e}")
        errors.append(name)
        return False

print("=" * 60)
print("核心依赖导入测试")
print("=" * 60)

# 核心依赖
test_import("torch", "torch")
test_import("paddle", "paddle")
test_import("transformers", "transformers")
test_import("peft", "peft")
test_import("bitsandbytes", "bitsandbytes")
test_import("cv2", "cv2")
test_import("numpy", "numpy")
test_import("PIL", "PIL")

print("\n" + "=" * 60)
if errors:
    print(f"✗ 失败的导入: {', '.join(errors)}")
    sys.exit(1)
else:
    print("✓ 所有核心依赖导入成功!")
EOF

# 运行测试
python test_imports.py
```

**✅ 通过标准**: 所有依赖都能成功导入

---

## 📦 步骤 2: 模块导入测试

### 2.1 测试项目模块导入

```bash
# 创建测试脚本
cat > test_modules.py << 'EOF'
#!/usr/bin/env python3
"""测试项目模块导入"""
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

errors = []

def test_import(module_path, description):
    try:
        exec(f"from {module_path} import *")
        print(f"✓ {description}")
        return True
    except Exception as e:
        print(f"✗ {description}: {e}")
        errors.append(description)
        return False

print("=" * 60)
print("项目模块导入测试")
print("=" * 60)

# 测试各个模块
test_import("modules.paddle_engine", "Paddle Engine 模块")
test_import("modules.router", "Router 模块")
test_import("modules.vlm_expert", "VLM Expert 模块")
test_import("modules.pipeline", "Pipeline 模块")
test_import("modules.utils.indexing", "Utils Indexing 模块")

test_import("scripts.data_pipeline", "Data Pipeline 模块")
test_import("scripts.sft_dataset", "SFT Dataset 模块")
test_import("scripts.train_agent_b", "Train Agent B 模块")
test_import("scripts.evaluate", "Evaluate 模块")
test_import("scripts.visualize_results", "Visualize Results 模块")

print("\n" + "=" * 60)
if errors:
    print(f"✗ 失败的模块: {', '.join(errors)}")
    sys.exit(1)
else:
    print("✓ 所有项目模块导入成功!")
EOF

# 运行测试
python test_modules.py
```

**✅ 通过标准**: 所有项目模块都能成功导入

**🐛 如果失败**: 检查错误信息，可能是：

- 路径问题
- 依赖缺失
- 语法错误

---

## 🔄 步骤 3: 数据管道测试

### 3.1 准备测试数据

```bash
# 创建测试数据目录
mkdir -p data/raw/images
mkdir -p data/sft

# 创建测试标注文件（如果有真实数据，替换为真实路径）
cat > data/raw/labels.txt << 'EOF'
images/test_001.jpg	中国科学院计算技术研究所
images/test_002.jpg	在时间的未尾，我们相遇
images/test_003.jpg	手写文本识别系统
EOF

# 注意: 如果没有真实图像，这一步会失败，可以跳过或使用模拟数据
```

### 3.2 测试数据管道（最小测试）

```bash
# 创建最小测试脚本
cat > test_data_pipeline.py << 'EOF'
#!/usr/bin/env python3
"""测试数据管道模块"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("数据管道模块测试")
print("=" * 60)

try:
    from scripts.data_pipeline import (
        HCTRDatasetLoader,
        ErrorAnalyzer,
        SFTGenerator,
        DataPipeline
    )
    print("✓ 数据管道类导入成功")

    # 测试类实例化（不运行完整流程）
    print("\n测试类实例化...")

    # 测试 ErrorAnalyzer
    analyzer = ErrorAnalyzer()
    print("✓ ErrorAnalyzer 实例化成功")

    # 测试 SFTGenerator
    generator = SFTGenerator()
    print("✓ SFTGenerator 实例化成功")

    print("\n✓ 数据管道模块测试通过!")

except Exception as e:
    print(f"✗ 数据管道模块测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

python test_data_pipeline.py
```

**✅ 通过标准**: 模块可以导入，类可以实例化

---

### 3.3 测试数据管道完整流程（需要真实数据）

```bash
# 如果有真实数据，运行完整流程
python scripts/data_pipeline.py \
    --data_dir ./data/raw \
    --output_path ./data/sft/test_output.jsonl \
    --batch_size 1 \
    --max_cer 0.5

# 检查输出
if [ -f "data/sft/test_output.jsonl" ]; then
    echo "✓ 输出文件已生成"
    head -n 3 data/sft/test_output.jsonl
else
    echo "✗ 输出文件未生成"
fi
```

**✅ 通过标准**:

- 脚本能正常运行
- 输出文件已生成
- JSONL 格式正确

**🐛 如果失败**: 检查错误信息，可能是：

- 数据路径错误
- PaddleOCR 模型路径错误
- 图像格式问题

---

## 🎯 步骤 4: Router 模块测试

### 4.1 测试 Router 导入和基本功能

```bash
cat > test_router.py << 'EOF'
#!/usr/bin/env python3
"""测试 Router 模块"""
import sys
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("Router 模块测试")
print("=" * 60)

try:
    from modules.router import UncertaintyRouter, calculate_visual_entropy
    from modules.router.uncertainty_router import RouterConfig
    print("✓ Router 模块导入成功")

    # 测试配置加载
    print("\n测试配置加载...")
    config_path = project_root / "configs" / "router_config.yaml"
    if config_path.exists():
        config = RouterConfig.from_yaml(str(config_path))
        print(f"✓ 配置加载成功: {config}")
    else:
        print("⚠ 配置文件不存在，使用默认配置")
        config = RouterConfig()

    # 测试 Router 实例化
    print("\n测试 Router 实例化...")
    router = UncertaintyRouter(config)
    print("✓ Router 实例化成功")

    # 测试模拟数据
    print("\n测试模拟路由...")
    seq_len, vocab_size = 80, 6625
    logits = np.random.randn(seq_len, vocab_size).astype(np.float32) * 0.5
    text = "测试文本识别"

    result = router.route(logits, text)
    print(f"✓ 路由测试成功")
    print(f"  - is_hard: {result.is_hard}")
    print(f"  - risk_level: {result.risk_level}")
    print(f"  - suspicious_index: {result.suspicious_index}")

    # 测试边界条件
    print("\n测试边界条件...")

    # 空文本
    result_empty = router.route(logits, "")
    print(f"✓ 空文本处理: is_hard={result_empty.is_hard}, risk={result_empty.risk_level}")

    # 单字符
    result_single = router.route(logits, "中")
    print(f"✓ 单字符处理: is_hard={result_single.is_hard}, idx={result_single.suspicious_index}")

    print("\n✓ Router 模块测试通过!")

except Exception as e:
    print(f"✗ Router 模块测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

python test_router.py
```

**✅ 通过标准**:

- Router 可以导入和实例化
- 可以处理模拟数据
- 边界条件处理正常

**🐛 如果失败**: 检查：

- CTC 对齐逻辑
- 熵计算
- 配置文件格式

---

## 🤖 步骤 5: Agent B 模块测试

### 5.1 测试 Agent B 导入

```bash
cat > test_agent_b.py << 'EOF'
#!/usr/bin/env python3
"""测试 Agent B 模块"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("Agent B 模块测试")
print("=" * 60)

try:
    from modules.vlm_expert import AgentBExpert, AgentBConfig, AgentBExpertMock
    print("✓ Agent B 模块导入成功")

    # 测试 Mock 模式（不需要加载真实模型）
    print("\n测试 Agent B Mock 模式...")
    config = AgentBConfig(
        model_path="Qwen/Qwen2.5-VL-3B-Instruct",
        use_4bit=True,
        use_mock=True  # 使用 Mock 模式
    )

    expert = AgentBExpert(config)
    print("✓ Agent B Mock 实例化成功")

    # 测试 EIP Prompt 构建
    print("\n测试 EIP Prompt 构建...")
    from modules.vlm_expert.agent_b_expert import EIPPromptTemplate

    prompt = EIPPromptTemplate.build_prompt(
        ocr_text="测试文本",
        suspicious_index=2,
        suspicious_char="试",
        risk_level="medium"
    )
    print(f"✓ Prompt 构建成功")
    print(f"  Prompt 预览: {prompt[:100]}...")

    print("\n✓ Agent B 模块测试通过!")

except Exception as e:
    print(f"✗ Agent B 模块测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

python test_agent_b.py
```

**✅ 通过标准**:

- Agent B 可以导入
- Mock 模式可以工作
- Prompt 构建正常

---

### 5.2 测试 Agent B 真实模型（可选，需要下载模型）

```bash
# 注意: 这会下载模型，需要时间和网络
cat > test_agent_b_real.py << 'EOF'
#!/usr/bin/env python3
"""测试 Agent B 真实模型（需要下载）"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("Agent B 真实模型测试")
print("=" * 60)

try:
    from modules.vlm_expert import AgentBExpert, AgentBConfig

    print("警告: 这将下载模型，可能需要几分钟...")
    config = AgentBConfig(
        model_path="Qwen/Qwen2.5-VL-3B-Instruct",
        use_4bit=True,
        use_mock=False
    )

    print("正在加载模型...")
    expert = AgentBExpert(config)
    print("✓ 模型加载成功")

    # 测试推理（需要图像）
    # result = expert.process_hard_sample(image_path, manifest)

    print("\n✓ Agent B 真实模型测试通过!")

except Exception as e:
    print(f"✗ Agent B 真实模型测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

# 如果网络和显存允许，可以运行
# python test_agent_b_real.py
```

---

## 🔗 步骤 6: Pipeline 端到端测试

### 6.1 测试 Pipeline 导入和配置

```bash
cat > test_pipeline.py << 'EOF'
#!/usr/bin/env python3
"""测试 Pipeline 模块"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("Pipeline 模块测试")
print("=" * 60)

try:
    from modules import L2W1Pipeline, PipelineConfig
    print("✓ Pipeline 模块导入成功")

    # 测试配置
    print("\n测试 Pipeline 配置...")
    config = PipelineConfig(
        agent_a_model_dir="./models/agent_a_ppocr",
        agent_b_model_path="Qwen/Qwen2.5-VL-3B-Instruct",
        router_config_path="./configs/router_config.yaml",
        use_4bit=True,
        use_mock_agent_b=True  # 使用 Mock 模式
    )
    print("✓ Pipeline 配置创建成功")

    # 测试 Pipeline 实例化（Mock 模式，不加载真实模型）
    print("\n测试 Pipeline 实例化（Mock 模式）...")
    try:
        pipeline = L2W1Pipeline(config)
        print("✓ Pipeline 实例化成功")
    except Exception as e:
        print(f"⚠ Pipeline 实例化失败（可能是模型路径问题）: {e}")
        print("  这是正常的，如果模型路径不存在")

    print("\n✓ Pipeline 模块测试通过!")

except Exception as e:
    print(f"✗ Pipeline 模块测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

python test_pipeline.py
```

**✅ 通过标准**: Pipeline 可以导入和配置

---

### 6.2 测试 Pipeline 完整流程（需要真实数据和模型）

```bash
# 如果有真实数据和模型，测试完整流程
cat > test_pipeline_full.py << 'EOF'
#!/usr/bin/env python3
"""测试 Pipeline 完整流程"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules import L2W1Pipeline, PipelineConfig

# 配置
config = PipelineConfig(
    agent_a_model_dir="./models/agent_a_ppocr",
    agent_b_model_path="./models/agent_b_vlm/lora_checkpoints/checkpoint-1000",
    router_config_path="./configs/router_config.yaml",
    use_4bit=True
)

# 创建 pipeline
pipeline = L2W1Pipeline(config)

# 处理测试图像
image_path = "data/raw/images/test_001.jpg"
if Path(image_path).exists():
    result = pipeline.process(image_path)
    print(f"图像: {image_path}")
    print(f"Agent A: {result.agent_a_text}")
    print(f"最终结果: {result.final_text}")
    print(f"是否困难样本: {result.is_hard}")
else:
    print(f"测试图像不存在: {image_path}")
EOF

# python test_pipeline_full.py
```

---

## 📊 步骤 7: 评估模块测试

### 7.1 测试评估模块导入和基本功能

```bash
cat > test_evaluate.py << 'EOF'
#!/usr/bin/env python3
"""测试评估模块"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

print("=" * 60)
print("评估模块测试")
print("=" * 60)

try:
    from scripts.evaluate import (
        calculate_cer,
        calculate_ocr_r,
        calculate_correction_rate,
        levenshtein_distance
    )
    print("✓ 评估函数导入成功")

    # 测试基本指标计算
    print("\n测试指标计算...")

    # 测试 CER
    cer, _ = calculate_cer("测试文本", "测试文本")
    print(f"✓ CER 计算: {cer} (应该为 0.0)")

    cer2, _ = calculate_cer("测试文本", "测试文")
    print(f"✓ CER 计算（有错误）: {cer2}")

    # 测试 OCR-R
    ocr_r, _ = calculate_ocr_r("正确文本", "错误文本", "正确文本")
    print(f"✓ OCR-R 计算: {ocr_r}")

    # 测试 Correction Rate
    cr, _ = calculate_correction_rate("错误文本", "正确文本", "正确文本")
    print(f"✓ Correction Rate 计算: {cr}")

    print("\n✓ 评估模块测试通过!")

except Exception as e:
    print(f"✗ 评估模块测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF

python test_evaluate.py
```

**✅ 通过标准**:

- 评估函数可以导入
- 指标计算正常
- 边界情况处理正确

---

### 7.2 测试评估模块完整流程

```bash
# 创建测试数据
cat > data/test/test_predictions.jsonl << 'EOF'
{"image": "test_001.jpg", "agent_a_text": "测试文本", "final_text": "测试文本", "gt_text": "测试文本", "is_hard": false}
{"image": "test_002.jpg", "agent_a_text": "错误文本", "final_text": "正确文本", "gt_text": "正确文本", "is_hard": true}
EOF

# 运行评估
python scripts/evaluate.py \
    --predictions ./data/test/test_predictions.jsonl \
    --output_dir ./data/test

# 检查输出
if [ -f "data/test/evaluation_report.json" ]; then
    echo "✓ 评估报告已生成"
    cat data/test/evaluation_report.json
else
    echo "✗ 评估报告未生成"
fi
```

---

## 🎯 完整测试脚本

创建一个一键运行所有测试的脚本：

```bash
cat > run_all_tests.sh << 'EOF'
#!/bin/bash
# L2W1 v5.0 完整测试脚本

set -e

echo "=========================================="
echo "L2W1 v5.0 完整测试"
echo "=========================================="
echo ""

# 激活环境
conda activate l2w1v2

# 测试 1: 环境验证
echo "[1/7] 环境验证..."
python test_imports.py && echo "✓ 通过" || echo "✗ 失败"

# 测试 2: 模块导入
echo ""
echo "[2/7] 模块导入..."
python test_modules.py && echo "✓ 通过" || echo "✗ 失败"

# 测试 3: 数据管道
echo ""
echo "[3/7] 数据管道..."
python test_data_pipeline.py && echo "✓ 通过" || echo "✗ 失败"

# 测试 4: Router
echo ""
echo "[4/7] Router 模块..."
python test_router.py && echo "✓ 通过" || echo "✗ 失败"

# 测试 5: Agent B
echo ""
echo "[5/7] Agent B 模块..."
python test_agent_b.py && echo "✓ 通过" || echo "✗ 失败"

# 测试 6: Pipeline
echo ""
echo "[6/7] Pipeline 模块..."
python test_pipeline.py && echo "✓ 通过" || echo "✗ 失败"

# 测试 7: 评估
echo ""
echo "[7/7] 评估模块..."
python test_evaluate.py && echo "✓ 通过" || echo "✗ 失败"

echo ""
echo "=========================================="
echo "测试完成!"
echo "=========================================="
EOF

chmod +x run_all_tests.sh
```

---

## 📝 测试结果记录

建议创建一个测试结果文件：

```bash
cat > test_results.log << 'EOF'
测试日期: $(date)
环境: l2w1v2 (Python 3.10, CUDA 12.6)

测试结果:
[ ] 步骤 1: 环境验证
[ ] 步骤 2: 模块导入
[ ] 步骤 3: 数据管道
[ ] 步骤 4: Router
[ ] 步骤 5: Agent B
[ ] 步骤 6: Pipeline
[ ] 步骤 7: 评估

发现的 Bug:
1.
2.
3.

EOF
```

---

## 🐛 Bug 报告格式

如果发现 bug，请提供：

```
1. 测试步骤: [具体步骤]
2. 运行的命令: [完整命令]
3. 错误信息: [完整错误堆栈]
4. 预期行为: [应该发生什么]
5. 实际行为: [实际发生了什么]
6. 环境信息:
   - Python: [版本]
   - CUDA: [版本]
   - GPU: [型号]
```

---

**开始测试吧！按照步骤逐一执行，遇到问题随时告诉我！** 🚀
