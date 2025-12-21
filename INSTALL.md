# L2W1 v5.0 环境安装指南

**部署环境**: Linux Server (Ubuntu 20.04+)  
**GPU**: NVIDIA RTX 2080Ti (22GB 显存)  
**CUDA**: 11.8  
**Python**: 3.9-3.11 (推荐 3.10)

---

## 📋 前置条件

### 1. 系统要求

```bash
# 检查系统版本
cat /etc/os-release

# 检查 GPU
nvidia-smi

# 检查 CUDA 版本
nvcc --version
# 或
cat /usr/local/cuda/version.txt
```

### 2. NVIDIA 驱动和 CUDA

**RTX 2080Ti 要求**:

- NVIDIA Driver: >= 450.80.02
- CUDA: 11.8 (推荐) 或 11.7
- cuDNN: 8.2+ (用于 PaddlePaddle)

**安装 CUDA 11.8** (如果未安装):

```bash
# 下载 CUDA 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run

# 安装
sudo sh cuda_11.8.0_520.61.05_linux.run

# 配置环境变量 (添加到 ~/.bashrc)
export PATH=/usr/local/cuda-11.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

# 重新加载
source ~/.bashrc
```

---

## 🐍 Python 环境

### 方法 1: 使用 Conda (推荐)

```bash
# 创建 conda 环境
conda create -n l2w1 python=3.10 -y
conda activate l2w1

# 安装 PyTorch (CUDA 11.8)
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 验证 PyTorch
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

### 方法 2: 使用 venv

```bash
# 创建虚拟环境
python3.10 -m venv venv
source venv/bin/activate

# 升级 pip
pip install --upgrade pip setuptools wheel
```

---

## 📦 安装依赖

### 步骤 1: 安装 PyTorch (如果使用 venv)

```bash
# CUDA 11.8 版本
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# 验证
python -c "import torch; print(torch.cuda.is_available())"
```

### 步骤 2: 安装 PaddlePaddle

```bash
# GPU 版本 (CUDA 11.2+)
python -m pip install paddlepaddle-gpu==2.6.1 -i https://mirror.baidu.com/pypi/simple

# 验证
python -c "import paddle; paddle.utils.run_check()"
```

**注意**:

- 如果安装失败，检查 CUDA 和 cuDNN 版本
- PaddlePaddle 需要 cuDNN 8.2+，可以通过 `conda install cudnn -c conda-forge` 安装

### 步骤 3: 安装其他依赖

```bash
# 进入项目目录
cd L2W1

# 安装基础依赖
pip install -r requirements.txt

# 如果需要开发工具
pip install -r requirements-dev.txt
```

### 步骤 4: 安装 BitsAndBytes (4-bit 量化)

```bash
# 方法 1: 直接安装 (自动检测 CUDA)
pip install bitsandbytes

# 方法 2: 如果失败，从源码编译
# git clone https://github.com/TimDettmers/bitsandbytes.git
# cd bitsandbytes
# CUDA_VERSION=118 make cuda11x
# python setup.py install
```

**验证**:

```python
python -c "import bitsandbytes as bnb; print('BitsAndBytes installed successfully')"
```

### 步骤 5: 安装 Flash Attention 2 (可选但推荐)

```bash
# 方法 1: 直接安装 (需要 CUDA 编译器)
pip install flash-attn --no-build-isolation

# 方法 2: 如果失败，从源码编译
# pip install flash-attn==2.5.0
```

**注意**: Flash Attention 2 需要编译，确保已安装 `ninja`:

```bash
pip install ninja
```

---

## ✅ 验证安装

### 1. 检查核心库

```bash
cd L2W1

python -c "
import torch
import paddle
import transformers
import peft
import bitsandbytes as bnb
import numpy as np
import cv2
from PIL import Image

print('=' * 60)
print('L2W1 环境验证')
print('=' * 60)
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
print(f'GPU Count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU Name: {torch.cuda.get_device_name(0)}')
print(f'PaddlePaddle: {paddle.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'PEFT: {peft.__version__}')
print(f'BitsAndBytes: {bnb.__version__}')
print(f'NumPy: {np.__version__}')
print(f'OpenCV: {cv2.__version__}')
print('=' * 60)
print('✅ 所有核心库安装成功!')
print('=' * 60)
"
```

### 2. 测试 GPU 显存

```python
python -c "
import torch
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    # 分配 1GB 显存测试
    x = torch.randn(1024, 1024, 256).to(device)
    print(f'✅ GPU 可用，已分配 {x.element_size() * x.nelement() / 1024**3:.2f} GB 显存')
    del x
    torch.cuda.empty_cache()
    print(f'✅ 显存已释放，当前使用: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB')
else:
    print('❌ GPU 不可用')
"
```

### 3. 测试 PaddleOCR 集成

```python
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute().parent))

from L2W1.modules.paddle_engine import TextRecognizerWithLogits

print('测试 PaddleOCR 集成...')
# 这里可以添加实际的测试代码
print('✅ PaddleOCR 模块导入成功')
"
```

---

## 🔧 常见问题

### Q1: BitsAndBytes 安装失败

**原因**: CUDA 版本不匹配或缺少编译工具

**解决方案**:

```bash
# 安装编译工具
sudo apt-get update
sudo apt-get install build-essential

# 检查 CUDA 版本
nvcc --version

# 重新安装
pip install bitsandbytes --no-cache-dir
```

### Q2: Flash Attention 2 安装失败

**原因**: 需要编译，CUDA 版本不匹配

**解决方案**:

```bash
# 方案 1: 跳过 Flash Attention (会使用普通 attention)
# 在代码中设置 use_flash_attention=False

# 方案 2: 安装编译依赖
pip install ninja packaging wheel
pip install flash-attn --no-build-isolation
```

### Q3: PaddlePaddle 无法检测 GPU

**原因**: CUDA/cuDNN 版本不匹配

**解决方案**:

```bash
# 检查 PaddlePaddle 安装
python -c "import paddle; paddle.utils.run_check()"

# 如果失败，重新安装匹配的版本
pip uninstall paddlepaddle-gpu
pip install paddlepaddle-gpu==2.6.1.post118 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

### Q4: 显存不足 (OOM)

**RTX 2080Ti 实际显存为 11GB**，如果遇到 OOM:

1. 确保使用 4-bit 量化:

   ```python
   use_4bit=True  # Agent B 配置
   ```

2. 减小 batch size:

   ```python
   batch_size=2  # 训练时
   ```

3. 启用梯度检查点:
   ```python
   gradient_checkpointing=True
   ```

### Q5: 版本冲突

**解决方案**: 使用虚拟环境隔离

```bash
# 重新创建干净环境
conda create -n l2w1_clean python=3.10 -y
conda activate l2w1_clean

# 按顺序安装
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install paddlepaddle-gpu==2.6.1 -i https://mirror.baidu.com/pypi/simple
pip install -r requirements.txt
```

---

## 📝 VS Code 配置

### 1. Python 解释器

在 VS Code 中:

1. 按 `Ctrl+Shift+P`
2. 输入 "Python: Select Interpreter"
3. 选择创建的虚拟环境: `./venv/bin/python` 或 `~/anaconda3/envs/l2w1/bin/python`

### 2. 推荐扩展

- Python (Microsoft)
- Pylance (Microsoft)
- Jupyter (Microsoft)
- Python Docstring Generator

### 3. 工作区设置

创建 `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "python.analysis.typeCheckingMode": "basic"
}
```

---

## 🚀 快速开始

```bash
# 1. 激活环境
conda activate l2w1  # 或 source venv/bin/activate

# 2. 运行数据管道
python scripts/data_pipeline.py --data_dir ./data/raw --output_dir ./data/sft

# 3. 训练 Agent B
python scripts/train_agent_b.py --data_path ./data/sft/agent_b_train.jsonl

# 4. 评估
python scripts/evaluate.py --predictions ./data/test/inference_results.jsonl
```

---

**安装完成!** 🎉

如有问题，请查看 `CODE_AUDIT_REPORT.md` 或提交 Issue。
