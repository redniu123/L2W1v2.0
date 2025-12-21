# L2W1 v5.0 Linux 服务器部署指南

**目标平台**: Ubuntu 20.04+ / CentOS 7+ / Debian 11+  
**显卡**: NVIDIA RTX 2080Ti (11GB 显存)  
**CUDA**: 11.8  
**Python**: 3.8 - 3.11 (推荐 3.10)

---

## 📋 目录

1. [系统要求](#系统要求)
2. [前置准备](#前置准备)
3. [快速安装](#快速安装)
4. [手动安装](#手动安装)
5. [环境验证](#环境验证)
6. [常见问题](#常见问题)
7. [VS Code 配置](#vs-code-配置)

---

## 🖥️ 系统要求

### 硬件要求
- **GPU**: NVIDIA RTX 2080Ti (11GB) 或更高
- **CPU**: 4 核心以上
- **内存**: 32GB+ (推荐)
- **磁盘**: 50GB+ 可用空间

### 软件要求
- **操作系统**: Ubuntu 20.04+, CentOS 7+, Debian 11+
- **CUDA**: 11.8 (RTX 2080Ti 支持 CUDA 11.0 - 11.8)
- **cuDNN**: 8.9.2 (CUDA 11.8 配套)
- **Python**: 3.8, 3.9, 3.10, 3.11 (推荐 3.10)
- **NVIDIA 驱动**: 520.61.05 或更高

---

## 🔧 前置准备

### 1. 检查系统环境

```bash
# 检查 Python 版本
python3 --version  # 应该 >= 3.8

# 检查 CUDA 版本
nvcc --version
nvidia-smi  # 查看 GPU 和驱动信息

# 检查磁盘空间
df -h
```

### 2. 安装系统依赖 (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    python3-dev \
    python3-pip \
    python3-venv
```

### 3. 安装 NVIDIA 驱动和 CUDA (如未安装)

#### 方法 1: 使用 apt (Ubuntu)

```bash
# 添加 NVIDIA 仓库
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# 安装 NVIDIA 驱动
sudo apt-get update
sudo apt-get install -y nvidia-driver-520 nvidia-cuda-toolkit-11-8
```

#### 方法 2: 使用 NVIDIA 官方安装器

```bash
# 下载并安装 CUDA Toolkit 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run
```

**验证安装**:
```bash
nvidia-smi
nvcc --version
```

---

## 🚀 快速安装

### 使用自动安装脚本

```bash
# 1. 克隆或进入项目目录
cd L2W1

# 2. 运行安装脚本
bash install_linux.sh

# 3. 激活虚拟环境
source l2w1_env/bin/activate

# 4. 验证安装
python check_env.py
```

---

## 📦 手动安装

### 步骤 1: 创建虚拟环境

```bash
# 创建虚拟环境
python3 -m venv l2w1_env

# 激活虚拟环境
source l2w1_env/bin/activate

# 升级 pip
pip install --upgrade pip setuptools wheel
```

### 步骤 2: 安装 PyTorch (CUDA 11.8)

```bash
# 安装 PyTorch 2.1.2 (CUDA 11.8)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

# 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### 步骤 3: 安装 PaddlePaddle

```bash
# 优先尝试 GPU 版本
pip install paddlepaddle-gpu>=2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 如果失败，使用 CPU 版本
# pip install paddlepaddle>=2.6.0
```

### 步骤 4: 安装项目依赖

```bash
# 从 requirements.txt 安装
pip install -r requirements.txt
```

**注意**: 如果遇到版本冲突，可以分步安装：

```bash
# 核心依赖
pip install transformers>=4.40.0 peft>=0.7.0 accelerate>=0.25.0

# 量化库
pip install bitsandbytes>=0.41.0

# 数据处理
pip install opencv-python pillow numpy pandas

# 可视化
pip install matplotlib seaborn

# 其他工具
pip install tqdm pyyaml tensorboard editdistance
```

### 步骤 5: (可选) 安装 Flash Attention 2

```bash
# 需要编译环境 (build-essential, CUDA Toolkit)
pip install flash-attn --no-build-isolation

# 如果安装失败，可以跳过（代码会自动回退到标准注意力机制）
```

---

## ✅ 环境验证

### 1. 运行环境检查脚本

```bash
python check_env.py
```

### 2. 手动验证关键依赖

```python
# test_env.py
import torch
import paddle
import transformers
import peft
import bitsandbytes

print("=" * 60)
print("L2W1 v5.0 环境验证")
print("=" * 60)

# PyTorch
print(f"\n[PyTorch]")
print(f"  版本: {torch.__version__}")
print(f"  CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA 版本: {torch.version.cuda}")
    print(f"  GPU 数量: {torch.cuda.device_count()}")
    print(f"  GPU 名称: {torch.cuda.get_device_name(0)}")
    print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# PaddlePaddle
print(f"\n[PaddlePaddle]")
print(f"  版本: {paddle.__version__}")
print(f"  CUDA 可用: {paddle.device.is_compiled_with_cuda()}")

# Transformers
print(f"\n[Transformers]")
print(f"  版本: {transformers.__version__}")

# PEFT
print(f"\n[PEFT]")
print(f"  版本: {peft.__version__}")

# BitsAndBytes
print(f"\n[BitsAndBytes]")
try:
    import bitsandbytes as bnb
    print(f"  版本: {bnb.__version__}")
    print("  ✓ 4-bit 量化支持可用")
except Exception as e:
    print(f"  ✗ 错误: {e}")

print("\n" + "=" * 60)
print("验证完成!")
print("=" * 60)
```

运行验证:
```bash
python test_env.py
```

---

## 💻 VS Code 配置

### 1. 安装 VS Code 扩展

在 VS Code 中安装以下扩展:
- **Python** (ms-python.python)
- **Pylance** (ms-python.vscode-pylance)
- **Jupyter** (ms-toolsai.jupyter)

### 2. 配置 Python 解释器

1. 按 `Ctrl+Shift+P` 打开命令面板
2. 输入 "Python: Select Interpreter"
3. 选择虚拟环境: `./l2w1_env/bin/python`

或者创建 `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/l2w1_env/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black"
}
```

### 3. 配置远程开发 (VS Code Server)

如果使用远程服务器:

1. 安装 **Remote - SSH** 扩展
2. 连接到服务器: `ssh user@server-ip`
3. 在远程服务器上安装 Python 扩展
4. 选择远程解释器路径

### 4. 配置终端自动激活虚拟环境

在 `~/.bashrc` 或 `~/.zshrc` 中添加:

```bash
# L2W1 虚拟环境自动激活
if [ -d "$HOME/path/to/L2W1/l2w1_env" ]; then
    source $HOME/path/to/L2W1/l2w1_env/bin/activate
fi
```

---

## ❓ 常见问题

### Q1: PyTorch CUDA 不可用

**问题**: `torch.cuda.is_available()` 返回 `False`

**解决方案**:
1. 检查 NVIDIA 驱动: `nvidia-smi`
2. 检查 CUDA 版本: `nvcc --version`
3. 重新安装 PyTorch (确保 CUDA 版本匹配):
   ```bash
   pip uninstall torch torchvision torchaudio
   pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
       --index-url https://download.pytorch.org/whl/cu118
   ```

### Q2: bitsandbytes 安装失败

**问题**: `ImportError: No module named 'bitsandbytes'`

**解决方案**:
1. 确保安装了编译工具: `sudo apt-get install build-essential`
2. 确保 CUDA Toolkit 已安装: `nvcc --version`
3. 尝试从源码安装:
   ```bash
   pip install git+https://github.com/TimDettmers/bitsandbytes.git
   ```
4. 如果仍失败，可以跳过 bitsandbytes (QLoRA 功能将受限)

### Q3: Flash Attention 2 编译失败

**问题**: Flash Attention 2 安装失败

**解决方案**:
- 这是可选的，代码会自动回退到标准注意力机制
- 如果确实需要，确保:
  1. CUDA Toolkit 已安装
  2. 编译工具已安装: `build-essential`, `cmake`
  3. 有足够的编译时间 (5-10 分钟)

### Q4: PaddlePaddle 导入错误

**问题**: `ImportError: libcudart.so.xxx: cannot open shared object file`

**解决方案**:
1. 检查 CUDA 库路径:
   ```bash
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```
2. 添加到 `~/.bashrc` 永久生效
3. 如果使用 CPU 版本，忽略此错误

### Q5: 显存不足 (OOM)

**问题**: 训练时出现 CUDA OOM 错误

**解决方案**:
1. 确保使用 4-bit 量化: `use_4bit=True`
2. 减小 batch size: `per_device_train_batch_size=1`
3. 启用梯度累积: `gradient_accumulation_steps=8`
4. 启用梯度检查点: `gradient_checkpointing=True`
5. 使用更小的模型 (如果可能)

### Q6: 虚拟环境激活后提示符不变

**问题**: 激活虚拟环境后，命令行提示符没有 `(l2w1_env)` 前缀

**解决方案**:
```bash
# 使用 source 激活 (而不是 bash)
source l2w1_env/bin/activate

# 或者直接使用完整路径
l2w1_env/bin/python your_script.py
```

---

## 📝 快速参考

### 激活/退出虚拟环境

```bash
# 激活
source l2w1_env/bin/activate

# 退出
deactivate
```

### 更新依赖

```bash
# 更新所有包到最新兼容版本
pip install --upgrade -r requirements.txt
```

### 查看已安装包

```bash
pip list
pip show <package_name>
```

### 卸载所有依赖

```bash
# 删除虚拟环境
deactivate
rm -rf l2w1_env

# 重新安装
bash install_linux.sh
```

---

## 🔗 相关文档

- [项目结构说明](./PROJECT_STRUCTURE.md)
- [代码审计报告](./CODE_AUDIT_REPORT.md)
- [加固变更记录](./HARDENING_CHANGELOG.md)
- [环境检查脚本](./check_env.py)

---

**部署完成后，请运行 `python check_env.py` 验证所有组件是否正常工作!** ✅

