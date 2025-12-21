# L2W1 v5.0 Conda 环境安装指南

## 📋 前提条件

- ✅ Anaconda/Miniconda 已安装
- ✅ 已创建 Conda 环境 `l2w1v2`
- ✅ Python 3.10
- ✅ CUDA 12.6
- ✅ GPU 可用

## 🚀 快速安装

### 方法 1: 使用安装脚本（推荐）

```bash
# 1. 进入项目目录
cd /path/to/L2W1

# 2. 运行安装脚本
bash install_conda.sh
```

脚本会自动：
- ✅ 检查并激活 conda 环境 `l2w1v2`
- ✅ 安装 PyTorch (CUDA 12.1 兼容 CUDA 12.6)
- ✅ 安装 PaddlePaddle (GPU 版本)
- ✅ 安装所有项目依赖
- ✅ 验证安装结果

### 方法 2: 手动安装

```bash
# 1. 激活 conda 环境
conda activate l2w1v2

# 2. 升级 pip
pip install --upgrade pip setuptools wheel

# 3. 安装 PyTorch (CUDA 12.1，兼容 CUDA 12.6)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. 安装 PaddlePaddle (GPU 版本)
pip install paddlepaddle-gpu>=2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 5. 安装项目依赖
pip install -r requirements.txt
```

## ⚙️ 关键配置说明

### PyTorch 版本选择

由于 PyTorch 官方暂未发布 CUDA 12.6 的专用版本，我们使用 **CUDA 12.1 版本**，因为：
- ✅ CUDA 12.6 向下兼容 CUDA 12.1
- ✅ 所有功能正常工作
- ✅ 性能无影响

### 安装源说明

- **PyTorch**: 使用官方源 `https://download.pytorch.org/whl/cu121`
- **PaddlePaddle**: 优先使用清华镜像（国内更快）
- **其他包**: 使用默认 PyPI 源

## 🔍 验证安装

### 快速验证

```bash
# 激活环境
conda activate l2w1v2

# 运行验证脚本
python verify_installation.py
```

### 手动验证

```bash
# 检查 PyTorch 和 CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# 检查 PaddlePaddle
python -c "import paddle; print(f'PaddlePaddle: {paddle.__version__}')"

# 检查关键依赖
python -c "import transformers, peft, bitsandbytes; print('✓ 核心依赖正常')"
```

## 📦 依赖列表

核心依赖：
- **PyTorch** 2.1.2+ (CUDA 12.1)
- **PaddlePaddle** 2.6.0+ (GPU)
- **Transformers** 4.40.0+
- **PEFT** 0.7.0+ (LoRA/QLoRA)
- **bitsandbytes** 0.41.0+ (4-bit 量化)
- **accelerate** 0.25.0+

完整列表请查看 `requirements.txt`

## ⚠️ 常见问题

### Q1: Conda 环境激活失败

**问题**: `conda: command not found`

**解决**:
```bash
# 初始化 conda
source ~/anaconda3/etc/profile.d/conda.sh
# 或
source ~/miniconda3/etc/profile.d/conda.sh

# 然后运行安装脚本
bash install_conda.sh
```

### Q2: bitsandbytes 安装失败

**问题**: `ImportError: No module named 'bitsandbytes'`

**原因**: bitsandbytes 需要编译，依赖 CUDA Toolkit

**解决**:
```bash
# 确保 CUDA Toolkit 已安装
nvcc --version

# 如果未安装，通过 conda 安装（推荐）
conda install -c conda-forge cudatoolkit-dev

# 重新安装 bitsandbytes
pip install bitsandbytes>=0.41.0
```

### Q3: PyTorch CUDA 不可用

**问题**: `torch.cuda.is_available()` 返回 `False`

**解决**:
```bash
# 检查 CUDA 驱动
nvidia-smi

# 检查 PyTorch 版本
python -c "import torch; print(torch.__version__)"

# 如果版本不对，重新安装
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Q4: PaddlePaddle 找不到 CUDA 库

**问题**: `libcudart.so.xxx: cannot open shared object file`

**解决**:
```bash
# 方法 1: 设置 LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# 方法 2: 使用 conda 安装的 CUDA（推荐）
conda install -c conda-forge cudatoolkit
```

## 🔄 更新依赖

```bash
# 激活环境
conda activate l2w1v2

# 更新所有包
pip install --upgrade -r requirements.txt

# 或更新特定包
pip install --upgrade <package_name>
```

## 📝 使用提示

1. **每次使用前激活环境**:
   ```bash
   conda activate l2w1v2
   ```

2. **验证环境已激活**:
   ```bash
   echo $CONDA_DEFAULT_ENV  # 应该显示 l2w1v2
   ```

3. **在 VS Code 中选择解释器**:
   - 按 `Ctrl+Shift+P`
   - 输入 "Python: Select Interpreter"
   - 选择: `~/anaconda3/envs/l2w1v2/bin/python`

## 🎯 下一步

安装完成后，可以：

1. ✅ 运行环境验证: `python verify_installation.py`
2. ✅ 查看运行指南: 准备运行代码
3. ✅ 开始使用 L2W1 v5.0!

---

**安装完成后，请运行 `python verify_installation.py` 验证所有组件！** ✅

