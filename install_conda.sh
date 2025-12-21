#!/bin/bash
# ============================================================================
# L2W1 v5.0 Conda 环境一键安装脚本
# 目标: Anaconda 环境 l2w1v2, CUDA 12.6, Python 3.10, GPU
# ============================================================================

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印函数
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# 配置
CONDA_ENV_NAME="l2w1v2"
CUDA_VERSION="12.1"  # CUDA 12.6 向下兼容 12.1

# ============================================================================
# 1. 检查 Conda 环境
# ============================================================================

print_info "开始 L2W1 v5.0 环境安装 (Conda 环境: $CONDA_ENV_NAME)..."

# 检查 conda 命令
if ! command -v conda &> /dev/null; then
    print_error "未找到 conda 命令，请确保 Anaconda/Miniconda 已安装并添加到 PATH"
    print_info "提示: 可以尝试 'source ~/anaconda3/etc/profile.d/conda.sh'"
    exit 1
fi

print_info "Conda 已找到: $(conda --version)"

# 初始化 conda（如果需要）
if ! conda info --envs &> /dev/null; then
    print_warn "正在初始化 conda..."
    eval "$(conda shell.bash hook)"
fi

# 检查环境是否存在
print_step "检查 Conda 环境: $CONDA_ENV_NAME"
if conda env list | grep -q "^${CONDA_ENV_NAME}\s"; then
    print_info "环境 $CONDA_ENV_NAME 已存在"
else
    print_error "环境 $CONDA_ENV_NAME 不存在！"
    print_info "请先创建环境: conda create -n $CONDA_ENV_NAME python=3.10"
    exit 1
fi

# 激活环境
print_step "激活 Conda 环境: $CONDA_ENV_NAME"
eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME

if [ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV_NAME" ]; then
    print_error "环境激活失败！当前环境: $CONDA_DEFAULT_ENV"
    exit 1
fi

print_info "当前环境: $(conda info --envs | grep '*' | awk '{print $1}')"
print_info "Python 版本: $(python --version)"

# ============================================================================
# 2. 检查系统环境
# ============================================================================

print_step "检查系统环境..."

# 检查 CUDA
if command -v nvcc &> /dev/null; then
    CUDA_VER=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d, -f1)
    print_info "CUDA 版本: $CUDA_VER"
else
    print_warn "未找到 nvcc，但可能通过 conda 安装的 CUDA 可用"
fi

# 检查 GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
    print_info "GPU: $GPU_NAME (${GPU_MEMORY}MB)"
else
    print_warn "未找到 nvidia-smi，GPU 可能不可用"
fi

# ============================================================================
# 3. 升级 pip 和基础工具
# ============================================================================

print_step "升级 pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel -q

# ============================================================================
# 4. 安装 PyTorch (CUDA 12.1 兼容 CUDA 12.6)
# ============================================================================

print_step "安装 PyTorch (CUDA $CUDA_VERSION 兼容版本)..."

# 检查是否已安装 PyTorch
if python -c "import torch" 2>/dev/null; then
    TORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
    print_warn "PyTorch 已安装: $TORCH_VER"
    read -p "是否重新安装 PyTorch? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "卸载现有 PyTorch..."
        pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
    else
        print_info "跳过 PyTorch 安装"
        SKIP_TORCH=true
    fi
fi

if [ "$SKIP_TORCH" != "true" ]; then
    print_info "从 PyTorch 官方源安装 (CUDA $CUDA_VERSION)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # 验证安装
    print_info "验证 PyTorch 安装..."
    python -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 版本: {torch.version.cuda}')
    print(f'GPU 数量: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('警告: CUDA 不可用')
" || {
        print_error "PyTorch 验证失败"
        exit 1
    }
fi

# ============================================================================
# 5. 安装 PaddlePaddle (GPU 版本)
# ============================================================================

print_step "安装 PaddlePaddle (GPU 版本)..."

# 检查是否已安装
if python -c "import paddle" 2>/dev/null; then
    print_warn "PaddlePaddle 已安装，跳过..."
else
    print_info "安装 PaddlePaddle GPU 版本..."
    # 优先使用清华镜像（国内更快）
    pip install paddlepaddle-gpu>=2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple || \
    pip install paddlepaddle-gpu>=2.6.0 || \
    pip install paddlepaddle>=2.6.0
    
    # 验证
    python -c "import paddle; print(f'PaddlePaddle 版本: {paddle.__version__}')" || print_warn "PaddlePaddle 验证失败"
fi

# ============================================================================
# 6. 安装项目依赖
# ============================================================================

print_step "安装项目依赖..."

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"

if [ ! -f "$REQUIREMENTS_FILE" ]; then
    print_error "未找到 requirements.txt: $REQUIREMENTS_FILE"
    exit 1
fi

print_info "从 requirements.txt 安装依赖..."

# 先安装一些可能有编译依赖的包
print_info "安装基础依赖..."
pip install numpy>=1.24.0 setuptools wheel -q

# 安装 transformers 相关（需要先安装）
print_info "安装 Transformers 生态..."
pip install transformers>=4.40.0 qwen-vl-utils sentencepiece -q

# 安装 bitsandbytes（需要编译）
print_step "安装 bitsandbytes (4-bit 量化支持)..."
if python -c "import bitsandbytes" 2>/dev/null; then
    print_warn "bitsandbytes 已安装，跳过..."
else
    print_info "安装 bitsandbytes (可能需要编译，请耐心等待)..."
    pip install bitsandbytes>=0.41.0 || {
        print_warn "bitsandbytes 安装失败，QLoRA 功能将受限"
        print_info "提示: 确保已安装 CUDA Toolkit 和编译工具"
    }
fi

# 安装 peft 和 accelerate
print_info "安装 PEFT 和 Accelerate..."
pip install peft>=0.7.0 accelerate>=0.25.0 -q

# 安装其他依赖（排除已安装的 PyTorch）
print_info "安装其他依赖..."
grep -v "^torch" "$REQUIREMENTS_FILE" | \
grep -v "^torchvision" | \
grep -v "^torchaudio" | \
grep -v "^paddlepaddle" | \
grep -v "^#" | \
grep -v "^$" | \
pip install -r /dev/stdin || {
    print_warn "部分依赖安装可能失败，请检查错误信息"
}

# ============================================================================
# 7. 安装可选依赖
# ============================================================================

print_step "安装可选依赖..."

# Flash Attention 2 (可选)
read -p "是否安装 Flash Attention 2? (需要编译，可能需要 5-10 分钟) (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if python -c "import flash_attn" 2>/dev/null; then
        print_warn "flash-attn 已安装，跳过..."
    else
        print_info "安装 Flash Attention 2..."
        if command -v nvcc &> /dev/null; then
            pip install flash-attn --no-build-isolation || \
            print_warn "Flash Attention 2 安装失败，将使用标准注意力机制"
        else
            print_warn "未找到 nvcc，跳过 Flash Attention 2 安装"
        fi
    fi
else
    print_info "跳过 Flash Attention 2 安装"
fi

# ============================================================================
# 8. 验证安装
# ============================================================================

print_step "验证核心依赖..."

python << 'EOF'
import sys

def check_import(module_name, package_name=None, critical=False):
    try:
        mod = __import__(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  ✓ {package_name or module_name}: {version}")
        return True
    except ImportError as e:
        status = "✗" if critical else "⚠"
        print(f"  {status} {package_name or module_name}: 未安装")
        if critical:
            print(f"    错误: {e}")
        return False

print("\n核心依赖检查:")
critical_checks = [
    ("torch", "PyTorch", True),
    ("paddle", "PaddlePaddle", True),
    ("transformers", "Transformers", True),
    ("peft", "PEFT", True),
]

all_ok = True
for mod, name, critical in critical_checks:
    if not check_import(mod, name, critical):
        if critical:
            all_ok = False

optional_checks = [
    ("bitsandbytes", "BitsAndBytes"),
    ("flash_attn", "Flash Attention"),
    ("cv2", "OpenCV"),
    ("numpy", "NumPy"),
    ("pandas", "Pandas"),
    ("matplotlib", "Matplotlib"),
    ("seaborn", "Seaborn"),
]

print("\n可选依赖检查:")
for mod, name in optional_checks:
    check_import(mod, name, False)

if all_ok:
    print("\n✓ 所有核心依赖安装成功!")
else:
    print("\n✗ 部分核心依赖安装失败!")
    sys.exit(1)
EOF

INSTALL_STATUS=$?

# ============================================================================
# 9. 运行环境检查脚本（如果存在）
# ============================================================================

if [ -f "$SCRIPT_DIR/verify_installation.py" ]; then
    print_step "运行详细环境检查..."
    python "$SCRIPT_DIR/verify_installation.py" || print_warn "环境检查脚本执行失败"
fi

# ============================================================================
# 完成
# ============================================================================

echo ""
if [ $INSTALL_STATUS -eq 0 ]; then
    print_info "=========================================="
    print_info "✓ 安装完成!"
    print_info "=========================================="
    echo ""
    print_info "环境信息:"
    print_info "  Conda 环境: $CONDA_ENV_NAME"
    print_info "  Python: $(python --version)"
    print_info "  当前目录: $(pwd)"
    echo ""
    print_info "后续步骤:"
    print_info "1. 确保环境已激活: conda activate $CONDA_ENV_NAME"
    print_info "2. 运行验证: python verify_installation.py"
    print_info "3. 开始使用 L2W1 v5.0!"
    echo ""
    print_info "提示: 每次使用前记得激活环境:"
    print_info "  conda activate $CONDA_ENV_NAME"
else
    print_error "安装过程中出现错误，请检查上面的错误信息"
    exit 1
fi

