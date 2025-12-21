#!/bin/bash
# ============================================================================
# L2W1 v5.0 Linux 环境安装脚本
# 目标: Ubuntu 20.04+, RTX 2080Ti (11GB), CUDA 11.8
# ============================================================================

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        return 1
    fi
    return 0
}

# ============================================================================
# 1. 环境检查
# ============================================================================

print_info "开始 L2W1 v5.0 环境安装..."

# 检查 Python 版本
print_info "检查 Python 版本..."
if check_command python3; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
    print_info "Python 版本: $(python3 --version)"
    if [[ $(echo "$PYTHON_VERSION >= 3.8" | bc -l 2>/dev/null || echo "0") == "0" ]]; then
        print_error "需要 Python 3.8 或更高版本"
        exit 1
    fi
else
    print_error "未找到 python3，请先安装 Python 3.8+"
    exit 1
fi

# 检查 pip
print_info "检查 pip..."
if ! check_command pip3; then
    print_warn "pip3 未找到，正在安装..."
    python3 -m ensurepip --upgrade
fi

# 检查 CUDA
print_info "检查 CUDA..."
if check_command nvcc; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d, -f1)
    print_info "CUDA 版本: $CUDA_VERSION"
else
    print_warn "未找到 CUDA，某些功能可能受限"
fi

# 检查 GPU
print_info "检查 GPU..."
if check_command nvidia-smi; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
    print_info "GPU: $GPU_NAME (${GPU_MEMORY}MB)"
else
    print_warn "未找到 NVIDIA GPU 或驱动未安装"
fi

# ============================================================================
# 2. 创建虚拟环境
# ============================================================================

ENV_NAME="l2w1_env"
if [ -d "$ENV_NAME" ]; then
    print_warn "虚拟环境 $ENV_NAME 已存在"
    read -p "是否删除并重新创建? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "删除现有虚拟环境..."
        rm -rf $ENV_NAME
    else
        print_info "使用现有虚拟环境"
        source $ENV_NAME/bin/activate
        skip_env_creation=true
    fi
fi

if [ "$skip_env_creation" != "true" ]; then
    print_info "创建虚拟环境: $ENV_NAME"
    python3 -m venv $ENV_NAME
    source $ENV_NAME/bin/activate
fi

# 升级 pip, setuptools, wheel
print_info "升级 pip, setuptools, wheel..."
pip install --upgrade pip setuptools wheel

# ============================================================================
# 3. 安装 PyTorch (CUDA 11.8)
# ============================================================================

print_info "安装 PyTorch (CUDA 11.8)..."
# 检查是否已安装 PyTorch
if python3 -c "import torch" 2>/dev/null; then
    print_warn "PyTorch 已安装，跳过..."
else
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
        --index-url https://download.pytorch.org/whl/cu118
    print_info "PyTorch 安装完成"
fi

# 验证 PyTorch CUDA
print_info "验证 PyTorch CUDA 支持..."
python3 -c "
import torch
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA 版本: {torch.version.cuda}')
    print(f'GPU 数量: {torch.cuda.device_count()}')
    print(f'GPU 名称: {torch.cuda.get_device_name(0)}')
else:
    print('警告: CUDA 不可用，某些功能将受限')
"

# ============================================================================
# 4. 安装 PaddlePaddle
# ============================================================================

print_info "安装 PaddlePaddle..."
pip install paddlepaddle-gpu>=2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple || \
pip install paddlepaddle>=2.6.0

# ============================================================================
# 5. 安装其他依赖
# ============================================================================

print_info "安装项目依赖..."
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REQUIREMENTS_FILE="$SCRIPT_DIR/requirements.txt"

if [ -f "$REQUIREMENTS_FILE" ]; then
    # 排除 PyTorch 相关行（已单独安装）
    pip install -r "$REQUIREMENTS_FILE" --extra-index-url https://download.pytorch.org/whl/cu118 2>&1 | \
        grep -v "torch\|torchvision\|torchaudio" || true
else
    print_error "未找到 requirements.txt 文件: $REQUIREMENTS_FILE"
    exit 1
fi

# ============================================================================
# 6. 安装 bitsandbytes (如果失败则跳过)
# ============================================================================

print_info "验证 bitsandbytes..."
if python3 -c "import bitsandbytes as bnb" 2>/dev/null; then
    print_info "bitsandbytes 已安装"
else
    print_warn "bitsandbytes 未正确安装，正在尝试修复..."
    # bitsandbytes 可能需要编译，如果失败则跳过
    pip install bitsandbytes>=0.41.0 || print_warn "bitsandbytes 安装失败，QLoRA 功能将受限"
fi

# ============================================================================
# 7. (可选) 安装 Flash Attention 2
# ============================================================================

print_info "是否安装 Flash Attention 2? (需要编译，可能需要 5-10 分钟)"
read -p "安装 Flash Attention 2? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "安装 Flash Attention 2..."
    # 检查编译环境
    if check_command nvcc && check_command make; then
        pip install flash-attn --no-build-isolation || \
            print_warn "Flash Attention 2 安装失败，将使用标准注意力机制"
    else
        print_warn "缺少编译工具 (nvcc/make)，跳过 Flash Attention 2"
    fi
else
    print_info "跳过 Flash Attention 2 安装"
fi

# ============================================================================
# 8. 验证安装
# ============================================================================

print_info "验证核心依赖..."
python3 << EOF
import sys

def check_import(module_name, package_name=None):
    try:
        __import__(module_name)
        print(f"  ✓ {package_name or module_name}")
        return True
    except ImportError as e:
        print(f"  ✗ {package_name or module_name}: {e}")
        return False

print("\n核心依赖检查:")
checks = [
    ("torch", "PyTorch"),
    ("paddle", "PaddlePaddle"),
    ("transformers", "Transformers"),
    ("peft", "PEFT"),
    ("bitsandbytes", "BitsAndBytes"),
    ("cv2", "OpenCV"),
    ("numpy", "NumPy"),
    ("pandas", "Pandas"),
    ("matplotlib", "Matplotlib"),
    ("seaborn", "Seaborn"),
]

all_ok = True
for mod, name in checks:
    if not check_import(mod, name):
        all_ok = False

if all_ok:
    print("\n✓ 所有核心依赖安装成功!")
else:
    print("\n✗ 部分依赖安装失败，请检查错误信息")
    sys.exit(1)
EOF

# ============================================================================
# 9. 运行环境检查脚本
# ============================================================================

if [ -f "$SCRIPT_DIR/check_env.py" ]; then
    print_info "运行环境检查脚本..."
    python3 "$SCRIPT_DIR/check_env.py" || print_warn "环境检查脚本执行失败"
fi

# ============================================================================
# 完成
# ============================================================================

print_info "安装完成!"
print_info ""
print_info "后续步骤:"
print_info "1. 激活虚拟环境: source $ENV_NAME/bin/activate"
print_info "2. 运行环境检查: python check_env.py"
print_info "3. 开始使用 L2W1 v5.0!"
print_info ""
print_info "如需退出虚拟环境，运行: deactivate"

