#!/bin/bash

# =============================================================================
# L2W1 v5.0 一键安装脚本
# 
# 使用说明:
#   chmod +x install.sh
#   ./install.sh
# 
# 环境要求:
#   - Linux (Ubuntu 20.04+)
#   - NVIDIA GPU (RTX 2080Ti)
#   - CUDA 11.8
#   - Python 3.9-3.11
# =============================================================================

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
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# 检查 Python 版本
check_python() {
    print_info "检查 Python 版本..."
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
        print_info "Python 版本: $(python3 --version)"
        if [[ $(echo "$PYTHON_VERSION >= 3.9" | bc -l 2>/dev/null || echo "0") == "1" ]]; then
            PYTHON_CMD=python3
            PIP_CMD=pip3
        else
            print_error "Python 版本过低，需要 3.9+"
            exit 1
        fi
    else
        print_error "未找到 Python3，请先安装 Python 3.9+"
        exit 1
    fi
}

# 检查 CUDA
check_cuda() {
    print_info "检查 CUDA 环境..."
    if command_exists nvidia-smi; then
        print_info "NVIDIA GPU 检测成功:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        if command_exists nvcc; then
            CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
            print_info "CUDA 版本: $CUDA_VERSION"
        else
            print_warn "未找到 nvcc，请确保 CUDA 已正确安装"
        fi
    else
        print_warn "未找到 NVIDIA 驱动，GPU 功能可能不可用"
    fi
}

# 检查 Conda
check_conda() {
    if command_exists conda; then
        USE_CONDA=true
        print_info "检测到 Conda，将使用 Conda 环境"
    else
        USE_CONDA=false
        print_info "未检测到 Conda，将使用 venv"
    fi
}

# 创建虚拟环境
create_venv() {
    if [ "$USE_CONDA" = true ]; then
        ENV_NAME="l2w1"
        print_info "创建 Conda 环境: $ENV_NAME"
        if conda env list | grep -q "^$ENV_NAME "; then
            print_warn "环境 $ENV_NAME 已存在，是否删除并重新创建? (y/n)"
            read -r response
            if [[ "$response" == "y" ]]; then
                conda env remove -n $ENV_NAME -y
                conda create -n $ENV_NAME python=3.10 -y
            else
                print_info "使用现有环境"
            fi
        else
            conda create -n $ENV_NAME python=3.10 -y
        fi
        print_info "激活环境: conda activate $ENV_NAME"
        print_warn "请手动运行: conda activate $ENV_NAME"
    else
        VENV_DIR="venv"
        print_info "创建 venv 虚拟环境: $VENV_DIR"
        if [ -d "$VENV_DIR" ]; then
            print_warn "虚拟环境已存在，是否删除并重新创建? (y/n)"
            read -r response
            if [[ "$response" == "y" ]]; then
                rm -rf $VENV_DIR
                $PYTHON_CMD -m venv $VENV_DIR
            fi
        else
            $PYTHON_CMD -m venv $VENV_DIR
        fi
        source $VENV_DIR/bin/activate
        print_info "虚拟环境已激活"
    fi
}

# 升级 pip
upgrade_pip() {
    print_info "升级 pip..."
    $PIP_CMD install --upgrade pip setuptools wheel
}

# 安装 PyTorch
install_pytorch() {
    print_info "安装 PyTorch (CUDA 11.8)..."
    if [ "$USE_CONDA" = true ]; then
        conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
    else
        $PIP_CMD install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
    fi
    
    # 验证
    python -c "import torch; print(f'PyTorch {torch.__version__} 安装成功'); print(f'CUDA 可用: {torch.cuda.is_available()}')" || {
        print_error "PyTorch 安装失败"
        exit 1
    }
}

# 安装 PaddlePaddle
install_paddlepaddle() {
    print_info "安装 PaddlePaddle (GPU 版本)..."
    $PIP_CMD install paddlepaddle-gpu==2.6.1 -i https://mirror.baidu.com/pypi/simple || {
        print_warn "PaddlePaddle 安装失败，尝试官方源..."
        $PIP_CMD install paddlepaddle-gpu==2.6.1
    }
    
    # 验证
    python -c "import paddle; paddle.utils.run_check()" || {
        print_warn "PaddlePaddle GPU 检测失败，但可能仍可使用 CPU 模式"
    }
}

# 安装其他依赖
install_requirements() {
    print_info "安装项目依赖..."
    $PIP_CMD install -r requirements.txt
    
    # 尝试安装 BitsAndBytes
    print_info "安装 BitsAndBytes (4-bit 量化)..."
    $PIP_CMD install bitsandbytes || {
        print_warn "BitsAndBytes 安装失败，将跳过（可在后续手动安装）"
    }
    
    # 尝试安装 Flash Attention (可选)
    print_warn "是否安装 Flash Attention 2? (需要编译，可能需要较长时间) (y/n)"
    read -r response
    if [[ "$response" == "y" ]]; then
        print_info "安装 Flash Attention 2..."
        $PIP_CMD install ninja
        $PIP_CMD install flash-attn --no-build-isolation || {
            print_warn "Flash Attention 2 安装失败，将使用普通 attention"
        }
    fi
}

# 验证安装
verify_installation() {
    print_info "验证安装..."
    python << EOF
import sys
try:
    import torch
    import paddle
    import transformers
    import numpy as np
    import cv2
    from PIL import Image
    
    print("=" * 60)
    print("✅ 核心库验证成功")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PaddlePaddle: {paddle.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"NumPy: {np.__version__}")
    print(f"OpenCV: {cv2.__version__}")
    print("=" * 60)
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    sys.exit(1)
EOF
}

# 主函数
main() {
    echo "============================================================"
    echo "L2W1 v5.0 环境安装脚本"
    echo "============================================================"
    echo ""
    
    # 检查环境
    check_python
    check_cuda
    check_conda
    
    echo ""
    print_warn "准备开始安装，是否继续? (y/n)"
    read -r response
    if [[ "$response" != "y" ]]; then
        print_info "安装已取消"
        exit 0
    fi
    
    # 创建虚拟环境
    create_venv
    
    # 如果使用 venv，已经激活；如果使用 conda，提示用户激活
    if [ "$USE_CONDA" = true ]; then
        print_warn "请先运行: conda activate $ENV_NAME"
        print_warn "然后重新运行此脚本的剩余部分，或手动执行以下步骤:"
        echo ""
        echo "1. conda activate $ENV_NAME"
        echo "2. pip install --upgrade pip"
        echo "3. conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y"
        echo "4. pip install -r requirements.txt"
        echo ""
        exit 0
    fi
    
    # 安装步骤
    upgrade_pip
    install_pytorch
    install_paddlepaddle
    install_requirements
    verify_installation
    
    echo ""
    print_info "============================================================"
    print_info "✅ 安装完成!"
    print_info "============================================================"
    print_info "下一步:"
    print_info "  1. 激活环境: source venv/bin/activate"
    print_info "  2. 运行测试: python -c \"from L2W1.modules.paddle_engine import TextRecognizerWithLogits\""
    print_info "  3. 查看文档: cat INSTALL.md"
    echo ""
}

# 运行主函数
main

