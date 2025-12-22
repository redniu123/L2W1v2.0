#!/bin/bash
# ============================================================================
# L2W1 v5.0 模型下载脚本
# 自动下载所有必需的模型
# ============================================================================

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 激活环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    print_info "激活 conda 环境..."
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate l2w1v2 2>/dev/null || {
        print_error "请先激活 conda 环境: conda activate l2w1v2"
        exit 1
    }
fi

print_info "当前环境: $CONDA_DEFAULT_ENV"
echo ""

# ============================================================================
# 1. 下载 Agent A 模型 (PP-OCRv5)
# ============================================================================

print_info "=========================================="
print_info "步骤 1: 下载 Agent A 模型 (PP-OCRv5)"
print_info "=========================================="

AGENT_A_DIR="models/agent_a_ppocr"
AGENT_A_URL="https://paddleocr.bj.bcebos.com/PP-OCRv5/chinese/ch_PP-OCRv5_rec_infer.tar"

if [ -f "$AGENT_A_DIR/inference.pdmodel" ] && [ -f "$AGENT_A_DIR/inference.pdiparams" ]; then
    print_warn "Agent A 模型已存在，跳过下载"
else
    print_info "创建目录: $AGENT_A_DIR"
    mkdir -p "$AGENT_A_DIR"
    cd "$AGENT_A_DIR"
    
    print_info "下载 PP-OCRv5 识别模型..."
    if command -v wget &> /dev/null; then
        wget "$AGENT_A_URL" -O model.tar
    elif command -v curl &> /dev/null; then
        curl -L "$AGENT_A_URL" -o model.tar
    else
        print_error "未找到 wget 或 curl，请手动下载: $AGENT_A_URL"
        exit 1
    fi
    
    print_info "解压模型..."
    tar -xf model.tar
    
    if [ -d "inference" ]; then
        mv inference/* .
        rmdir inference
    fi
    
    rm -f model.tar
    
    cd - > /dev/null
    
    # 验证
    if [ -f "$AGENT_A_DIR/inference.pdmodel" ] && [ -f "$AGENT_A_DIR/inference.pdiparams" ]; then
        print_info "✓ Agent A 模型下载完成"
    else
        print_error "✗ Agent A 模型下载失败"
        exit 1
    fi
fi

echo ""

# ============================================================================
# 2. 下载 Agent B 模型 (Qwen2.5-VL-3B)
# ============================================================================

print_info "=========================================="
print_info "步骤 2: 下载 Agent B 模型 (Qwen2.5-VL-3B)"
print_info "=========================================="
print_warn "注意: 模型较大 (~6GB)，下载可能需要较长时间"

python << 'EOF'
import sys
from pathlib import Path

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    print(f"正在下载 {model_name}...")
    print("这可能需要几分钟，请耐心等待...")
    
    # 下载模型
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name,
        trust_remote_code=True,
        resume_download=True
    )
    print("✓ 模型下载完成")
    
    # 下载处理器
    print("正在下载处理器...")
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        resume_download=True
    )
    print("✓ 处理器下载完成")
    
    print(f"\n模型位置: {Path.home() / '.cache' / 'huggingface' / 'hub'}")
    print("✓ Agent B 模型准备完成")
    
except ImportError:
    print("✗ 错误: 未安装 transformers")
    print("  请运行: pip install transformers")
    sys.exit(1)
except Exception as e:
    print(f"✗ 下载失败: {e}")
    print("  提示: 可以稍后在首次使用时自动下载")
    sys.exit(1)
EOF

DOWNLOAD_STATUS=$?

if [ $DOWNLOAD_STATUS -eq 0 ]; then
    print_info "✓ Agent B 模型下载完成"
else
    print_warn "Agent B 模型下载失败，将在首次使用时自动下载"
fi

echo ""

# ============================================================================
# 3. 下载 Router 语言模型 (可选)
# ============================================================================

print_info "=========================================="
print_info "步骤 3: 下载 Router 语言模型 (可选)"
print_info "=========================================="

read -p "是否下载 Router 语言模型 (Qwen2.5-0.5B)? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    python << 'EOF'
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        print(f"正在下载 {model_name}...")
        
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        print("✓ Router 语言模型下载完成")
    except Exception as e:
        print(f"✗ 下载失败: {e}")
        print("  将在首次使用时自动下载")
EOF
else
    print_info "跳过 Router 语言模型下载（将在首次使用时自动下载）"
fi

echo ""

# ============================================================================
# 完成
# ============================================================================

print_info "=========================================="
print_info "模型下载完成!"
print_info "=========================================="
echo ""
print_info "已下载的模型:"
echo "  - Agent A: $AGENT_A_DIR"
echo "  - Agent B: ~/.cache/huggingface/hub/ (自动管理)"
echo ""
print_info "下一步:"
echo "  1. 准备数据: 创建 data/raw/images/ 和 data/raw/labels.txt"
echo "  2. 运行 Pipeline: python run_pipeline.py data/raw/images/line_001.jpg"
echo ""

