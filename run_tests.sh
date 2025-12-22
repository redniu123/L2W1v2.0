#!/bin/bash
# ============================================================================
# L2W1 v5.0 完整测试脚本
# 一键运行所有测试模块
# ============================================================================

set -e

# 颜色输出
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_step() {
    echo -e "${BLUE}[步骤 $1]${NC} $2"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

# 检查 conda 环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    print_info "未检测到 conda 环境，尝试激活 l2w1v2..."
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate l2w1v2 2>/dev/null || {
        print_error "请先激活 conda 环境: conda activate l2w1v2"
        exit 1
    }
fi

print_info "当前环境: $CONDA_DEFAULT_ENV"
print_info "Python: $(python --version)"
echo ""

# 测试计数
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# 运行测试函数
run_test() {
    local test_name=$1
    local test_file=$2
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    print_step "$TOTAL_TESTS" "测试 $test_name"
    
    if [ -f "$test_file" ]; then
        if python "$test_file" 2>&1; then
            print_success "$test_name 通过"
            PASSED_TESTS=$((PASSED_TESTS + 1))
            return 0
        else
            print_error "$test_name 失败"
            FAILED_TESTS=$((FAILED_TESTS + 1))
            return 1
        fi
    else
        print_error "测试文件不存在: $test_file"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

echo "=========================================="
echo "L2W1 v5.0 完整测试套件"
echo "=========================================="
echo ""

# 测试 1: 环境验证
run_test "环境验证" "test_imports.py"

# 测试 2: 模块导入
run_test "模块导入" "test_modules.py"

# 测试 3: 数据管道
run_test "数据管道" "test_data_pipeline.py"

# 测试 4: Router
run_test "Router 模块" "test_router.py"

# 测试 5: Agent B
run_test "Agent B 模块" "test_agent_b.py"

# 测试 6: Pipeline
run_test "Pipeline 模块" "test_pipeline.py"

# 测试 7: 评估模块
run_test "评估模块" "test_evaluate.py"

# 总结
echo ""
echo "=========================================="
echo "测试总结"
echo "=========================================="
echo "总测试数: $TOTAL_TESTS"
print_success "通过: $PASSED_TESTS"
if [ $FAILED_TESTS -gt 0 ]; then
    print_error "失败: $FAILED_TESTS"
else
    print_success "失败: $FAILED_TESTS"
fi
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    print_success "所有测试通过! 🎉"
    exit 0
else
    print_error "部分测试失败，请检查错误信息"
    exit 1
fi

