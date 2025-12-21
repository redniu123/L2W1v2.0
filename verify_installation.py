#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L2W1 v5.0 安装验证脚本

快速验证所有关键依赖是否正确安装并可用。

Usage:
    python verify_installation.py
"""

import sys
import os
from pathlib import Path

# 修复 Windows 控制台编码
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# 颜色输出 (Linux/Mac)
USE_COLOR = sys.platform != "win32" or "ANSICON" in os.environ


class Colors:
    if USE_COLOR:
        GREEN = "\033[92m"
        RED = "\033[91m"
        YELLOW = "\033[93m"
        BLUE = "\033[94m"
        END = "\033[0m"
        BOLD = "\033[1m"
        CHECK = "✓"
        CROSS = "✗"
        WARN = "⚠"
    else:
        GREEN = RED = YELLOW = BLUE = END = BOLD = ""
        CHECK = "[OK]"
        CROSS = "[FAIL]"
        WARN = "[WARN]"


def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 60}{Colors.END}\n")


def print_success(text):
    print(f"{Colors.GREEN}{Colors.CHECK}{Colors.END} {text}")


def print_error(text):
    print(f"{Colors.RED}{Colors.CROSS}{Colors.END} {text}")


def print_warning(text):
    print(f"{Colors.YELLOW}{Colors.WARN}{Colors.END} {text}")


def check_import(module_name, package_name=None, optional=False):
    """检查模块是否可以导入"""
    name = package_name or module_name
    try:
        if module_name == "cv2":
            import cv2

            version = cv2.__version__
        elif module_name == "paddle":
            import paddle

            version = paddle.__version__
        else:
            mod = __import__(module_name)
            version = getattr(mod, "__version__", "unknown")

        print_success(f"{name}: {version}")
        return True
    except ImportError as e:
        if optional:
            print_warning(f"{name}: 未安装 (可选)")
        else:
            print_error(f"{name}: 导入失败 - {e}")
        return False
    except Exception as e:
        print_error(f"{name}: 错误 - {e}")
        return False


def check_pytorch():
    """检查 PyTorch 和 CUDA"""
    try:
        import torch

        print_success(f"PyTorch: {torch.__version__}")

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            print_success(f"CUDA 可用: {torch.version.cuda}")
            print_success(f"GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print_success(f"  GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
        else:
            print_warning("CUDA 不可用 (某些功能将受限)")

        return True
    except ImportError:
        print_error("PyTorch: 未安装")
        return False


def check_paddle():
    """检查 PaddlePaddle"""
    try:
        import paddle

        print_success(f"PaddlePaddle: {paddle.__version__}")

        cuda_compiled = paddle.device.is_compiled_with_cuda()
        if cuda_compiled:
            print_success("PaddlePaddle 支持 CUDA")
        else:
            print_warning("PaddlePaddle 为 CPU 版本")

        return True
    except ImportError:
        print_error("PaddlePaddle: 未安装")
        return False


def check_bitsandbytes():
    """检查 bitsandbytes"""
    try:
        import bitsandbytes as bnb

        print_success(f"bitsandbytes: {bnb.__version__}")

        # 测试 CUDA 功能
        try:
            from bitsandbytes.functional import quantize_blockwise

            print_success("bitsandbytes CUDA 功能正常")
        except Exception as e:
            print_warning(f"bitsandbytes CUDA 功能异常: {e}")

        return True
    except ImportError:
        print_error("bitsandbytes: 未安装 (QLoRA 功能将受限)")
        return False


def check_flash_attention():
    """检查 Flash Attention"""
    try:
        import flash_attn

        print_success(f"flash-attn: {flash_attn.__version__}")
        return True
    except ImportError:
        print_warning("flash-attn: 未安装 (将使用标准注意力机制)")
        return False


def main():
    print_header("L2W1 v5.0 安装验证")

    # Python 版本
    print(f"Python 版本: {sys.version}")
    print()

    # 核心依赖
    print_header("核心依赖检查")
    core_checks = [
        ("torch", "PyTorch"),
        ("paddle", "PaddlePaddle"),
        ("transformers", "Transformers"),
        ("peft", "PEFT"),
        ("accelerate", "Accelerate"),
    ]

    core_results = []
    for mod, name in core_checks:
        if mod == "torch":
            result = check_pytorch()
        elif mod == "paddle":
            result = check_paddle()
        else:
            result = check_import(mod, name)
        core_results.append(result)

    # 量化库
    print_header("量化与优化")
    check_bitsandbytes()
    check_flash_attention()

    # 数据处理
    print_header("数据处理库")
    data_checks = [
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("PIL", "Pillow"),
        ("scipy", "SciPy"),
    ]

    for mod, name in data_checks:
        check_import(mod, name)

    # 可视化
    print_header("可视化库")
    check_import("matplotlib", "Matplotlib")
    check_import("seaborn", "Seaborn")

    # 工具库
    print_header("工具库")
    util_checks = [
        ("tqdm", "tqdm"),
        ("yaml", "PyYAML"),
        ("tensorboard", "TensorBoard"),
        ("editdistance", "editdistance"),
    ]

    for mod, name in util_checks:
        check_import(mod, name)

    # 总结
    print_header("验证总结")

    all_core_ok = all(core_results)

    if all_core_ok:
        print_success("所有核心依赖安装正确!")
        print()
        print("可以开始使用 L2W1 v5.0:")
        print("  python scripts/data_pipeline.py --help")
        print("  python scripts/train_agent_b.py --help")
        print("  python check_env.py  # 运行完整环境检查")
    else:
        print_error("部分核心依赖安装失败!")
        print()
        print("请检查错误信息并重新安装失败的依赖:")
        print("  pip install -r requirements.txt")
        print("  或运行: bash install_linux.sh")
        sys.exit(1)

    print()


if __name__ == "__main__":
    main()
