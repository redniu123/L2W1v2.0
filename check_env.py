#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L2W1 v5.0 环境检查脚本

使用方法:
    python check_env.py

检查项目:
    - Python 版本
    - CUDA/GPU 可用性
    - 核心依赖库
    - 版本兼容性
"""

import sys
import platform
import io
from pathlib import Path

# 修复 Windows 控制台 Unicode 编码问题
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


def print_header(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_status(name, status, details=""):
    symbol = "✅" if status else "❌"
    print(f"{symbol} {name}")
    if details:
        print(f"   {details}")


def check_python():
    print_header("Python 环境")
    version = sys.version_info
    print_status(
        f"Python {version.major}.{version.minor}.{version.micro}",
        version.major == 3 and version.minor >= 9,
        f"{sys.executable}",
    )
    return version.major == 3 and version.minor >= 9


def check_cuda():
    print_header("CUDA/GPU 环境")

    # 检查 nvidia-smi
    import subprocess

    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.split("\n")
            for line in lines:
                if "Driver Version" in line:
                    driver_version = line.split("Driver Version:")[1].split()[0]
                    print_status("NVIDIA Driver", True, f"版本: {driver_version}")
                if "CUDA Version" in line:
                    cuda_version = line.split("CUDA Version:")[1].split()[0]
                    print_status("CUDA", True, f"版本: {cuda_version}")
        else:
            print_status("NVIDIA Driver", False, "nvidia-smi 命令失败")
            return False
    except FileNotFoundError:
        print_status("NVIDIA Driver", False, "未找到 nvidia-smi 命令")
        return False
    except Exception as e:
        print_status("NVIDIA Driver", False, f"检查失败: {e}")
        return False

    # 检查 PyTorch CUDA
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        print_status("PyTorch CUDA", cuda_available)
        if cuda_available:
            print(f"   GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                print(
                    f"   显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB"
                )
        return cuda_available
    except ImportError:
        print_status("PyTorch", False, "未安装")
        return False


def check_core_libs():
    print_header("核心依赖库")

    libs = {
        "torch": "PyTorch",
        "paddle": "PaddlePaddle",
        "transformers": "Transformers",
        "peft": "PEFT",
        "numpy": "NumPy",
        "cv2": "OpenCV",
        "PIL": "Pillow",
        "tqdm": "tqdm",
        "yaml": "PyYAML",
    }

    results = {}
    for module_name, display_name in libs.items():
        try:
            if module_name == "cv2":
                import cv2

                version = cv2.__version__
            elif module_name == "PIL":
                from PIL import Image
                import PIL

                version = PIL.__version__
            elif module_name == "yaml":
                import yaml

                version = yaml.__version__
            elif module_name == "paddle":
                import paddle

                version = paddle.__version__
            else:
                module = __import__(module_name)
                version = getattr(module, "__version__", "unknown")

            print_status(display_name, True, f"版本: {version}")
            results[module_name] = True
        except ImportError:
            print_status(display_name, False, "未安装")
            results[module_name] = False

    return all(results.values())


def check_optional_libs():
    print_header("可选依赖库")

    optional_libs = {
        "bitsandbytes": "BitsAndBytes (4-bit 量化)",
        "flash_attn": "Flash Attention 2",
        "matplotlib": "Matplotlib (可视化)",
        "seaborn": "Seaborn (可视化)",
        "pandas": "Pandas",
        "editdistance": "EditDistance",
    }

    for module_name, display_name in optional_libs.items():
        try:
            if module_name == "flash_attn":
                import flash_attn

                version = "installed"
            elif module_name == "bitsandbytes":
                import bitsandbytes as bnb

                version = bnb.__version__
            else:
                module = __import__(module_name)
                version = getattr(module, "__version__", "installed")
            print_status(display_name, True, f"版本: {version}")
        except ImportError:
            print_status(display_name, False, "未安装 (可选)")


def check_project_structure():
    print_header("项目结构")

    required_dirs = [
        "modules",
        "modules/paddle_engine",
        "modules/router",
        "modules/vlm_expert",
        "modules/utils",
        "scripts",
        "configs",
        "data",
    ]

    base_path = Path(__file__).parent
    all_exist = True

    for dir_path in required_dirs:
        full_path = base_path / dir_path
        exists = full_path.exists()
        print_status(dir_path, exists)
        if not exists:
            all_exist = False

    return all_exist


def check_paddle_gpu():
    print_header("PaddlePaddle GPU 检测")

    try:
        import paddle

        paddle.utils.run_check()
        print_status("PaddlePaddle GPU", True, "GPU 可用")
        return True
    except Exception as e:
        print_status("PaddlePaddle GPU", False, f"检测失败: {str(e)}")
        return False


def check_bitsandbytes():
    print_header("BitsAndBytes 量化支持")

    try:
        import bitsandbytes as bnb

        print_status("BitsAndBytes", True, f"版本: {bnb.__version__}")

        # 测试量化功能
        import torch

        if torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig

                config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
                print_status("4-bit 量化配置", True, "测试通过")
                return True
            except Exception as e:
                print_status("4-bit 量化配置", False, f"测试失败: {e}")
                return False
        else:
            print_status("4-bit 量化配置", False, "需要 CUDA 支持")
            return False
    except ImportError:
        print_status("BitsAndBytes", False, "未安装")
        return False


def main():
    print("=" * 60)
    print(" L2W1 v5.0 环境检查")
    print("=" * 60)
    print(f"系统: {platform.system()} {platform.release()}")
    print(f"架构: {platform.machine()}")

    # 执行检查
    checks = [
        ("Python", check_python),
        ("CUDA/GPU", check_cuda),
        ("核心库", check_core_libs),
        ("项目结构", check_project_structure),
        ("PaddlePaddle GPU", check_paddle_gpu),
        ("BitsAndBytes", check_bitsandbytes),
    ]

    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ {name} 检查出错: {e}")
            results[name] = False

    # 可选库
    check_optional_libs()

    # 总结
    print_header("检查总结")

    all_passed = all(results.values())
    for name, passed in results.items():
        status = "通过" if passed else "失败"
        symbol = "✅" if passed else "❌"
        print(f"{symbol} {name}: {status}")

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 环境检查通过！可以开始使用 L2W1 v5.0")
    else:
        print("❌ 环境检查未完全通过，请查看上面的错误信息")
        print("   参考 INSTALL.md 进行环境配置")
    print("=" * 60 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
