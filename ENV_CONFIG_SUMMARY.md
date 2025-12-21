# L2W1 v5.0 环境配置总结

## 📦 配置文件清单

本次更新创建/更新了以下环境配置文件：

| 文件 | 用途 | 说明 |
|------|------|------|
| `requirements.txt` | 完整依赖列表 | 包含所有必需和可选依赖，带版本约束 |
| `install_linux.sh` | 自动安装脚本 | 一键安装所有依赖，包含环境检查 |
| `DEPLOYMENT_LINUX.md` | 部署指南 | 详细的 Linux 服务器部署文档 |
| `verify_installation.py` | 验证脚本 | 快速验证所有依赖是否正确安装 |

---

## 🎯 核心依赖版本

### 深度学习框架
- **PyTorch**: 2.1.2 (CUDA 11.8)
- **PaddlePaddle**: >=2.6.0
- **Transformers**: >=4.40.0, <4.46.0

### 量化与微调
- **bitsandbytes**: >=0.41.0, <0.43.0 (4-bit 量化)
- **peft**: >=0.7.0, <0.11.0 (LoRA/QLoRA)
- **accelerate**: >=0.25.0, <0.30.0

### 数据处理
- **NumPy**: >=1.24.0, <2.0.0
- **Pandas**: >=2.0.0, <3.0.0
- **OpenCV**: >=4.8.0, <5.0.0
- **Pillow**: >=10.0.0, <11.0.0

### 可视化
- **Matplotlib**: >=3.7.0, <4.0.0
- **Seaborn**: >=0.12.0, <0.14.0

---

## 🚀 快速开始

### 1. 使用自动安装脚本 (推荐)

```bash
cd L2W1
bash install_linux.sh
source l2w1_env/bin/activate
python verify_installation.py
```

### 2. 手动安装

```bash
# 创建虚拟环境
python3 -m venv l2w1_env
source l2w1_env/bin/activate

# 安装 PyTorch (CUDA 11.8)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt

# 验证安装
python verify_installation.py
```

---

## 🔍 版本兼容性说明

### CUDA 版本
- **推荐**: CUDA 11.8
- **支持**: CUDA 11.0 - 11.8
- **RTX 2080Ti**: 完全支持 CUDA 11.8

### Python 版本
- **推荐**: Python 3.10
- **支持**: Python 3.8, 3.9, 3.10, 3.11
- **不支持**: Python 3.12+ (部分库可能未适配)

### 显存要求
- **最小**: 11GB (RTX 2080Ti)
- **推荐**: 16GB+ (用于更大 batch size)
- **优化**: 使用 4-bit 量化可将显存占用降至 <8GB

---

## ⚠️ 已知问题与解决方案

### 1. bitsandbytes 编译问题

**症状**: `ImportError: No module named 'bitsandbytes'`

**原因**: bitsandbytes 需要编译，依赖 CUDA Toolkit 和编译工具

**解决**:
```bash
# 安装编译工具
sudo apt-get install build-essential

# 确保 CUDA Toolkit 已安装
nvcc --version

# 重新安装
pip install bitsandbytes>=0.41.0
```

### 2. Flash Attention 2 安装失败

**症状**: Flash Attention 2 编译失败或安装超时

**原因**: 需要编译，可能需要 5-10 分钟

**解决**: 
- 这是**可选依赖**，可以跳过
- 代码会自动回退到标准注意力机制
- 性能影响: 约 10-20% 推理速度差异

### 3. PaddlePaddle CUDA 库找不到

**症状**: `libcudart.so.xxx: cannot open shared object file`

**解决**:
```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
# 添加到 ~/.bashrc 永久生效
```

---

## 📊 依赖关系图

```
L2W1 v5.0
├── Agent A (PP-OCRv5)
│   └── PaddlePaddle >= 2.6.0
├── Router (Uncertainty)
│   ├── NumPy >= 1.24.0
│   └── Transformers (语义 PPL)
├── Agent B (Qwen2.5-VL)
│   ├── PyTorch 2.1.2 (CUDA 11.8)
│   ├── Transformers >= 4.40.0
│   ├── bitsandbytes >= 0.41.0 (4-bit)
│   ├── peft >= 0.7.0 (LoRA)
│   └── qwen-vl-utils
└── 工具链
    ├── OpenCV (图像处理)
    ├── Matplotlib/Seaborn (可视化)
    └── editdistance (评估指标)
```

---

## 🔄 更新依赖

### 更新单个包
```bash
pip install --upgrade <package_name>
```

### 更新所有包
```bash
pip install --upgrade -r requirements.txt
```

### 检查过时的包
```bash
pip list --outdated
```

---

## 📝 验证清单

安装完成后，运行以下检查：

- [ ] `python verify_installation.py` - 所有核心依赖正常
- [ ] `python check_env.py` - 完整环境检查通过
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` - CUDA 可用
- [ ] `python -c "import bitsandbytes as bnb"` - 量化库正常
- [ ] `nvidia-smi` - GPU 识别正常

---

## 📚 相关文档

- **[DEPLOYMENT_LINUX.md](./DEPLOYMENT_LINUX.md)**: 详细部署指南
- **[requirements.txt](./requirements.txt)**: 完整依赖列表
- **[install_linux.sh](./install_linux.sh)**: 自动安装脚本
- **[verify_installation.py](./verify_installation.py)**: 验证脚本

---

## 🆘 获取帮助

如果遇到问题：

1. 查看 [DEPLOYMENT_LINUX.md](./DEPLOYMENT_LINUX.md) 的"常见问题"部分
2. 运行 `python verify_installation.py` 查看详细错误
3. 检查系统日志: `dmesg | grep -i nvidia`
4. 验证 CUDA 环境: `nvcc --version && nvidia-smi`

---

**配置完成日期**: 2025-12-21  
**配置版本**: v5.0.1  
**目标平台**: Linux (Ubuntu 20.04+), RTX 2080Ti, CUDA 11.8

