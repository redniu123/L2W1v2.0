# L2W1 v5.0 环境配置总结

**最后更新**: 2025-12-21  
**目标环境**: Linux Server + RTX 2080Ti (22GB) + CUDA 11.8

---

## 📦 配置文件清单

| 文件                   | 用途                                 |
| ---------------------- | ------------------------------------ |
| `requirements.txt`     | 生产环境依赖（所有必需库）           |
| `requirements-dev.txt` | 开发环境依赖（包含测试、格式化工具） |
| `install.sh`           | 一键安装脚本（自动化部署）           |
| `check_env.py`         | 环境检查脚本（验证安装）             |
| `INSTALL.md`           | 详细安装指南（手动步骤）             |

---

## 🚀 快速开始

### 方法 1: 使用安装脚本 (推荐)

```bash
# 1. 给脚本执行权限
chmod +x install.sh

# 2. 运行安装脚本
./install.sh

# 3. 激活环境 (如果使用 venv)
source venv/bin/activate

# 4. 验证安装
python check_env.py
```

### 方法 2: 使用 Conda

```bash
# 1. 创建环境
conda create -n l2w1 python=3.10 -y
conda activate l2w1

# 2. 安装 PyTorch (CUDA 11.8)
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 3. 安装 PaddlePaddle
pip install paddlepaddle-gpu==2.6.1 -i https://mirror.baidu.com/pypi/simple

# 4. 安装其他依赖
pip install -r requirements.txt

# 5. 验证
python check_env.py
```

### 方法 3: 手动安装

参考 `INSTALL.md` 中的详细步骤。

---

## 🔑 关键依赖版本

### 核心框架

| 库           | 版本     | 说明           |
| ------------ | -------- | -------------- |
| Python       | 3.9-3.11 | 推荐 3.10      |
| PyTorch      | 2.1.2    | CUDA 11.8 版本 |
| PaddlePaddle | 2.6.1    | GPU 版本       |
| Transformers | >=4.40.0 | HuggingFace    |
| PEFT         | >=0.8.0  | LoRA 微调      |
| BitsAndBytes | >=0.41.0 | 4-bit 量化     |

### 关键特性

- **CUDA 11.8**: RTX 2080Ti 支持的 CUDA 版本
- **4-bit 量化**: 使用 BitsAndBytes 适配 11GB 显存
- **Flash Attention 2**: 可选，用于长序列加速

---

## ✅ 验证清单

运行 `python check_env.py` 后，应看到:

```
✅ Python 3.10.x
✅ NVIDIA Driver (版本 >= 450.80.02)
✅ CUDA 11.8
✅ PyTorch CUDA 可用
✅ 核心依赖库全部安装
✅ PaddlePaddle GPU 可用
✅ BitsAndBytes 4-bit 量化支持
```

---

## 🔧 常见问题

### 1. BitsAndBytes 安装失败

```bash
# 确保有编译工具
sudo apt-get install build-essential

# 重新安装
pip install bitsandbytes --no-cache-dir --force-reinstall
```

### 2. Flash Attention 2 安装失败

可以跳过，系统会自动使用普通 attention。如果需要安装:

```bash
pip install ninja
pip install flash-attn --no-build-isolation
```

### 3. PaddlePaddle 无法使用 GPU

```bash
# 检查 CUDA 版本
nvcc --version

# 重新安装匹配版本
pip uninstall paddlepaddle-gpu
pip install paddlepaddle-gpu==2.6.1.post118 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

### 4. 显存不足

RTX 2080Ti 实际显存为 **11GB**。如果遇到 OOM:

1. 确保使用 4-bit 量化 (代码中 `use_4bit=True`)
2. 减小 batch size (`batch_size=2`)
3. 启用梯度检查点 (`gradient_checkpointing=True`)

---

## 📝 VS Code 配置

### 1. Python 解释器

在 VS Code 中按 `Ctrl+Shift+P`，选择 "Python: Select Interpreter"，选择:

- `./venv/bin/python` (如果使用 venv)
- `~/anaconda3/envs/l2w1/bin/python` (如果使用 conda)

### 2. 推荐扩展

- Python (Microsoft)
- Pylance (Microsoft)
- Jupyter (Microsoft)

### 3. 工作区设置

创建 `.vscode/settings.json`:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.linting.enabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true
}
```

---

## 🎯 下一步

安装完成后，可以:

1. **运行数据管道**:

   ```bash
   python scripts/data_pipeline.py --data_dir ./data/raw --output_dir ./data/sft
   ```

2. **训练 Agent B**:

   ```bash
   python scripts/train_agent_b.py --data_path ./data/sft/agent_b_train.jsonl
   ```

3. **评估模型**:
   ```bash
   python scripts/evaluate.py --predictions ./data/test/inference_results.jsonl
   ```

---

## 📚 相关文档

- `INSTALL.md`: 详细安装指南
- `PROJECT_STRUCTURE.md`: 项目结构说明
- `CODE_AUDIT_REPORT.md`: 代码审计报告
- `HARDENING_CHANGELOG.md`: 代码加固记录

---

**环境配置完成!** 🎉

如有问题，请查看 `INSTALL.md` 或提交 Issue。
