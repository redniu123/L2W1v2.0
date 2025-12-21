# L2W1 项目

分层多智能体 OCR 系统，基于 PaddleOCR-VL 和 Qwen2.5-VL 实现。

## 项目结构

```
L2W1/
├── configs/                # 配置文件：模型超参数、推理阈值、Prompt 模板
├── data/                   # 数据工程核心目录
│   ├── raw/                # 原始数据集：SCUT-HCCDoc, VisCGEC, CASIA
│   ├── processed/          # 经过自动化流水线处理的行级图像与标注
│   └── sft/                # 用于 Agent B 微调的 JSON 格式"错题集"
├── docs/                   # 文档：技术规范(Spec)、开题报告、论文草稿
├── models/                 # 模型权重管理（建议链接到大容量存储盘）
│   ├── agent_a_ppocr/      # PP-OCRv5 预训练权重
│   ├── router_qwen/        # Qwen2.5-0.5B 权重
│   └── agent_b_vlm/        # Qwen2.5-VL-3B (Int4/FP16) 权重
├── modules/                # 核心功能模块化实现
│   ├── paddle_engine/      # Agent A: 包含源码手术后的 predict_rec_modified.py
│   ├── router/             # Router: 视觉熵计算与语义 PPL 评估器
│   └── vlm_expert/         # Agent B: 动态分辨率配置与显式索引推理逻辑
├── notebooks/              # 实验分析：Loss 曲线分析、Badcase 可视化分析
├── outputs/                # 实验产出：日志、微调后的 Checkpoints、可视化热力图
├── scripts/                # 执行脚本
│   ├── data_pipeline.py    # 自动化数据处理流水线
│   ├── train_agent_b.py    # Agent B 的 SFT 训练脚本
│   └── evaluate.py         # 核心评价指标（CER, OCR-R）计算脚本
├── requirements.txt        # 依赖环境
└── README.md               # 项目主说明文档
```

## 核心背景

L2W1 是分层多智能体架构：
- **Agent A (PaddleOCR-VL)**: 作为初步扫描器，负责初步 OCR 识别
- **Router**: 负责路由决策，判断是否需要调用 Agent B
- **Agent B (Qwen2.5-VL)**: 作为专家模型，处理复杂或困难样本

## 关键技术点

1. **CTC Logits 导出**: 解决 PaddleOCR-VL 封装过深导致无法获取 CTC Logits 的问题
2. **动态路由**: 基于视觉熵计算与语义 PPL 评估器进行路由决策
3. **BBox 裁剪**: Agent B 适配逻辑 - BBox 坐标对齐 + 0.3 倍上下文扩充 + 336x336 标准化画布

## 环境依赖

见 `requirements.txt`

## 使用说明

（待补充）

## 参考资料

- PaddleOCR 官方文档
- Qwen2.5-VL 模型文档

