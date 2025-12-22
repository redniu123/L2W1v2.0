data/raw/[数据集名称]/
├── images/ # 核心图像库：存放所有行级原始图像
│ ├── train/  
│ ├── val/  
│ └── test/  
├── train.jsonl # 训练集元数据（用于 Agent B SFT 训练）
├── val.jsonl # 验证集元数据（用于训练监控）
└── test.jsonl # 测试集元数据（用于最终科研指标评估）
