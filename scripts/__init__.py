# L2W1 执行脚本模块

"""
SH-DA++ Stage 2 脚本模块

核心脚本:
1. adapt_geology_data.py - 地质数据适配器
   - 将原始数据转换为 V2.0 协议格式
   
2. prepare_calibration_data.py - 特征提取与标签构造
   - 从 Agent A 输出提取特征
   - 自动构造 y_deletion 标签
   
3. train_calibrator.py - 校准训练器
   - 训练 Logistic Regression
   - 保存权重到配置文件
   
4. test_stage2_modules.py - 模块单元测试
   - 测试标签生成器、评分器、回填控制器
   
5. test_stage2_integration.py - 集成测试
   - 测试完整 Pipeline
   
6. run_stage2_execution.py - 完整执行脚本
   - 数据适配 → 特征提取 → 校准训练
"""

__all__ = []
