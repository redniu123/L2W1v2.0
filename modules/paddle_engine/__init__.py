# Agent A: PaddleOCR 引擎模块
# 包含源码手术后的 predict_rec_modified.py

"""
L2W1 PaddleOCR 引擎模块

核心组件:
- TextRecognizerWithLogits: 增强版文本识别器，支持 Logits 导出

关键技术:
1. 在 self.predictor.run() 之后拦截原始 Logits Tensor
2. 使用 deepcopy 防止 PaddlePredictor 内存复用机制覆盖数据
3. 返回字典格式: {'results': [...], 'logits': [...], 'elapsed_time': ...}

Tensor 规格:
- raw_logits 形状: [Batch, Seq_Len, Vocab_Size]
- Seq_Len: 通常为 80 左右 (PP-OCR 默认)
- Vocab_Size: 约 6000+ (中文词表)

使用示例:
```python
from L2W1.modules.paddle_engine import TextRecognizerWithLogits

# 初始化
recognizer = TextRecognizerWithLogits(args)

# 推理
output = recognizer([img1, img2])

# 获取结果
results = output['results']  # [(text, conf), ...]
logits = output['logits']    # [np.ndarray, ...]
elapsed = output['elapsed_time']

# 计算熵
for i, logit in enumerate(logits):
    if logit is not None:
        # logit 形状: [Seq_Len, Vocab_Size]
        probs = softmax(logit, axis=-1)
        entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1)
        print(f"样本 {i} 平均熵: {np.mean(entropy):.4f}")
```
"""

# 延迟导入，避免在未安装 PaddleOCR 时报错
def get_text_recognizer_with_logits():
    """获取 TextRecognizerWithLogits 类"""
    from .predict_rec_modified import TextRecognizerWithLogits
    return TextRecognizerWithLogits

__all__ = [
    'get_text_recognizer_with_logits',
]
