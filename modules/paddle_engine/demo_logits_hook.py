"""
L2W1 Logits Hooking 验证脚本

功能演示:
1. 验证熵计算算法的正确性
2. 演示 Logits 形状和视觉不确定性计算
3. (可选) 使用真实 PaddleOCR 模型进行测试

运行模式:
- 默认模式: 使用模拟数据验证算法 (不需要 PaddleOCR)
- 真实模式: 需要 PaddleOCR 环境和模型权重

使用方法:
    # 模拟数据演示 (推荐，不需要额外依赖)
    python demo_logits_hook.py --demo_only

    # 真实模型测试 (需要 PaddleOCR 环境)
    python demo_logits_hook.py --image_dir ./test_images/ --rec_model_dir ./models/rec/
"""

import os
import sys
import argparse
import numpy as np

# 添加项目路径
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../.."))
sys.path.insert(0, PROJECT_ROOT)

# 检查是否有 PaddleOCR 环境 (可选)
PADDLEOCR_AVAILABLE = False
try:
    import paddle
    from ppocr.postprocess import build_post_process

    PADDLEOCR_AVAILABLE = True
except ImportError:
    pass


def compute_visual_entropy(logits: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    计算视觉不确定性 (Visual Entropy)

    公式: H = -sum(P * log(P + epsilon))

    Args:
        logits: 原始 logits，形状 [Seq_Len, Vocab_Size]
        epsilon: 防止 log(0) 的小常数

    Returns:
        entropy_seq: 熵序列，形状 [Seq_Len,]
    """
    # Softmax 归一化
    probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = probs / np.sum(probs, axis=-1, keepdims=True)

    # Shannon Entropy 计算
    entropy = -np.sum(probs * np.log(probs + epsilon), axis=-1)

    return entropy


def demo_logits_extraction():
    """演示 Logits 提取和熵计算"""

    print("=" * 60)
    print("L2W1 Logits Hooking 演示")
    print("=" * 60)

    # 模拟 logits 数据（实际使用时从模型获取）
    # 形状: [Seq_Len=80, Vocab_Size=6625]
    batch_size = 2
    seq_len = 80
    vocab_size = 6625

    print(f"\n[模拟数据] 生成测试 logits...")
    print(f"  - Batch Size: {batch_size}")
    print(f"  - Sequence Length: {seq_len}")
    print(f"  - Vocabulary Size: {vocab_size}")

    # 模拟两种情况的 logits
    # Case 1: 高置信度（低熵）- 单峰分布
    high_conf_logits = np.random.randn(seq_len, vocab_size) * 0.1
    high_conf_logits[:, 100] = 10.0  # 在某个字符上有高置信度

    # Case 2: 低置信度（高熵）- 平坦分布
    low_conf_logits = np.random.randn(seq_len, vocab_size) * 0.1

    print("\n[熵计算] 计算视觉不确定性...")

    # 计算熵
    high_conf_entropy = compute_visual_entropy(high_conf_logits)
    low_conf_entropy = compute_visual_entropy(low_conf_logits)

    print(f"\n高置信度样本:")
    print(f"  - 熵序列形状: {high_conf_entropy.shape}")
    print(f"  - 平均熵: {np.mean(high_conf_entropy):.4f}")
    print(f"  - 最大熵: {np.max(high_conf_entropy):.4f}")
    print(f"  - 最小熵: {np.min(high_conf_entropy):.4f}")

    print(f"\n低置信度样本:")
    print(f"  - 熵序列形状: {low_conf_entropy.shape}")
    print(f"  - 平均熵: {np.mean(low_conf_entropy):.4f}")
    print(f"  - 最大熵: {np.max(low_conf_entropy):.4f}")
    print(f"  - 最小熵: {np.min(low_conf_entropy):.4f}")

    print("\n[结论]")
    print(
        f"  高置信度样本的平均熵 ({np.mean(high_conf_entropy):.4f}) < "
        f"低置信度样本的平均熵 ({np.mean(low_conf_entropy):.4f})"
    )
    print("  -> 熵可以有效区分高/低置信度样本")

    print("\n" + "=" * 60)
    print("验证完成!")
    print("=" * 60)


def demo_with_real_model(args):
    """使用真实模型进行演示 (需要完整 PaddleOCR 环境)"""

    if not PADDLEOCR_AVAILABLE:
        print("=" * 60)
        print("PaddleOCR 环境检测")
        print("=" * 60)
        print("\n[信息] 未检测到完整的 PaddleOCR 环境")
        print("       这是正常的，如果您只部署了 L2W1 项目代码")
        print("\n[说明] 真实模型测试需要:")
        print("       1. 完整的 PaddleOCR 仓库")
        print("       2. 预训练模型权重")
        print("       3. pip install paddleocr")
        print("\n[建议] 使用模拟数据演示验证核心算法:")
        print("       python demo_logits_hook.py --demo_only")
        print("\n" + "-" * 60)
        print("自动切换到模拟数据演示模式...")
        print("-" * 60 + "\n")
        demo_logits_extraction()
        return

    try:
        import cv2

        # 尝试导入完整 PaddleOCR 环境
        PADDLEOCR_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, ".."))
        sys.path.insert(0, PADDLEOCR_ROOT)
        sys.path.insert(0, os.path.join(PADDLEOCR_ROOT, "tools/infer"))

        import tools.infer.utility as utility
        from predict_rec_modified import TextRecognizerWithLogits

        print("=" * 60)
        print("L2W1 真实模型 Logits Hooking 演示")
        print("=" * 60)

        # 初始化识别器
        print("\n[初始化] 加载 TextRecognizerWithLogits...")
        text_recognizer = TextRecognizerWithLogits(args)
        print("  -> 加载成功!")

        # 准备测试图像
        if hasattr(args, "image_dir") and os.path.exists(args.image_dir):
            from ppocr.utils.utility import get_image_file_list, check_and_read

            image_files = get_image_file_list(args.image_dir)[:3]  # 取前3张
            img_list = []

            for img_file in image_files:
                img, flag, _ = check_and_read(img_file)
                if not flag:
                    img = cv2.imread(img_file)
                if img is not None:
                    img_list.append(img)

            if img_list:
                print(f"\n[推理] 处理 {len(img_list)} 张图像...")

                # 执行推理
                output = text_recognizer(img_list)

                print("\n[结果] 推理输出:")
                print(f"  - 返回类型: {type(output)}")
                print(f"  - 耗时: {output['elapsed_time']:.3f}s")

                for i, (result, logits) in enumerate(
                    zip(output["results"], output["logits"])
                ):
                    text, conf = result[0] if isinstance(result, list) else result
                    print(f"\n  样本 {i + 1}:")
                    print(f"    - 识别文本: '{text}'")
                    print(f"    - 置信度: {conf:.4f}")

                    if logits is not None:
                        print(f"    - Logits 形状: {logits.shape}")
                        print(
                            f"      (Seq_Len={logits.shape[0]}, Vocab_Size={logits.shape[1]})"
                        )

                        # 计算熵
                        entropy = compute_visual_entropy(logits)
                        print(f"    - 平均熵: {np.mean(entropy):.4f}")
                        print(
                            f"    - 熵范围: [{np.min(entropy):.4f}, {np.max(entropy):.4f}]"
                        )
                    else:
                        print("    - Logits: None (非 CTC 算法)")

                print("\n" + "=" * 60)
                print("真实模型验证完成!")
                print("=" * 60)
            else:
                print("未找到有效图像，使用模拟数据演示")
                demo_logits_extraction()
        else:
            print("未指定 image_dir 或目录不存在，使用模拟数据演示")
            demo_logits_extraction()

    except ImportError as e:
        print(f"\n[信息] PaddleOCR 工具模块不可用: {e}")
        print("       自动切换到模拟数据演示...\n")
        demo_logits_extraction()


def main():
    """主函数"""

    parser = argparse.ArgumentParser(description="L2W1 Logits Hooking Demo")
    parser.add_argument("--demo_only", action="store_true", help="仅运行模拟数据演示")
    parser.add_argument(
        "--image_dir", type=str, default="./test_images", help="测试图像目录"
    )
    parser.add_argument("--rec_model_dir", type=str, default="", help="识别模型目录")

    # PaddleOCR 相关参数（如果需要真实模型测试）
    parser.add_argument("--rec_image_shape", type=str, default="3,48,320")
    parser.add_argument("--rec_batch_num", type=int, default=6)
    parser.add_argument("--rec_algorithm", type=str, default="SVTR_LCNet")
    parser.add_argument(
        "--rec_char_dict_path", type=str, default="./ppocr/utils/ppocr_keys_v1.txt"
    )
    parser.add_argument("--use_space_char", type=bool, default=True)
    parser.add_argument("--max_text_length", type=int, default=25)
    parser.add_argument("--rec_image_inverse", type=bool, default=False)
    parser.add_argument("--use_gpu", type=bool, default=True)
    parser.add_argument("--use_onnx", type=bool, default=False)
    parser.add_argument("--benchmark", type=bool, default=False)
    parser.add_argument("--return_word_box", type=bool, default=False)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--warmup", type=bool, default=False)
    parser.add_argument("--save_log_path", type=str, default="./output/")

    args = parser.parse_args()

    if args.demo_only:
        demo_logits_extraction()
    else:
        demo_with_real_model(args)


if __name__ == "__main__":
    main()
