# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Modified for L2W1 Project - Logits Hooking for CTC Entropy Calculation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
L2W1 源码外科手术模块 (Surgical Hooking Module)

关键修改点:
1. 在 self.predictor.run() 之后拦截原始 Logits Tensor
2. 使用 deepcopy 防止 PaddlePredictor 内存复用机制覆盖数据
3. 返回字典格式: {'text': rec_text, 'conf': rec_score, 'logits': raw_logits}

Tensor 规格:
- raw_logits 形状: [Batch, Seq_Len, Vocab_Size]
- Seq_Len: 通常为 80 左右 (PP-OCR 默认)
- Vocab_Size: 约 6000+ (中文词表)
"""

import os
import sys
from PIL import Image
from copy import deepcopy

__dir__ = os.path.dirname(os.path.abspath(__file__))

# ========== L2W1 关键修改: 使用本地精简版模块 ==========
# 添加 L2W1 项目根目录到路径，导入本地 tools 和 ppocr 模块
L2W1_ROOT = os.path.abspath(os.path.join(__dir__, "../.."))
sys.path.insert(0, L2W1_ROOT)

os.environ["FLAGS_allocator_strategy"] = "auto_growth"

import cv2
import numpy as np
import math
import time
import traceback
import paddle

# 从 L2W1 本地模块导入 (不再依赖外部 PaddleOCR 代码库)
import tools.infer.utility as utility
from ppocr.postprocess import build_post_process
from ppocr.utils.logging import get_logger
from ppocr.utils.utility import get_image_file_list, check_and_read

logger = get_logger()


# =============================================================================
# SH-DA++ v4.0 高性能工具函数
# =============================================================================


def compute_softmax(
    logits: np.ndarray,
    axis: int = -1,
    temperature: float = 1.0,
    debug: bool = False,
) -> np.ndarray:
    """
    智能 Softmax 计算 (Smart Softmax with Auto-Detection)

    关键特性：
        - 自动检测输入是否已经是概率（PP-OCRv5 输出的就是概率，不是原始 logits）
        - 如果输入已是概率（范围 [0, 1]），直接返回，跳过 Softmax
        - 如果输入是原始 logits，才进行 Softmax + 温度缩放

    PP-OCRv5 特性：
        PaddleOCR 的 CTC Head 已经内置了 Softmax 层，模型输出的 `outputs[0]`
        已经是概率分布（每行和为 1），而不是原始 logits。因此不应再次应用 Softmax。

    实现公式 (仅当输入是原始 logits 时):
        Softmax(x_i / T) = exp((x_i - max(x)) / T) / Σ exp((x_j - max(x)) / T)

    Args:
        logits: 输入张量，可能是：
                - 原始 logits (范围可能 < 0 或 > 1)
                - 已是概率 (范围 [0, 1]，每行和为 1)
        axis: 沿哪个轴计算 Softmax，默认为最后一轴 (-1)
        temperature: 温度参数 T，默认 1.0（仅对原始 logits 有效）
        debug: 是否打印调试信息

    Returns:
        E: 归一化后的 Emission 矩阵，E ∈ [0,1]^{T×C}
           满足 Σ_c E[t,c] = 1 对于所有时间步 t
    """
    # 检测输入是否已经是概率分布
    # PP-OCRv5 的输出已经是 Softmax 后的概率
    input_min = logits.min()
    input_max = logits.max()

    # 计算每行的和，检查是否接近 1（概率分布的特征）
    row_sums = np.sum(logits, axis=axis)
    is_probability = (
        input_min >= -0.001  # 概率最小值应 >= 0
        and input_max <= 1.001  # 概率最大值应 <= 1
        and np.allclose(row_sums, 1.0, atol=0.01)  # 每行和应接近 1
    )

    if debug:
        print(
            f"[compute_softmax] 输入范围: min={input_min:.4f}, max={input_max:.4f}, "
            f"row_sum_mean={row_sums.mean():.4f}"
        )
        if is_probability:
            print(f"[compute_softmax] ✓ 检测到输入已是概率分布，跳过 Softmax")
        else:
            print(
                f"[compute_softmax] ✓ 检测到输入是原始 logits，应用 Softmax (T={temperature})"
            )

    # 如果输入已经是概率，直接返回
    if is_probability:
        return logits

    # 以下是原始 logits 的处理逻辑
    # 温度参数验证
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    # Step 1: 减去最大值以保证数值稳定性 (防止 exp 溢出)
    logits_max = np.max(logits, axis=axis, keepdims=True)
    logits_shifted = logits - logits_max

    # Step 2: 应用温度缩放 (Temperature Scaling)
    logits_scaled = logits_shifted / temperature

    # Step 3: 计算 exp
    exp_logits = np.exp(logits_scaled)

    # Step 4: 归一化
    sum_exp = np.sum(exp_logits, axis=axis, keepdims=True)
    softmax_output = exp_logits / sum_exp

    return softmax_output


def calculate_boundary_stats(
    E: np.ndarray,
    blank_id: int,
    rho: float = 0.1,
) -> dict:
    """
    计算 CTC Emission 矩阵的边界统计量 (Boundary Statistics)

    该函数用于 SH-DA++ v4.0 路由器的边界风险检测，通过分析
    序列首尾区域的 blank 概率分布来判断是否存在边界字符丢失风险。

    核心假设:
        如果边界区域的 blank 概率异常高，说明模型可能"跳过"了边界字符。

    边界区域定义 (根据 SH-DA++ v4.0 规范):
        L = [0, floor(ρ × T)]     # 左边界区域
        R = [ceil((1-ρ) × T), T]  # 右边界区域

    Args:
        E: 归一化后的 Emission 矩阵，形状为 [T, C]
           其中 T 为序列长度，C 为词表大小
           要求 E ∈ [0,1]^{T×C} 且每行和为 1
        blank_id: CTC blank 符号的索引，PP-OCR 默认为 0
        rho: 边界区域比例因子，默认 0.1 (即首尾各 10%)
             有效范围: (0, 0.5)

    Returns:
        dict: 包含以下统计量的字典:
            - blank_mean_L (float): 左边界区域 blank 概率均值
            - blank_mean_R (float): 右边界区域 blank 概率均值
            - blank_peak_L (float): 左边界区域 blank 概率峰值
            - blank_peak_R (float): 右边界区域 blank 概率峰值
            - T (int): 序列总长度
            - L_range (tuple): 左边界区域索引范围 [start, end)
            - R_range (tuple): 右边界区域索引范围 [start, end)
            - valid (bool): 计算是否有效（T >= 2 时为 True）

    边界情况处理:
        - T < 2: 返回 valid=False，所有统计量置为 0.0
        - rho * T < 1: 左/右边界区域至少包含 1 个时间步

    Complexity:
        Time: O(ρ × T)
        Space: O(1) 额外空间

    Example:
        >>> E = compute_softmax(np.random.randn(80, 6625))
        >>> stats = calculate_boundary_stats(E, blank_id=0, rho=0.1)
        >>> print(f"Left boundary blank mean: {stats['blank_mean_L']:.4f}")
        >>> print(f"Right boundary blank peak: {stats['blank_peak_R']:.4f}")

    References:
        [1] SH-DA++ v4.0 Specification, Stage 0, Section B
    """
    T = E.shape[0]

    # ========== 边界情况处理 ==========
    if T < 2:
        return {
            "blank_mean_L": 0.0,
            "blank_mean_R": 0.0,
            "blank_peak_L": 0.0,
            "blank_peak_R": 0.0,
            "T": T,
            "L_range": (0, 0),
            "R_range": (0, 0),
            "valid": False,
        }

    # ========== 计算边界区域索引 ==========
    # L = [0, floor(ρ × T)]
    L_end = max(1, int(rho * T))  # 至少包含 1 个时间步

    # R = [ceil((1-ρ) × T), T]
    R_start = min(T - 1, int(np.ceil((1 - rho) * T)))  # 至少包含 1 个时间步

    # 确保 R_start > L_end 以避免区域重叠（当 T 很小时）
    if R_start <= L_end:
        # T 太小，左右区域会重叠，采用平分策略
        mid = T // 2
        L_end = max(1, mid)
        R_start = min(T - 1, mid)

    # ========== 提取边界区域的 blank 概率 ==========
    # 左边界: E[0:L_end, blank_id]
    blank_probs_L = E[0:L_end, blank_id]

    # 右边界: E[R_start:T, blank_id]
    blank_probs_R = E[R_start:T, blank_id]

    # ========== 计算统计量 ==========
    blank_mean_L = float(np.mean(blank_probs_L)) if len(blank_probs_L) > 0 else 0.0
    blank_mean_R = float(np.mean(blank_probs_R)) if len(blank_probs_R) > 0 else 0.0
    blank_peak_L = float(np.max(blank_probs_L)) if len(blank_probs_L) > 0 else 0.0
    blank_peak_R = float(np.max(blank_probs_R)) if len(blank_probs_R) > 0 else 0.0

    return {
        "blank_mean_L": blank_mean_L,
        "blank_mean_R": blank_mean_R,
        "blank_peak_L": blank_peak_L,
        "blank_peak_R": blank_peak_R,
        "T": T,
        "L_range": (0, L_end),
        "R_range": (R_start, T),
        "valid": True,
    }


class TextRecognizerWithLogits(object):
    """
    L2W1 增强版文本识别器 - 支持 Logits 导出

    与原版 TextRecognizer 的关键区别:
    1. __call__ 返回字典格式，包含原始 logits
    2. 支持获取 CTC 解码前的概率分布，用于熵计算
    """

    def __init__(self, args, logger=None):
        if os.path.exists(f"{args.rec_model_dir}/inference.yml"):
            model_config = utility.load_config(f"{args.rec_model_dir}/inference.yml")
            model_name = model_config.get("Global", {}).get("model_name", "")
            if model_name and model_name not in [
                "PP-OCRv5_mobile_rec",
                "PP-OCRv5_server_rec",
                "korean_PP-OCRv5_mobile_rec",
                "eslav_PP-OCRv5_mobile_rec",
                "latin_PP-OCRv5_mobile_rec",
                "en_PP-OCRv5_mobile_rec",
                "th_PP-OCRv5_mobile_rec",
                "el_PP-OCRv5_mobile_rec",
            ]:
                raise ValueError(
                    f"{model_name} is not supported. Please check if the model is supported by the PaddleOCR wheel."
                )

            if args.rec_char_dict_path == "./ppocr/utils/ppocrv5_dict.txt":
                rec_char_list = model_config.get("PostProcess", {}).get(
                    "character_dict", []
                )
                if rec_char_list:
                    new_rec_char_dict_path = f"{args.rec_model_dir}/ppocr_keys.txt"
                    with open(new_rec_char_dict_path, "w", encoding="utf-8") as f:
                        f.writelines([char + "\n" for char in rec_char_list])
                    args.rec_char_dict_path = new_rec_char_dict_path

        if logger is None:
            logger = get_logger()
        self.rec_image_shape = [int(v) for v in args.rec_image_shape.split(",")]
        self.rec_batch_num = args.rec_batch_num
        self.rec_algorithm = args.rec_algorithm

        # 后处理参数配置
        postprocess_params = {
            "name": "CTCLabelDecode",
            "character_dict_path": args.rec_char_dict_path,
            "use_space_char": args.use_space_char,
        }
        if self.rec_algorithm == "SRN":
            postprocess_params = {
                "name": "SRNLabelDecode",
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char,
            }
        elif self.rec_algorithm == "RARE":
            postprocess_params = {
                "name": "AttnLabelDecode",
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char,
            }
        elif self.rec_algorithm == "NRTR":
            postprocess_params = {
                "name": "NRTRLabelDecode",
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char,
            }
        elif self.rec_algorithm == "SAR":
            postprocess_params = {
                "name": "SARLabelDecode",
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char,
            }
        elif self.rec_algorithm == "VisionLAN":
            postprocess_params = {
                "name": "VLLabelDecode",
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char,
                "max_text_length": args.max_text_length,
            }
        elif self.rec_algorithm == "ViTSTR":
            postprocess_params = {
                "name": "ViTSTRLabelDecode",
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char,
            }
        elif self.rec_algorithm == "ABINet":
            postprocess_params = {
                "name": "ABINetLabelDecode",
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char,
            }
        elif self.rec_algorithm == "SPIN":
            postprocess_params = {
                "name": "SPINLabelDecode",
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char,
            }
        elif self.rec_algorithm == "RobustScanner":
            postprocess_params = {
                "name": "SARLabelDecode",
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char,
                "rm_symbol": True,
            }
        elif self.rec_algorithm == "RFL":
            postprocess_params = {
                "name": "RFLLabelDecode",
                "character_dict_path": None,
                "use_space_char": args.use_space_char,
            }
        elif self.rec_algorithm == "SATRN":
            postprocess_params = {
                "name": "SATRNLabelDecode",
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char,
                "rm_symbol": True,
            }
        elif self.rec_algorithm in ["CPPD", "CPPDPadding"]:
            postprocess_params = {
                "name": "CPPDLabelDecode",
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char,
                "rm_symbol": True,
            }
        elif self.rec_algorithm == "PREN":
            postprocess_params = {"name": "PRENLabelDecode"}
        elif self.rec_algorithm == "CAN":
            self.inverse = args.rec_image_inverse
            postprocess_params = {
                "name": "CANLabelDecode",
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char,
            }
        elif self.rec_algorithm == "LaTeXOCR":
            postprocess_params = {
                "name": "LaTeXOCRDecode",
                "rec_char_dict_path": args.rec_char_dict_path,
            }
        elif self.rec_algorithm == "ParseQ":
            postprocess_params = {
                "name": "ParseQLabelDecode",
                "character_dict_path": args.rec_char_dict_path,
                "use_space_char": args.use_space_char,
            }
        self.postprocess_op = build_post_process(postprocess_params)
        self.postprocess_params = postprocess_params

        # ========== SH-DA++ v4.0: 保存字符表和 blank_id ==========
        # 从 postprocess_op 动态获取 blank_id (CTCLabelDecode 约定为 0)
        # 严格遵循 CTCLabelDecode.get_ignored_tokens() 的约定
        self.blank_id = 0  # 默认值，CTCLabelDecode 的 blank token 固定为 0
        if hasattr(self.postprocess_op, "get_ignored_tokens"):
            try:
                ignored_tokens = self.postprocess_op.get_ignored_tokens()
                if ignored_tokens and len(ignored_tokens) > 0:
                    # CTCLabelDecode.get_ignored_tokens() 返回 [0]
                    self.blank_id = int(ignored_tokens[0])
            except Exception as e:
                logger.warning(
                    f"Failed to get ignored_tokens from postprocess_op: {e}. Using default blank_id=0"
                )
                self.blank_id = 0

        self.rho = 0.1  # 边界区域比例因子

        # SH-DA++ v4.0: 温度参数用于 Softmax 缩放
        # 当 logits 值范围较小时（如 [-1, 1]），标准 Softmax 会导致概率趋于均匀
        # 通过设置 T < 1 可以有效放大差异，恢复真实的置信度分布
        # 默认值 0.1 适用于 PP-OCRv5 的 logits 范围
        self.softmax_temperature = getattr(args, "softmax_temperature", 0.1)

        # 从 postprocess_op 获取字符表用于 Top-2 索引转换
        self.character_list = None
        if hasattr(self.postprocess_op, "character"):
            self.character_list = self.postprocess_op.character
        # ==========================================================
        (
            self.predictor,
            self.input_tensor,
            self.output_tensors,
            self.config,
        ) = utility.create_predictor(args, "rec", logger)
        self.benchmark = args.benchmark
        self.use_onnx = args.use_onnx
        if args.benchmark:
            import auto_log

            pid = os.getpid()
            gpu_id = utility.get_infer_gpuid()
            self.autolog = auto_log.AutoLogger(
                model_name="rec",
                model_precision=args.precision,
                batch_size=args.rec_batch_num,
                data_shape="dynamic",
                save_path=None,  # not used if logger is not None
                inference_config=self.config,
                pids=pid,
                process_name=None,
                gpu_ids=gpu_id if args.use_gpu else None,
                time_keys=["preprocess_time", "inference_time", "postprocess_time"],
                warmup=0,
                logger=logger,
            )
        self.return_word_box = args.return_word_box

    def resize_norm_img(self, img, max_wh_ratio):
        imgC, imgH, imgW = self.rec_image_shape
        if self.rec_algorithm == "NRTR" or self.rec_algorithm == "ViTSTR":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # return padding_im
            image_pil = Image.fromarray(np.uint8(img))
            if self.rec_algorithm == "ViTSTR":
                img = image_pil.resize([imgW, imgH], Image.BICUBIC)
            else:
                img = image_pil.resize([imgW, imgH], Image.Resampling.LANCZOS)
            img = np.array(img)
            norm_img = np.expand_dims(img, -1)
            norm_img = norm_img.transpose((2, 0, 1))
            if self.rec_algorithm == "ViTSTR":
                norm_img = norm_img.astype(np.float32) / 255.0
            else:
                norm_img = norm_img.astype(np.float32) / 128.0 - 1.0
            return norm_img
        elif self.rec_algorithm == "RFL":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(img, (imgW, imgH), interpolation=cv2.INTER_CUBIC)
            resized_image = resized_image.astype("float32")
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
            resized_image -= 0.5
            resized_image /= 0.5
            return resized_image

        assert imgC == img.shape[2]
        imgW = int((imgH * max_wh_ratio))
        if self.use_onnx:
            w = self.input_tensor.shape[3:][0]
            if isinstance(w, str):
                pass
            elif w is not None and w > 0:
                imgW = w
        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        if self.rec_algorithm == "RARE":
            if resized_w > self.rec_image_shape[2]:
                resized_w = self.rec_image_shape[2]
            imgW = self.rec_image_shape[2]
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def resize_norm_img_vl(self, img, image_shape):
        imgC, imgH, imgW = image_shape
        img = img[:, :, ::-1]  # bgr2rgb
        resized_image = cv2.resize(img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        return resized_image

    def resize_norm_img_srn(self, img, image_shape):
        imgC, imgH, imgW = image_shape

        img_black = np.zeros((imgH, imgW))
        im_hei = img.shape[0]
        im_wid = img.shape[1]

        if im_wid <= im_hei * 1:
            img_new = cv2.resize(img, (imgH * 1, imgH))
        elif im_wid <= im_hei * 2:
            img_new = cv2.resize(img, (imgH * 2, imgH))
        elif im_wid <= im_hei * 3:
            img_new = cv2.resize(img, (imgH * 3, imgH))
        else:
            img_new = cv2.resize(img, (imgW, imgH))

        img_np = np.asarray(img_new)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        img_black[:, 0 : img_np.shape[1]] = img_np
        img_black = img_black[:, :, np.newaxis]

        row, col, c = img_black.shape
        c = 1

        return np.reshape(img_black, (c, row, col)).astype(np.float32)

    def srn_other_inputs(self, image_shape, num_heads, max_text_length):
        imgC, imgH, imgW = image_shape
        feature_dim = int((imgH / 8) * (imgW / 8))

        encoder_word_pos = (
            np.array(range(0, feature_dim)).reshape((feature_dim, 1)).astype("int64")
        )
        gsrm_word_pos = (
            np.array(range(0, max_text_length))
            .reshape((max_text_length, 1))
            .astype("int64")
        )

        gsrm_attn_bias_data = np.ones((1, max_text_length, max_text_length))
        gsrm_slf_attn_bias1 = np.triu(gsrm_attn_bias_data, 1).reshape(
            [-1, 1, max_text_length, max_text_length]
        )
        gsrm_slf_attn_bias1 = np.tile(gsrm_slf_attn_bias1, [1, num_heads, 1, 1]).astype(
            "float32"
        ) * [-1e9]

        gsrm_slf_attn_bias2 = np.tril(gsrm_attn_bias_data, -1).reshape(
            [-1, 1, max_text_length, max_text_length]
        )
        gsrm_slf_attn_bias2 = np.tile(gsrm_slf_attn_bias2, [1, num_heads, 1, 1]).astype(
            "float32"
        ) * [-1e9]

        encoder_word_pos = encoder_word_pos[np.newaxis, :]
        gsrm_word_pos = gsrm_word_pos[np.newaxis, :]

        return [
            encoder_word_pos,
            gsrm_word_pos,
            gsrm_slf_attn_bias1,
            gsrm_slf_attn_bias2,
        ]

    def process_image_srn(self, img, image_shape, num_heads, max_text_length):
        norm_img = self.resize_norm_img_srn(img, image_shape)
        norm_img = norm_img[np.newaxis, :]

        [
            encoder_word_pos,
            gsrm_word_pos,
            gsrm_slf_attn_bias1,
            gsrm_slf_attn_bias2,
        ] = self.srn_other_inputs(image_shape, num_heads, max_text_length)

        gsrm_slf_attn_bias1 = gsrm_slf_attn_bias1.astype(np.float32)
        gsrm_slf_attn_bias2 = gsrm_slf_attn_bias2.astype(np.float32)
        encoder_word_pos = encoder_word_pos.astype(np.int64)
        gsrm_word_pos = gsrm_word_pos.astype(np.int64)

        return (
            norm_img,
            encoder_word_pos,
            gsrm_word_pos,
            gsrm_slf_attn_bias1,
            gsrm_slf_attn_bias2,
        )

    def resize_norm_img_sar(self, img, image_shape, width_downsample_ratio=0.25):
        imgC, imgH, imgW_min, imgW_max = image_shape
        h = img.shape[0]
        w = img.shape[1]
        valid_ratio = 1.0
        # make sure new_width is an integral multiple of width_divisor.
        width_divisor = int(1 / width_downsample_ratio)
        # resize
        ratio = w / float(h)
        resize_w = math.ceil(imgH * ratio)
        if resize_w % width_divisor != 0:
            resize_w = round(resize_w / width_divisor) * width_divisor
        if imgW_min is not None:
            resize_w = max(imgW_min, resize_w)
        if imgW_max is not None:
            valid_ratio = min(1.0, 1.0 * resize_w / imgW_max)
            resize_w = min(imgW_max, resize_w)
        resized_image = cv2.resize(img, (resize_w, imgH))
        resized_image = resized_image.astype("float32")
        # norm
        if image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        resize_shape = resized_image.shape
        padding_im = -1.0 * np.ones((imgC, imgH, imgW_max), dtype=np.float32)
        padding_im[:, :, 0:resize_w] = resized_image
        pad_shape = padding_im.shape

        return padding_im, resize_shape, pad_shape, valid_ratio

    def resize_norm_img_spin(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # return padding_im
        img = cv2.resize(img, tuple([100, 32]), cv2.INTER_CUBIC)
        img = np.array(img, np.float32)
        img = np.expand_dims(img, -1)
        img = img.transpose((2, 0, 1))
        mean = [127.5]
        std = [127.5]
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        mean = np.float32(mean.reshape(1, -1))
        stdinv = 1 / np.float32(std.reshape(1, -1))
        img -= mean
        img *= stdinv
        return img

    def resize_norm_img_svtr(self, img, image_shape):
        imgC, imgH, imgW = image_shape
        max_wh_ratio = imgW * 1.0 / imgH
        h, w = img.shape[0], img.shape[1]
        ratio = w * 1.0 / h
        max_wh_ratio = min(max(max_wh_ratio, ratio), max_wh_ratio)
        imgW = int(imgH * max_wh_ratio)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def resize_norm_img_cppd_padding(
        self, img, image_shape, padding=True, interpolation=cv2.INTER_LINEAR
    ):
        imgC, imgH, imgW = image_shape
        h = img.shape[0]
        w = img.shape[1]
        if not padding:
            resized_image = cv2.resize(img, (imgW, imgH), interpolation=interpolation)
            resized_w = imgW
        else:
            ratio = w / float(h)
            if math.ceil(imgH * ratio) > imgW:
                resized_w = imgW
            else:
                resized_w = int(math.ceil(imgH * ratio))
            resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype("float32")
        if image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image

        return padding_im

    def resize_norm_img_abinet(self, img, image_shape):
        imgC, imgH, imgW = image_shape

        resized_image = cv2.resize(img, (imgW, imgH), interpolation=cv2.INTER_LINEAR)
        resized_image = resized_image.astype("float32")
        resized_image = resized_image / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        resized_image = (resized_image - mean[None, None, ...]) / std[None, None, ...]
        resized_image = resized_image.transpose((2, 0, 1))
        resized_image = resized_image.astype("float32")

        return resized_image

    def norm_img_can(self, img, image_shape):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # CAN only predict gray scale image

        if self.inverse:
            img = 255 - img

        if self.rec_image_shape[0] == 1:
            h, w = img.shape
            _, imgH, imgW = self.rec_image_shape
            if h < imgH or w < imgW:
                padding_h = max(imgH - h, 0)
                padding_w = max(imgW - w, 0)
                img_padded = np.pad(
                    img,
                    ((0, padding_h), (0, padding_w)),
                    "constant",
                    constant_values=(255),
                )
                img = img_padded

        img = np.expand_dims(img, 0) / 255.0  # h,w,c -> c,h,w
        img = img.astype("float32")

        return img

    def pad_(self, img, divable=32):
        threshold = 128
        data = np.array(img.convert("LA"))
        if data[..., -1].var() == 0:
            data = (data[..., 0]).astype(np.uint8)
        else:
            data = (255 - data[..., -1]).astype(np.uint8)
        data = (data - data.min()) / (data.max() - data.min()) * 255
        if data.mean() > threshold:
            # To invert the text to white
            gray = 255 * (data < threshold).astype(np.uint8)
        else:
            gray = 255 * (data > threshold).astype(np.uint8)
            data = 255 - data

        coords = cv2.findNonZero(gray)  # Find all non-zero points (text)
        a, b, w, h = cv2.boundingRect(coords)  # Find minimum spanning bounding box
        rect = data[b : b + h, a : a + w]
        im = Image.fromarray(rect).convert("L")
        dims = []
        for x in [w, h]:
            div, mod = divmod(x, divable)
            dims.append(divable * (div + (1 if mod > 0 else 0)))
        padded = Image.new("L", dims, 255)
        padded.paste(im, (0, 0, im.size[0], im.size[1]))
        return padded

    def minmax_size_(
        self,
        img,
        max_dimensions,
        min_dimensions,
    ):
        if max_dimensions is not None:
            ratios = [a / b for a, b in zip(img.size, max_dimensions)]
            if any([r > 1 for r in ratios]):
                size = np.array(img.size) // max(ratios)
                img = img.resize(tuple(size.astype(int)), Image.BILINEAR)
        if min_dimensions is not None:
            # hypothesis: there is a dim in img smaller than min_dimensions, and return a proper dim >= min_dimensions
            padded_size = [
                max(img_dim, min_dim)
                for img_dim, min_dim in zip(img.size, min_dimensions)
            ]
            if padded_size != list(img.size):  # assert hypothesis
                padded_im = Image.new("L", padded_size, 255)
                padded_im.paste(img, img.getbbox())
                img = padded_im
        return img

    def norm_img_latexocr(self, img):
        # CAN only predict gray scale image
        shape = (1, 1, 3)
        mean = [0.7931, 0.7931, 0.7931]
        std = [0.1738, 0.1738, 0.1738]
        scale = np.float32(1.0 / 255.0)
        min_dimensions = [32, 32]
        max_dimensions = [672, 192]
        mean = np.array(mean).reshape(shape).astype("float32")
        std = np.array(std).reshape(shape).astype("float32")

        im_h, im_w = img.shape[:2]
        if (
            min_dimensions[0] <= im_w <= max_dimensions[0]
            and min_dimensions[1] <= im_h <= max_dimensions[1]
        ):
            pass
        else:
            img = Image.fromarray(np.uint8(img))
            img = self.minmax_size_(self.pad_(img), max_dimensions, min_dimensions)
            img = np.array(img)
            im_h, im_w = img.shape[:2]
            img = np.dstack([img, img, img])
        img = (img.astype("float32") * scale - mean) / std
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        divide_h = math.ceil(im_h / 16) * 16
        divide_w = math.ceil(im_w / 16) * 16
        img = np.pad(
            img, ((0, divide_h - im_h), (0, divide_w - im_w)), constant_values=(1, 1)
        )
        img = img[:, :, np.newaxis].transpose(2, 0, 1)
        img = img.astype("float32")
        return img

    def __call__(self, img_list):
        """
        L2W1 核心修改: 增强版推理接口，支持 Logits 导出

        Args:
            img_list: 输入图像列表

        Returns:
            dict: {
                'results': List[Tuple[str, float]] - 识别结果列表 [(text, confidence), ...]
                'logits': List[np.ndarray] - 原始 logits 列表，形状为 [Seq_Len, Vocab_Size]
                'elapsed_time': float - 推理耗时
            }

        Note:
            - logits 在 CTC Decode 之前拦截，用于熵计算
            - 使用 deepcopy 防止 PaddlePredictor 内存复用覆盖
        """
        img_num = len(img_list)
        # Calculate the aspect ratio of all text bars
        width_list = []
        for img in img_list:
            width_list.append(img.shape[1] / float(img.shape[0]))
        # Sorting can speed up the recognition process
        indices = np.argsort(np.array(width_list))
        rec_res = [["", 0.0]] * img_num

        # ========== SH-DA++ v4.0: Emission 矩阵与边界统计量存储 ==========
        all_boundary_stats = [None] * img_num  # 每个样本的边界统计量
        all_top2_info = [None] * img_num  # 每个样本的 Top-2 信息
        lat_router_ms_list = [0.0] * img_num  # 路由器耗时 (毫秒)
        # ==================================================================

        batch_num = self.rec_batch_num
        st = time.time()
        if self.benchmark:
            self.autolog.times.start()
        for beg_img_no in range(0, img_num, batch_num):
            end_img_no = min(img_num, beg_img_no + batch_num)
            norm_img_batch = []
            if self.rec_algorithm == "SRN":
                encoder_word_pos_list = []
                gsrm_word_pos_list = []
                gsrm_slf_attn_bias1_list = []
                gsrm_slf_attn_bias2_list = []
            if self.rec_algorithm == "SAR":
                valid_ratios = []
            imgC, imgH, imgW = self.rec_image_shape[:3]
            max_wh_ratio = imgW / imgH
            wh_ratio_list = []
            for ino in range(beg_img_no, end_img_no):
                h, w = img_list[indices[ino]].shape[0:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
                wh_ratio_list.append(wh_ratio)
            for ino in range(beg_img_no, end_img_no):
                if self.rec_algorithm == "SAR":
                    norm_img, _, _, valid_ratio = self.resize_norm_img_sar(
                        img_list[indices[ino]], self.rec_image_shape
                    )
                    norm_img = norm_img[np.newaxis, :]
                    valid_ratio = np.expand_dims(valid_ratio, axis=0)
                    valid_ratios.append(valid_ratio)
                    norm_img_batch.append(norm_img)
                elif self.rec_algorithm == "SRN":
                    norm_img = self.process_image_srn(
                        img_list[indices[ino]], self.rec_image_shape, 8, 25
                    )
                    encoder_word_pos_list.append(norm_img[1])
                    gsrm_word_pos_list.append(norm_img[2])
                    gsrm_slf_attn_bias1_list.append(norm_img[3])
                    gsrm_slf_attn_bias2_list.append(norm_img[4])
                    norm_img_batch.append(norm_img[0])
                elif self.rec_algorithm in ["SVTR", "SATRN", "ParseQ", "CPPD"]:
                    norm_img = self.resize_norm_img_svtr(
                        img_list[indices[ino]], self.rec_image_shape
                    )
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
                elif self.rec_algorithm in ["CPPDPadding"]:
                    norm_img = self.resize_norm_img_cppd_padding(
                        img_list[indices[ino]], self.rec_image_shape
                    )
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
                elif self.rec_algorithm in ["VisionLAN", "PREN"]:
                    norm_img = self.resize_norm_img_vl(
                        img_list[indices[ino]], self.rec_image_shape
                    )
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
                elif self.rec_algorithm == "SPIN":
                    norm_img = self.resize_norm_img_spin(img_list[indices[ino]])
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
                elif self.rec_algorithm == "ABINet":
                    norm_img = self.resize_norm_img_abinet(
                        img_list[indices[ino]], self.rec_image_shape
                    )
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
                elif self.rec_algorithm == "RobustScanner":
                    norm_img, _, _, valid_ratio = self.resize_norm_img_sar(
                        img_list[indices[ino]],
                        self.rec_image_shape,
                        width_downsample_ratio=0.25,
                    )
                    norm_img = norm_img[np.newaxis, :]
                    valid_ratio = np.expand_dims(valid_ratio, axis=0)
                    valid_ratios = []
                    valid_ratios.append(valid_ratio)
                    norm_img_batch.append(norm_img)
                    word_positions_list = []
                    word_positions = np.array(range(0, 40)).astype("int64")
                    word_positions = np.expand_dims(word_positions, axis=0)
                    word_positions_list.append(word_positions)
                elif self.rec_algorithm == "CAN":
                    norm_img = self.norm_img_can(img_list[indices[ino]], max_wh_ratio)
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
                    norm_image_mask = np.ones(norm_img.shape, dtype="float32")
                    word_label = np.ones([1, 36], dtype="int64")
                    norm_img_mask_batch = []
                    word_label_list = []
                    norm_img_mask_batch.append(norm_image_mask)
                    word_label_list.append(word_label)
                elif self.rec_algorithm == "LaTeXOCR":
                    norm_img = self.norm_img_latexocr(img_list[indices[ino]])
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
                else:
                    norm_img = self.resize_norm_img(
                        img_list[indices[ino]], max_wh_ratio
                    )
                    norm_img = norm_img[np.newaxis, :]
                    norm_img_batch.append(norm_img)
            norm_img_batch = np.concatenate(norm_img_batch)
            norm_img_batch = norm_img_batch.copy()
            if self.benchmark:
                self.autolog.times.stamp()

            # ========== SH-DA++ v4.0: 旧的 batch_raw_logits 拦截逻辑已移除 ==========

            if self.rec_algorithm == "SRN":
                encoder_word_pos_list = np.concatenate(encoder_word_pos_list)
                gsrm_word_pos_list = np.concatenate(gsrm_word_pos_list)
                gsrm_slf_attn_bias1_list = np.concatenate(gsrm_slf_attn_bias1_list)
                gsrm_slf_attn_bias2_list = np.concatenate(gsrm_slf_attn_bias2_list)

                inputs = [
                    norm_img_batch,
                    encoder_word_pos_list,
                    gsrm_word_pos_list,
                    gsrm_slf_attn_bias1_list,
                    gsrm_slf_attn_bias2_list,
                ]
                if self.use_onnx:
                    input_dict = {}
                    input_dict[self.input_tensor.name] = norm_img_batch
                    outputs = self.predictor.run(self.output_tensors, input_dict)
                    preds = {"predict": outputs[2]}
                else:
                    input_names = self.predictor.get_input_names()
                    for i in range(len(input_names)):
                        input_tensor = self.predictor.get_input_handle(input_names[i])
                        input_tensor.copy_from_cpu(inputs[i])
                    self.predictor.run()
                    outputs = []
                    for output_tensor in self.output_tensors:
                        output = output_tensor.copy_to_cpu()
                        outputs.append(output)
                    if self.benchmark:
                        self.autolog.times.stamp()
                    preds = {"predict": outputs[2]}
            elif self.rec_algorithm == "SAR":
                valid_ratios = np.concatenate(valid_ratios)
                inputs = [
                    norm_img_batch,
                    np.array([valid_ratios], dtype=np.float32).T,
                ]
                if self.use_onnx:
                    input_dict = {}
                    input_dict[self.input_tensor.name] = norm_img_batch
                    outputs = self.predictor.run(self.output_tensors, input_dict)
                    preds = outputs[0]
                else:
                    input_names = self.predictor.get_input_names()
                    for i in range(len(input_names)):
                        input_tensor = self.predictor.get_input_handle(input_names[i])
                        input_tensor.copy_from_cpu(inputs[i])
                    self.predictor.run()
                    outputs = []
                    for output_tensor in self.output_tensors:
                        output = output_tensor.copy_to_cpu()
                        outputs.append(output)
                    if self.benchmark:
                        self.autolog.times.stamp()
                    preds = outputs[0]
            elif self.rec_algorithm == "RobustScanner":
                valid_ratios = np.concatenate(valid_ratios)
                word_positions_list = np.concatenate(word_positions_list)
                inputs = [norm_img_batch, valid_ratios, word_positions_list]

                if self.use_onnx:
                    input_dict = {}
                    input_dict[self.input_tensor.name] = norm_img_batch
                    outputs = self.predictor.run(self.output_tensors, input_dict)
                    preds = outputs[0]
                else:
                    input_names = self.predictor.get_input_names()
                    for i in range(len(input_names)):
                        input_tensor = self.predictor.get_input_handle(input_names[i])
                        input_tensor.copy_from_cpu(inputs[i])
                    self.predictor.run()
                    outputs = []
                    for output_tensor in self.output_tensors:
                        output = output_tensor.copy_to_cpu()
                        outputs.append(output)
                    if self.benchmark:
                        self.autolog.times.stamp()
                    preds = outputs[0]
            elif self.rec_algorithm == "CAN":
                norm_img_mask_batch = np.concatenate(norm_img_mask_batch)
                word_label_list = np.concatenate(word_label_list)
                inputs = [norm_img_batch, norm_img_mask_batch, word_label_list]
                if self.use_onnx:
                    input_dict = {}
                    input_dict[self.input_tensor.name] = norm_img_batch
                    outputs = self.predictor.run(self.output_tensors, input_dict)
                    preds = outputs
                else:
                    input_names = self.predictor.get_input_names()
                    input_tensor = []
                    for i in range(len(input_names)):
                        input_tensor_i = self.predictor.get_input_handle(input_names[i])
                        input_tensor_i.copy_from_cpu(inputs[i])
                        input_tensor.append(input_tensor_i)
                    self.input_tensor = input_tensor
                    self.predictor.run()
                    outputs = []
                    for output_tensor in self.output_tensors:
                        output = output_tensor.copy_to_cpu()
                        outputs.append(output)
                    if self.benchmark:
                        self.autolog.times.stamp()
                    preds = outputs
            elif self.rec_algorithm == "LaTeXOCR":
                inputs = [norm_img_batch]
                if self.use_onnx:
                    input_dict = {}
                    input_dict[self.input_tensor.name] = norm_img_batch
                    outputs = self.predictor.run(self.output_tensors, input_dict)
                    preds = outputs
                else:
                    input_names = self.predictor.get_input_names()
                    input_tensor = []
                    for i in range(len(input_names)):
                        input_tensor_i = self.predictor.get_input_handle(input_names[i])
                        input_tensor_i.copy_from_cpu(inputs[i])
                        input_tensor.append(input_tensor_i)
                    self.input_tensor = input_tensor
                    self.predictor.run()
                    outputs = []
                    for output_tensor in self.output_tensors:
                        output = output_tensor.copy_to_cpu()
                        outputs.append(output)
                    if self.benchmark:
                        self.autolog.times.stamp()
                    preds = outputs
            else:
                # ========== SH-DA++ v4.0: 默认 CTC 算法路径 ==========
                # 这是 PP-OCRv5 使用的主要路径
                batch_raw_logits = None  # 原始 logits 用于后续处理

                if self.use_onnx:
                    input_dict = {}
                    input_dict[self.input_tensor.name] = norm_img_batch
                    outputs = self.predictor.run(self.output_tensors, input_dict)
                    preds = outputs[0]
                    batch_raw_logits = deepcopy(outputs[0])
                else:
                    self.input_tensor.copy_from_cpu(norm_img_batch)
                    self.predictor.run()
                    outputs = []
                    for output_tensor in self.output_tensors:
                        output = output_tensor.copy_to_cpu()
                        outputs.append(output)
                    if self.benchmark:
                        self.autolog.times.stamp()
                    if len(outputs) != 1:
                        preds = outputs
                        batch_raw_logits = deepcopy(outputs[0])
                    else:
                        preds = outputs[0]
                        batch_raw_logits = deepcopy(outputs[0])
                # =======================================================

            if self.postprocess_params["name"] == "CTCLabelDecode":
                rec_result = self.postprocess_op(
                    preds,
                    return_word_box=self.return_word_box,
                    wh_ratio_list=wh_ratio_list,
                    max_wh_ratio=max_wh_ratio,
                )
            elif self.postprocess_params["name"] == "LaTeXOCRDecode":
                preds = [p.reshape([-1]) for p in preds]
                rec_result = self.postprocess_op(preds)
            else:
                rec_result = self.postprocess_op(preds)
            for rno in range(len(rec_result)):
                rec_res[indices[beg_img_no + rno]] = rec_result[rno]

            # ========== SH-DA++ v4.0: Emission 矩阵计算与边界统计量提取 ==========
            if (
                batch_raw_logits is not None
                and self.postprocess_params["name"] == "CTCLabelDecode"
            ):
                for rno in range(len(rec_result)):
                    # 计时开始
                    router_start = time.perf_counter()

                    original_idx = indices[beg_img_no + rno]

                    # 提取当前样本的 logits: [Seq_Len, Vocab_Size]
                    # 使用 .copy() 防止 Paddle 显存复用导致数据被覆盖
                    sample_logits = batch_raw_logits[rno].copy()
                    T, C = sample_logits.shape

                    # Step 1: 计算 Softmax 得到 Emission 矩阵 E ∈ [0,1]^{T×C}
                    # 使用温度缩放来放大 logits 差异（当原始值范围较小时）
                    # debug 参数：只在全局第 0, 1000, 2000... 个样本打印调试信息
                    global_sample_idx = beg_img_no + rno
                    E = compute_softmax(
                        sample_logits,
                        axis=-1,
                        temperature=self.softmax_temperature,
                        debug=(global_sample_idx % 1000 == 0),  # 每 1000 个样本打印一次
                    )

                    # Step 2: 计算边界统计量
                    boundary_stats = calculate_boundary_stats(
                        E, blank_id=self.blank_id, rho=self.rho
                    )
                    all_boundary_stats[original_idx] = boundary_stats

                    # Step 3: Top-2 提取 - 每个时间步的最大和次大概率
                    top2_status = "available"  # 默认状态
                    top2_info = None

                    try:
                        # 获取 Top-2 索引和概率
                        top2_indices = np.argsort(E, axis=-1)[:, -2:][:, ::-1]  # [T, 2]
                        top2_probs = np.take_along_axis(
                            E, top2_indices, axis=-1
                        )  # [T, 2]

                        # 计算 Top-1 和 Top-2 的置信度统计
                        top1_conf_mean = float(np.mean(top2_probs[:, 0]))
                        top2_conf_mean = float(np.mean(top2_probs[:, 1]))
                        conf_gap_mean = float(
                            np.mean(top2_probs[:, 0] - top2_probs[:, 1])
                        )

                        # 转换索引为字符 (如果字符表可用)
                        top1_chars = None
                        top2_chars = None
                        if (
                            self.character_list is not None
                            and len(self.character_list) > 0
                        ):
                            top1_chars = []
                            top2_chars = []
                            for t in range(T):
                                idx1, idx2 = (
                                    int(top2_indices[t, 0]),
                                    int(top2_indices[t, 1]),
                                )
                                # 注意: blank_id 可能对应 '<blank>' 或特殊标记
                                char1 = (
                                    self.character_list[idx1]
                                    if 0 <= idx1 < len(self.character_list)
                                    else "<unk>"
                                )
                                char2 = (
                                    self.character_list[idx2]
                                    if 0 <= idx2 < len(self.character_list)
                                    else "<unk>"
                                )
                                top1_chars.append(char1)
                                top2_chars.append(char2)
                        else:
                            # 字符表不可用，但 Top-2 提取本身成功
                            top2_status = "available_no_chars"

                        # 组装 Top-2 信息
                        top2_info = {
                            "top2_status": top2_status,
                            "T": T,
                            "C": C,
                            "top1_conf_mean": top1_conf_mean,
                            "top2_conf_mean": top2_conf_mean,
                            "conf_gap_mean": conf_gap_mean,
                            # 完整序列 (可选，用于详细分析，仅当 T <= 100 时保存)
                            "top2_indices": top2_indices.tolist() if T <= 100 else None,
                            "top2_probs": top2_probs.tolist() if T <= 100 else None,
                            "top1_chars": top1_chars
                            if top2_status == "available" and T <= 100
                            else None,
                            "top2_chars": top2_chars
                            if top2_status == "available" and T <= 100
                            else None,
                        }
                    except Exception as e:
                        # Top-2 提取失败，标记为 missing
                        top2_status = "missing"
                        top2_info = {
                            "top2_status": top2_status,
                            "T": T,
                            "C": C,
                            "top1_conf_mean": 0.0,
                            "top2_conf_mean": 0.0,
                            "conf_gap_mean": 0.0,
                            "error": str(e) if logger else None,
                        }

                    all_top2_info[original_idx] = top2_info

                    # 计时结束
                    router_end = time.perf_counter()
                    lat_router_ms_list[original_idx] = (
                        router_end - router_start
                    ) * 1000.0
            # ========================================================================

            if self.benchmark:
                self.autolog.times.end(stamp=True)

        elapsed_time = time.time() - st

        # ========== SH-DA++ v4.0: 扩展返回格式，包含路由器特征信号 ==========
        return {
            # 基础识别结果
            "results": rec_res,  # List[Tuple[str, float]] - [(text, conf), ...]
            "elapsed_time": elapsed_time,
            # SH-DA++ v4.0 路由器特征信号
            "boundary_stats": all_boundary_stats,  # List[dict] - 边界统计量
            "top2_info": all_top2_info,  # List[dict] - Top-2 信息
            "lat_router_ms": lat_router_ms_list,  # List[float] - 路由器耗时 (ms)
            # 配置元信息
            "router_config": {
                "blank_id": self.blank_id,
                "rho": self.rho,
                "softmax_temperature": self.softmax_temperature,
            },
        }


def main(args):
    """测试入口函数"""
    image_file_list = get_image_file_list(args.image_dir)
    valid_image_file_list = []
    img_list = []

    # logger
    log_file = args.save_log_path
    if os.path.isdir(args.save_log_path) or (
        not os.path.exists(args.save_log_path) and args.save_log_path.endswith("/")
    ):
        log_file = os.path.join(log_file, "benchmark_recognition.log")
    logger = get_logger(log_file=log_file)

    # create text recognizer with logits export
    text_recognizer = TextRecognizerWithLogits(args)

    logger.info(
        "In PP-OCRv3, rec_image_shape parameter defaults to '3, 48, 320', "
        "if you are using recognition model with PP-OCRv2 or an older version, please set --rec_image_shape='3,32,320"
    )

    # warmup 2 times
    if args.warmup:
        img = np.random.uniform(0, 255, [48, 320, 3]).astype(np.uint8)
        for i in range(2):
            res = text_recognizer([img] * int(args.rec_batch_num))

    for image_file in image_file_list:
        img, flag, _ = check_and_read(image_file)
        if not flag:
            img = cv2.imread(image_file)
        if img is None:
            logger.info("error in loading image:{}".format(image_file))
            continue
        valid_image_file_list.append(image_file)
        img_list.append(img)
    try:
        # SH-DA++ v4.0: 获取增强返回值
        output = text_recognizer(img_list)
        rec_res = output["results"]
        boundary_stats_list = output.get("boundary_stats", [])
        top2_info_list = output.get("top2_info", [])
        lat_router_ms_list = output.get("lat_router_ms", [])
        router_config = output.get("router_config", {})
        elapsed = output["elapsed_time"]

        logger.info(
            f"[SH-DA++ v4.0] Router Config: blank_id={router_config.get('blank_id', 'N/A')}, rho={router_config.get('rho', 'N/A')}"
        )

    except Exception as E:
        logger.info(traceback.format_exc())
        logger.info(E)
        exit()

    # 打印识别结果和特征
    for ino in range(len(img_list)):
        logger.info(
            "Predicts of {}:{}".format(valid_image_file_list[ino], rec_res[ino])
        )

        # SH-DA++ v4.0: 打印边界统计量
        if ino < len(boundary_stats_list) and boundary_stats_list[ino] is not None:
            bs = boundary_stats_list[ino]
            if bs.get("valid", False):
                logger.info(
                    "  -> Boundary Stats: "
                    f"blank_mean_L={bs.get('blank_mean_L', 0):.4f}, "
                    f"blank_mean_R={bs.get('blank_mean_R', 0):.4f}, "
                    f"blank_peak_L={bs.get('blank_peak_L', 0):.4f}, "
                    f"blank_peak_R={bs.get('blank_peak_R', 0):.4f}, "
                    f"T={bs.get('T', 0)}"
                )
            else:
                logger.info("  -> Boundary Stats: invalid (T < 2)")

        # SH-DA++ v4.0: 打印 Top-2 信息
        if ino < len(top2_info_list) and top2_info_list[ino] is not None:
            top2 = top2_info_list[ino]
            status = top2.get("top2_status", "unknown")
            logger.info(
                f"  -> Top-2 Info: status={status}, "
                f"T={top2.get('T', 0)}, C={top2.get('C', 0)}, "
                f"top1_conf_mean={top2.get('top1_conf_mean', 0):.4f}, "
                f"conf_gap_mean={top2.get('conf_gap_mean', 0):.4f}"
            )
            if status == "missing":
                error_msg = top2.get("error", "Unknown error")
                logger.warning(f"    Top-2 extraction failed: {error_msg}")

        # SH-DA++ v4.0: 打印路由器耗时
        if ino < len(lat_router_ms_list):
            logger.info(f"  -> Router Latency: {lat_router_ms_list[ino]:.2f} ms")

    logger.info(f"Total elapsed time: {elapsed:.3f}s")

    # SH-DA++ v4.0: 统计信息汇总
    valid_boundary_stats = [
        bs for bs in boundary_stats_list if bs and bs.get("valid", False)
    ]
    if valid_boundary_stats:
        avg_blank_mean_L = np.mean(
            [bs.get("blank_mean_L", 0) for bs in valid_boundary_stats]
        )
        avg_blank_mean_R = np.mean(
            [bs.get("blank_mean_R", 0) for bs in valid_boundary_stats]
        )
        logger.info(
            f"[Summary] Valid boundary stats samples: {len(valid_boundary_stats)}/{len(img_list)}, "
            f"avg blank_mean_L={avg_blank_mean_L:.4f}, avg blank_mean_R={avg_blank_mean_R:.4f}"
        )

    if lat_router_ms_list:
        avg_router_lat = np.mean([lat for lat in lat_router_ms_list if lat > 0])
        max_router_lat = max(lat_router_ms_list) if lat_router_ms_list else 0
        logger.info(
            f"[Summary] Router latency: avg={avg_router_lat:.2f} ms, max={max_router_lat:.2f} ms"
        )

    if args.benchmark:
        text_recognizer.autolog.report()


if __name__ == "__main__":
    main(utility.parse_args())
