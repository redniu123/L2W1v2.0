# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
# Modified for L2W1 Project - Minimal standalone version for Agent A (Text Recognizer)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""
L2W1 精简版推理工具

只包含 Agent A (PP-OCRv5 Rec) 推理所需的函数:
- init_args: 参数解析
- create_predictor: 创建 Paddle Inference 预测器
- get_output_tensors: 获取输出张量
- load_config: 加载 YAML 配置
"""

import argparse
import os
import sys
import cv2
import numpy as np
import paddle
from paddle import inference
import yaml

# 导入本地 logging 模块
try:
    from ppocr.utils.logging import get_logger
except ImportError:
    import logging
    def get_logger(name="ppocr"):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                datefmt="%Y/%m/%d %H:%M:%S"
            ))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


def str2bool(v):
    return v.lower() in ("true", "yes", "t", "y", "1")


def str2int_tuple(v):
    return tuple([int(i.strip()) for i in v.split(",")])


def init_args():
    """初始化参数解析器 (精简版 - 仅保留 Rec 相关参数)"""
    parser = argparse.ArgumentParser()
    
    # params for prediction engine
    parser.add_argument("--use_gpu", type=str2bool, default=True)
    parser.add_argument("--use_xpu", type=str2bool, default=False)
    parser.add_argument("--use_npu", type=str2bool, default=False)
    parser.add_argument("--use_mlu", type=str2bool, default=False)
    parser.add_argument("--use_metax_gpu", type=str2bool, default=False)
    parser.add_argument("--use_gcu", type=str2bool, default=False)
    parser.add_argument("--ir_optim", type=str2bool, default=True)
    parser.add_argument("--use_tensorrt", type=str2bool, default=False)
    parser.add_argument("--min_subgraph_size", type=int, default=15)
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--gpu_mem", type=int, default=500)
    parser.add_argument("--gpu_id", type=int, default=0)

    # params for text recognizer
    parser.add_argument("--rec_algorithm", type=str, default="SVTR_LCNet")
    parser.add_argument("--rec_model_dir", type=str)
    parser.add_argument("--rec_image_inverse", type=str2bool, default=True)
    parser.add_argument("--rec_image_shape", type=str, default="3, 48, 320")
    parser.add_argument("--rec_batch_num", type=int, default=6)
    parser.add_argument("--max_text_length", type=int, default=25)
    parser.add_argument("--rec_char_dict_path", type=str, default="./ppocr/utils/ppocr_keys_v1.txt")
    parser.add_argument("--use_space_char", type=str2bool, default=True)
    parser.add_argument("--drop_score", type=float, default=0.5)
    
    # params for inference
    parser.add_argument("--enable_mkldnn", type=str2bool, default=None)
    parser.add_argument("--cpu_threads", type=int, default=10)
    parser.add_argument("--warmup", type=str2bool, default=False)
    parser.add_argument("--benchmark", type=str2bool, default=False)
    parser.add_argument("--save_log_path", type=str, default="./log_output/")
    parser.add_argument("--show_log", type=str2bool, default=True)
    parser.add_argument("--use_onnx", type=str2bool, default=False)
    parser.add_argument("--max_batch_size", type=int, default=10)
    parser.add_argument("--return_word_box", type=str2bool, default=False)

    return parser


def parse_args():
    parser = init_args()
    return parser.parse_args()


def load_config(file_path):
    """加载 YAML 配置文件"""
    _, ext = os.path.splitext(file_path)
    if ext not in [".yml", ".yaml"]:
        raise ValueError(f"only support yaml files for now, got {file_path}")
    with open(file_path, "rb") as file:
        config = yaml.load(file, Loader=yaml.SafeLoader)
    return config


def get_infer_gpuid():
    """获取推理使用的 GPU ID"""
    logger = get_logger()
    if not paddle.device.is_compiled_with_rocm:
        gpu_id_str = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    else:
        gpu_id_str = os.environ.get("HIP_VISIBLE_DEVICES", "0")
    
    gpu_ids = gpu_id_str.split(",")
    if len(gpu_ids) > 0 and gpu_ids[0]:
        return int(gpu_ids[0])
    return 0


def create_predictor(args, mode, logger):
    """
    创建 Paddle Inference 预测器
    
    Args:
        args: 参数对象
        mode: 模式 ("rec" for recognizer)
        logger: 日志器
        
    Returns:
        (predictor, input_tensor, output_tensors, config)
    """
    if mode == "rec":
        model_dir = args.rec_model_dir
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    if model_dir is None:
        logger.info("not find {} model file path {}".format(mode, model_dir))
        sys.exit(0)
    
    # ONNX 模式
    if args.use_onnx:
        import onnxruntime as ort
        
        model_file_path = model_dir
        if not os.path.exists(model_file_path):
            raise ValueError("not find model file path {}".format(model_file_path))
        
        if args.use_gpu:
            sess = ort.InferenceSession(
                model_file_path,
                providers=[
                    (
                        "CUDAExecutionProvider",
                        {"device_id": args.gpu_id, "cudnn_conv_algo_search": "DEFAULT"},
                    )
                ],
            )
        else:
            sess = ort.InferenceSession(
                model_file_path,
                providers=["CPUExecutionProvider"],
            )
        inputs = sess.get_inputs()
        return (
            sess,
            inputs[0] if len(inputs) == 1 else [vo.name for vo in inputs],
            None,
            None,
        )
    
    # Paddle Inference 模式
    file_names = ["model", "inference"]
    for file_name in file_names:
        params_file_path = f"{model_dir}/{file_name}.pdiparams"
        if os.path.exists(params_file_path):
            break
    
    if not os.path.exists(params_file_path):
        raise ValueError(f"not find {file_name}.pdiparams in {model_dir}")
    
    if not (
        os.path.exists(f"{model_dir}/{file_name}.pdmodel")
        or os.path.exists(f"{model_dir}/{file_name}.json")
    ):
        raise ValueError(
            f"neither {file_name}.json nor {file_name}.pdmodel was found in {model_dir}."
        )
    
    if os.path.exists(f"{model_dir}/{file_name}.json"):
        model_file_path = f"{model_dir}/{file_name}.json"
    else:
        model_file_path = f"{model_dir}/{file_name}.pdmodel"
    
    config = inference.Config(model_file_path, params_file_path)
    
    # 精度配置
    if hasattr(args, "precision"):
        if args.precision == "fp16" and args.use_tensorrt:
            precision = inference.PrecisionType.Half
        elif args.precision == "int8":
            precision = inference.PrecisionType.Int8
        else:
            precision = inference.PrecisionType.Float32
    else:
        precision = inference.PrecisionType.Float32
    
    # GPU 配置
    if args.use_gpu:
        gpu_id = get_infer_gpuid()
        if gpu_id is None:
            logger.warning(
                "GPU is not found in current device by nvidia-smi. Please check your device."
            )
        config.enable_use_gpu(args.gpu_mem, args.gpu_id)
        
        if args.use_tensorrt:
            config.enable_tensorrt_engine(
                workspace_size=1 << 30,
                precision_mode=precision,
                max_batch_size=args.max_batch_size,
                min_subgraph_size=args.min_subgraph_size,
                use_calib_mode=False,
            )
            
            # collect shape
            trt_shape_f = os.path.join(model_dir, f"{mode}_trt_dynamic_shape.txt")
            if not os.path.exists(trt_shape_f):
                config.collect_shape_range_info(trt_shape_f)
                logger.info(f"collect dynamic shape info into : {trt_shape_f}")
            try:
                config.enable_tuned_tensorrt_dynamic_shape(trt_shape_f, True)
            except Exception as E:
                logger.info(E)
                logger.info("Please keep your paddlepaddle-gpu >= 2.3.0!")
    
    elif args.use_npu:
        config.enable_custom_device("npu")
    elif args.use_mlu:
        config.enable_custom_device("mlu")
    elif args.use_xpu:
        config.enable_xpu(10 * 1024 * 1024)
    else:
        config.disable_gpu()
        if args.enable_mkldnn is not None and args.enable_mkldnn:
            config.set_mkldnn_cache_capacity(10)
            config.enable_mkldnn()
            if args.precision == "fp16":
                config.enable_mkldnn_bfloat16()
        
        if hasattr(args, "cpu_threads"):
            config.set_cpu_math_library_num_threads(args.cpu_threads)
        else:
            config.set_cpu_math_library_num_threads(10)
        
        if hasattr(config, "enable_new_ir"):
            config.enable_new_ir()
        if hasattr(config, "enable_new_executor"):
            config.enable_new_executor()
    
    # 内存优化
    config.enable_memory_optim()
    config.disable_glog_info()
    config.delete_pass("conv_transpose_eltwiseadd_bn_fuse_pass")
    config.delete_pass("matmul_transpose_reshape_fuse_pass")
    
    if mode == "rec" and args.rec_algorithm == "SRN":
        config.delete_pass("gpu_cpu_map_matmul_v2_to_matmul_pass")
    
    config.switch_use_feed_fetch_ops(False)
    config.switch_ir_optim(True)
    
    # 创建预测器
    predictor = inference.create_predictor(config)
    input_names = predictor.get_input_names()
    
    for name in input_names:
        input_tensor = predictor.get_input_handle(name)
    
    output_tensors = get_output_tensors(args, mode, predictor)
    return predictor, input_tensor, output_tensors, config


def get_output_tensors(args, mode, predictor):
    """获取输出张量"""
    output_names = predictor.get_output_names()
    output_tensors = []
    
    if mode == "rec" and args.rec_algorithm in ["CRNN", "SVTR_LCNet", "SVTR_HGNet"]:
        output_name = "softmax_0.tmp_0"
        if output_name in output_names:
            return [predictor.get_output_handle(output_name)]
        else:
            for output_name in output_names:
                output_tensor = predictor.get_output_handle(output_name)
                output_tensors.append(output_tensor)
    else:
        for output_name in output_names:
            output_tensor = predictor.get_output_handle(output_name)
            output_tensors.append(output_tensor)
    
    return output_tensors

