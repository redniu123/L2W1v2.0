# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
# Modified for L2W1 Project - Minimal standalone version
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""
L2W1 精简版日志模块
"""

import os
import sys
import logging
import functools

logger_initialized = {}


@functools.lru_cache()
def get_logger(name="ppocr", log_file=None, log_level=logging.DEBUG, log_ranks="0"):
    """
    初始化并获取日志器
    
    Args:
        name: 日志器名称
        log_file: 日志文件路径
        log_level: 日志级别
        log_ranks: 记录日志的 GPU ID
        
    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    formatter = logging.Formatter(
        "[%(asctime)s] %(name)s %(levelname)s: %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
    )

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    if log_file is not None:
        try:
            log_file_folder = os.path.split(log_file)[0]
            os.makedirs(log_file_folder, exist_ok=True)
            file_handler = logging.FileHandler(log_file, "a")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception:
            pass

    if isinstance(log_ranks, str):
        log_ranks = [int(i) for i in log_ranks.split(",")]
    elif isinstance(log_ranks, int):
        log_ranks = [log_ranks]

    # 简化版：始终设置日志级别
    logger.setLevel(log_level)
    logger_initialized[name] = True
    logger.propagate = False
    return logger

