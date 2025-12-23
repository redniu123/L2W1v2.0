# copyright (c) 2020 PaddlePaddle Authors. All Rights Reserve.
# Modified for L2W1 Project - Minimal standalone version
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""
L2W1 精简版后处理模块

只包含 Agent A (PP-OCRv5 Rec) 推理所需的后处理:
- CTCLabelDecode: CTC 解码
- build_post_process: 构建后处理器
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy

__all__ = ["build_post_process"]

from .rec_postprocess import CTCLabelDecode


def build_post_process(config, global_config=None):
    """
    构建后处理器
    
    Args:
        config: 后处理配置字典
        global_config: 全局配置
        
    Returns:
        后处理器实例
    """
    # L2W1 精简版支持的后处理器
    support_dict = {
        "CTCLabelDecode": CTCLabelDecode,
    }
    
    config = copy.deepcopy(config)
    module_name = config.pop("name")
    
    if module_name == "None":
        return None
    
    if global_config is not None:
        config.update(global_config)
    
    if module_name not in support_dict:
        raise ValueError(
            f"L2W1 精简版仅支持 {list(support_dict.keys())}，"
            f"不支持 {module_name}"
        )
    
    module_class = support_dict[module_name](**config)
    return module_class

