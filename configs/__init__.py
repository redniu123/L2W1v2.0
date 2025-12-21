# L2W1 配置模块

"""
配置管理模块

提供统一的配置加载和管理功能
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# 配置目录
CONFIG_DIR = Path(__file__).parent


def load_config(config_name: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_name: 配置文件名 (不含后缀) 或完整路径
        
    Returns:
        dict: 配置字典
    """
    # 尝试多种路径
    possible_paths = [
        CONFIG_DIR / f"{config_name}.yaml",
        CONFIG_DIR / f"{config_name}.yml",
        Path(config_name),
    ]
    
    for path in possible_paths:
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
    
    raise FileNotFoundError(f"Config file not found: {config_name}")


def get_router_config() -> Dict[str, Any]:
    """获取路由器配置"""
    return load_config("router_config")


def get_visual_thresholds() -> Dict[str, float]:
    """获取视觉熵阈值"""
    config = get_router_config()
    return config.get("visual", {})


def get_semantic_thresholds() -> Dict[str, Any]:
    """获取语义 PPL 阈值"""
    config = get_router_config()
    return config.get("semantic", {})


__all__ = [
    'load_config',
    'get_router_config',
    'get_visual_thresholds',
    'get_semantic_thresholds',
    'CONFIG_DIR',
]

