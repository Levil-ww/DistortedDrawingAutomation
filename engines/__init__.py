#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像处理引擎包

提供统一的抽象接口和多种实现。

依赖方向：engines 依赖 models，不依赖 processors/services/gui
"""

from .base import ImageEngine
from .photoshop_engine import PhotoshopEngine
from .pillow_engine import PillowEngine

__all__ = ["ImageEngine", "PhotoshopEngine", "PillowEngine", "create_engine"]


def create_engine(preferred: str = "auto") -> ImageEngine:
    """
    工厂函数：根据环境自动创建最佳引擎
    Args:
        preferred: "auto" | "photoshop" | "pillow"
    """
    if preferred == "photoshop":
        return PhotoshopEngine()
    if preferred == "pillow":
        return PillowEngine()

    # auto模式：优先Photoshop，降级到Pillow
    try:
        return PhotoshopEngine()
    except RuntimeError:
        return PillowEngine()
