#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像处理算法包

包含智能对齐、色彩调整、图像合成等纯算法模块。
不依赖任何引擎，可被任何上层模块调用。

依赖方向：processors 依赖 models，不依赖 engines/services/gui
"""

from .aligner import SmartAligner
from .color_adjuster import ColorAdjuster
from .compositor import ImageCompositor

__all__ = ["SmartAligner", "ColorAdjuster", "ImageCompositor"]
