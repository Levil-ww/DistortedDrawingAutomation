#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像合成模块
负责将花纹图层合成到基础图像上
纯Pillow实现，无引擎依赖
"""

import logging
from PIL import Image

logger = logging.getLogger(__name__)


class ImageCompositor:
    """图像合成器"""

    def compose(
        self,
        base: Image.Image,
        pattern: Image.Image,
        offset_x: int,
        offset_y: int,
        mask: Image.Image = None,
    ) -> Image.Image:
        """
        将花纹合成到基础图像上
        Args:
            base: 基础图像 (RGB)
            pattern: 花纹图像 (RGBA)
            offset_x, offset_y: 花纹放置位置
            mask: 可选的蒙版图像 (L)
        """
        result = base.copy().convert("RGBA")
        pattern_rgba = pattern.convert("RGBA")

        if mask is not None:
            # 使用蒙版裁剪
            result.paste(pattern_rgba, (offset_x, offset_y), pattern_rgba)
            mask_rgba = Image.merge("RGBA", [mask, mask, mask, mask])
            result = Image.alpha_composite(result, mask_rgba)
        else:
            result.paste(pattern_rgba, (offset_x, offset_y), pattern_rgba)

        return result.convert("RGB")
