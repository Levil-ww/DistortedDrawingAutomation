#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
色彩调整模块
纯Pillow实现，与引擎无关
"""

from PIL import Image, ImageEnhance
import numpy as np


class ColorAdjuster:
    """色彩调整器"""

    def adjust(
        self,
        image: Image.Image,
        brightness: int = 0,
        contrast: int = 0,
        saturation: int = 0,
        hue_shift: int = 0,
        warmth: int = 0,
    ) -> Image.Image:
        """应用全套色彩调整"""
        img = image.copy()

        if brightness != 0:
            img = ImageEnhance.Brightness(img).enhance(1 + brightness / 100)

        if contrast != 0:
            img = ImageEnhance.Contrast(img).enhance(1 + contrast / 100)

        if saturation != 0:
            img = ImageEnhance.Color(img).enhance(1 + saturation / 100)

        if hue_shift != 0:
            img = self._apply_hue_shift(img, hue_shift)

        if warmth != 0:
            img = self._apply_warmth(img, warmth)

        return img

    def _apply_hue_shift(self, img: Image.Image, shift: int) -> Image.Image:
        """色相偏移"""
        arr = np.array(img.convert("HSV"))
        arr[:, :, 0] = (arr[:, :, 0].astype(int) + shift) % 180
        return Image.fromarray(arr, "HSV").convert("RGB")

    def _apply_warmth(self, img: Image.Image, warmth: int) -> Image.Image:
        """色温调整"""
        arr = np.array(img).astype(np.float32)
        factor = warmth / 100.0
        arr[:, :, 0] = np.clip(arr[:, :, 0] + factor * 30, 0, 255)
        arr[:, :, 2] = np.clip(arr[:, :, 2] - factor * 20, 0, 255)
        return Image.fromarray(arr.astype(np.uint8))
