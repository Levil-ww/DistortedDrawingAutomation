#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能对齐模块
使用OpenCV检测边框区域，计算最佳缩放和居中偏移
完全独立于引擎，只操作PIL Image
"""

import logging
from typing import Tuple

from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class SmartAligner:
    """智能对齐器"""

    def __init__(self, margin_percent: float = 2.0):
        self.margin_percent = margin_percent

    def align(self, base_img: Image.Image, pattern_img: Image.Image) -> Tuple[float, int, int]:
        """
        检测边框并计算最佳缩放和偏移
        使用 min 缩放使图案适配（而非覆盖）边框区域
        Returns: (scale, offset_x, offset_y)
        """
        try:
            return self._detect_border_align(base_img, pattern_img)
        except ImportError:
            logger.warning("OpenCV未安装，使用居中填充")
            return self._center_fill(base_img, pattern_img)
        except Exception as e:
            logger.warning(f"智能对齐失败: {e}，使用居中填充")
            return self._center_fill(base_img, pattern_img)

    def _detect_border_align(self, base_img: Image.Image, pattern_img: Image.Image) -> Tuple[float, int, int]:
        import cv2

        cv_img = cv2.cvtColor(np.array(base_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._center_fill(base_img, pattern_img)

        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)

        margin = self.margin_percent / 100.0
        mw, mh = int(w * margin), int(h * margin)
        x = max(0, x - mw)
        y = max(0, y - mh)
        w = min(base_img.width - x, w + mw * 2)
        h = min(base_img.height - y, h + mh * 2)

        pw, ph = pattern_img.size
        scale_x = w / pw
        scale_y = h / ph
        scale = min(scale_x, scale_y) * 0.98

        new_pw, new_ph = int(pw * scale), int(ph * scale)
        offset_x = x + (w - new_pw) // 2
        offset_y = y + (h - new_ph) // 2

        logger.info(f"智能对齐: 边框({x},{y},{w},{h}), 缩放={scale:.3f}, 偏移=({offset_x},{offset_y})")
        return scale, offset_x, offset_y

    def _center_fill(self, base_img: Image.Image, pattern_img: Image.Image) -> Tuple[float, int, int]:
        pw, ph = pattern_img.size
        scale = min(base_img.width / pw, base_img.height / ph) * 0.98
        new_pw, new_ph = int(pw * scale), int(ph * scale)
        offset_x = (base_img.width - new_pw) // 2
        offset_y = (base_img.height - new_ph) // 2
        return scale, offset_x, offset_y
