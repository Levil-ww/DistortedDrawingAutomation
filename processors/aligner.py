#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能对齐模块 - 轮廓对齐

支持两种对齐模式：
1. contour（轮廓对齐）：图案跟随EPS轮廓的实际形状，通过蒙版裁剪实现
2. bounding（矩形对齐）：基于边界矩形的传统对齐（回退方案）

轮廓对齐是核心：确保图案的每一层都符合CAD轮廓的弧度和角度。
"""

import logging
from typing import Tuple, Optional, List

from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)


class SmartAligner:
    """智能对齐器"""

    def __init__(self, margin_percent: float = 2.0):
        self.margin_percent = margin_percent

    def align(self, base_img: Image.Image, pattern_img: Image.Image,
              fill_mode: str = "cover") -> Tuple[float, int, int]:
        """计算对齐参数（缩放 + 偏移）

        Args:
            fill_mode: "cover" = max缩放覆盖整个区域
                      "contain" = min缩放适配区域
        """
        try:
            return self._detect_border_align(base_img, pattern_img, fill_mode)
        except ImportError:
            logger.warning("OpenCV未安装，使用居中填充")
            return self._center_fill(base_img, pattern_img, fill_mode)
        except Exception as e:
            logger.warning(f"智能对齐失败: {e}，使用居中填充")
            return self._center_fill(base_img, pattern_img, fill_mode)

    def align_to_contour(self, base_img: Image.Image, pattern_img: Image.Image,
                          contour_mask: Image.Image,
                          erosion_px: int = 0) -> Tuple[float, int, int]:
        """轮廓对齐：将PSD图层对齐到EPS轮廓

        核心逻辑：
        1. 从轮廓蒙版提取边界矩形
        2. 考虑腐蚀量调整有效区域
        3. 缩放PSD图层使其适配轮廓区域
        4. 居中放置

        Args:
            base_img: EPS栅格化图像
            pattern_img: PSD图层图像
            contour_mask: 轮廓蒙版
            erosion_px: 腐蚀量（像素）
        Returns:
            (scale, offset_x, offset_y)
        """
        import numpy as np

        mask_arr = np.array(contour_mask)

        # 找到蒙版的有效区域
        ys, xs = np.where(mask_arr > 128)
        if len(xs) == 0:
            logger.warning("轮廓蒙版为空，回退到全图对齐")
            return self._center_fill(base_img, pattern_img, "cover")

        # 轮廓的边界矩形
        cx, cy = int(xs.min()), int(ys.min())
        cw, ch = int(xs.max() - cx + 1), int(ys.max() - cy + 1)

        # 应用腐蚀量（向内收缩）
        effective_x = cx + erosion_px
        effective_y = cy + erosion_px
        effective_w = max(1, cw - 2 * erosion_px)
        effective_h = max(1, ch - 2 * erosion_px)

        pw, ph = pattern_img.size

        # 计算缩放：使PSD图层适配有效区域
        scale_x = effective_w / pw
        scale_y = effective_h / ph
        scale = max(scale_x, scale_y) * 1.02  # 稍微放大确保覆盖

        new_pw = int(pw * scale)
        new_ph = int(ph * scale)

        # 居中放置在有效区域
        offset_x = effective_x + (effective_w - new_pw) // 2
        offset_y = effective_y + (effective_h - new_ph) // 2

        logger.info(f"轮廓对齐: 轮廓({cx},{cy},{cw},{ch}), "
                     f"腐蚀={erosion_px}px, 有效区({effective_x},{effective_y},{effective_w},{effective_h}), "
                     f"缩放={scale:.4f}, 偏移=({offset_x},{offset_y})")

        return scale, offset_x, offset_y

    def align_layers_to_contours(self, base_img: Image.Image,
                                  layers_data: List[Tuple[Image.Image, int]],
                                  contour_masks: List[Image.Image]) -> List[Tuple[float, int, int]]:
        """多层轮廓对齐

        为每个PSD图层分配对应的轮廓蒙版并计算对齐参数

        Args:
            base_img: EPS栅格化图像
            layers_data: [(pattern_img, erosion_px), ...] 按面积降序
            contour_masks: [mask_level0, mask_level1, ...] 按面积降序
        Returns:
            每个图层的对齐参数列表 [(scale, ox, oy), ...]
        """
        results = []

        for i, (pattern_img, erosion_px) in enumerate(layers_data):
            if i < len(contour_masks):
                # 使用对应的轮廓蒙版
                scale, ox, oy = self.align_to_contour(
                    base_img, pattern_img, contour_masks[i], erosion_px
                )
            else:
                # 轮廓不够，使用最后一个轮廓
                last_mask = contour_masks[-1] if contour_masks else None
                if last_mask:
                    scale, ox, oy = self.align_to_contour(
                        base_img, pattern_img, last_mask, erosion_px
                    )
                else:
                    scale, ox, oy = self._center_fill(base_img, pattern_img, "cover")

            results.append((scale, ox, oy))

        return results

    def _detect_border_align(self, base_img: Image.Image, pattern_img: Image.Image,
                              fill_mode: str = "cover") -> Tuple[float, int, int]:
        import cv2

        cv_img = cv2.cvtColor(np.array(base_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, kernel, iterations=2)
        edges = cv2.erode(edges, kernel, iterations=1)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return self._center_fill(base_img, pattern_img, fill_mode)

        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)

        margin = self.margin_percent / 100.0
        mw, mh = int(w * margin), int(h * margin)
        x = max(0, x - mw)
        y = max(0, y - mh)
        w = min(base_img.width - x, w + mw * 2)
        h = min(base_img.height - y, h + mh * 2)

        pw, ph = pattern_img.size

        if fill_mode == "cover":
            scale_x = base_img.width / pw
            scale_y = base_img.height / ph
            scale = max(scale_x, scale_y) * 1.02
            new_pw, new_ph = int(pw * scale), int(ph * scale)
            offset_x = (base_img.width - new_pw) // 2
            offset_y = (base_img.height - new_ph) // 2
        else:
            scale_x = w / pw
            scale_y = h / ph
            scale = min(scale_x, scale_y) * 0.98
            new_pw, new_ph = int(pw * scale), int(ph * scale)
            offset_x = x + (w - new_pw) // 2
            offset_y = y + (h - new_ph) // 2

        logger.info(f"智能对齐[{fill_mode}]: 边框({x},{y},{w},{h}), 缩放={scale:.3f}, 偏移=({offset_x},{offset_y})")
        return scale, offset_x, offset_y

    def _center_fill(self, base_img: Image.Image, pattern_img: Image.Image,
                     fill_mode: str = "cover") -> Tuple[float, int, int]:
        pw, ph = pattern_img.size
        if fill_mode == "cover":
            scale = max(base_img.width / pw, base_img.height / ph) * 1.02
        else:
            scale = min(base_img.width / pw, base_img.height / ph) * 0.98
        new_pw, new_ph = int(pw * scale), int(ph * scale)
        offset_x = (base_img.width - new_pw) // 2
        offset_y = (base_img.height - new_ph) // 2
        return scale, offset_x, offset_y
