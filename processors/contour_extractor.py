#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多层轮廓提取模块

从EPS栅格化图像中提取多层嵌套闭合轮廓，为每个轮廓生成蒙版。
支持轮廓腐蚀/膨胀以匹配PSD各层的边框间距。

核心思路：
1. 检测外轮廓 → 生成基础蒙版
2. 通过形态学腐蚀/膨胀生成多级轮廓蒙版
3. 每层PSD素材使用对应的轮廓蒙版进行裁剪
   → 各层自然跟随CAD轮廓的弧度和角度
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ContourLevel:
    """单层轮廓信息"""
    level: int                    # 层级索引（0=最外层）
    mask: Image.Image             # 该轮廓的蒙版 (L模式, 255=填充区)
    bounding_rect: Tuple[int, int, int, int]  # (x, y, w, h)
    area: int                     # 轮廓面积（像素）
    hierarchy: int = 0            # 层次深度


class ContourExtractor:
    """多层轮廓提取器"""

    def __init__(self, max_levels: int = 10, min_area_ratio: float = 0.01):
        self.max_levels = max_levels
        self.min_area_ratio = min_area_ratio

    def extract_contours(self, base_img: Image.Image,
                       num_levels: int = 5) -> List[ContourLevel]:
        """从EPS栅格化图像提取多层嵌套轮廓

        策略：
        1. 优先提取真实轮廓（通过fillPoly检测）
        2. 若真实轮廓不足，通过线条膨胀检测更多
        3. 若仍不足，基于最外层轮廓系统腐蚀生成虚拟轮廓
        4. 最终返回不超过num_levels层的轮廓

        Args:
            base_img: EPS栅格化图像
            num_levels: 期望的轮廓层数

        Returns:
            按面积从大到小排序的轮廓列表
        """
        try:
            import cv2
        except ImportError:
            logger.warning("OpenCV未安装，使用单层轮廓回退")
            single = self._extract_single_contour(base_img)
            if len(single) == 1 and num_levels > 1:
                return self._generate_virtual_contours(base_img, single[0], num_levels)
            return single

        # 1. 尝试真实轮廓提取
        levels = self._extract_with_cv2(base_img)
        logger.info(f"真实轮廓提取: {len(levels)} 层")

        # 2. 若真实轮廓太少，尝试线条检测
        if len(levels) < num_levels:
            logger.info(f"真实轮廓不足，尝试线条膨胀检测...")
            line_levels = self._extract_line_nested_contours(base_img)
            if len(line_levels) > len(levels):
                logger.info(f"线条检测: {len(line_levels)} 层")
                levels = line_levels

        # 3. 若仍不足，基于最外层生成虚拟轮廓
        if len(levels) < num_levels and len(levels) >= 1:
            needed = num_levels - len(levels)
            logger.info(f"需要{needed}层虚拟轮廓补充...")
            # 从最外层轮廓生成足够多的虚拟层
            base_contour = levels[0]
            real_areas = {c.area for c in levels}
            virtual = self._generate_virtual_contours(
                base_img, base_contour, num_levels
            )
            # 从虚拟轮廓中挑选真实轮廓没有的层
            for v in virtual:
                is_new = True
                for ea in real_areas:
                    if abs(ea - v.area) / max(ea, 1) < 0.08:
                        is_new = False
                        break
                if is_new and len(levels) < num_levels:
                    levels.append(v)
                    real_areas.add(v.area)

        # 4. 按面积排序并限制层数
        levels.sort(key=lambda c: c.area, reverse=True)
        for i, m in enumerate(levels):
            m.level = i

        if len(levels) > num_levels:
            levels = levels[:num_levels]

        logger.info(f"最终轮廓: {len(levels)} 层")
        return levels

    def _extract_with_cv2(self, base_img: Image.Image) -> List[ContourLevel]:
        """使用OpenCV提取真实轮廓

        使用RETR_CCOMP获取两级层次结构（外层+内层）。
        过滤掉面积过小的轮廓，并确保去重。
        """
        import cv2

        arr = np.array(base_img.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        # 二值化：背景变白(255)，CAD线框/图形变黑(0)
        _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)

        # 形态学闭运算：闭合线框的微小缺口
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=3)

        # 反相：线框变白(255)，背景变黑(0)
        inv = cv2.bitwise_not(closed)

        # 膨胀腐蚀确保轮廓完整
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        inv = cv2.dilate(inv, kernel_dilate, iterations=2)
        inv = cv2.erode(inv, kernel_dilate, iterations=1)

        # 使用RETR_CCOMP获取两级层次结构
        contours, hierarchy = cv2.findContours(inv, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return self._extract_single_contour(base_img)

        total_area = base_img.width * base_img.height
        levels: List[ContourLevel] = []
        seen_areas = set()

        # 按面积排序
        sorted_contours = sorted(
            enumerate(contours),
            key=lambda x: cv2.contourArea(x[1]),
            reverse=True
        )

        for rank, (orig_idx, contour) in enumerate(sorted_contours):
            area = int(cv2.contourArea(contour))
            if area < total_area * self.min_area_ratio:
                continue
            if len(levels) >= self.max_levels:
                break

            # 去重
            is_dup = any(abs(sa - area) / max(sa, 1) < 0.05 for sa in seen_areas)
            if is_dup:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # 创建蒙版
            mask_arr = np.zeros((base_img.height, base_img.width), dtype=np.uint8)
            cv2.fillPoly(mask_arr, [contour], 255)

            # 形态学开运算平滑边缘
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask_arr = cv2.morphologyEx(mask_arr, cv2.MORPH_OPEN, kernel_open, iterations=1)

            # 验证蒙版有效性
            mask_pixels = np.sum(mask_arr > 128)
            if mask_pixels < total_area * 0.005:
                continue

            mask_img = Image.fromarray(mask_arr, mode="L")
            seen_areas.add(area)

            levels.append(ContourLevel(
                level=len(levels),
                mask=mask_img,
                bounding_rect=(x, y, w, h),
                area=area,
                hierarchy=rank,
            ))

        if not levels:
            return self._extract_single_contour(base_img)

        logger.info(f"提取到 {len(levels)} 层真实轮廓 (CCOMP): "
                     f"面积 {levels[0].area}~{levels[-1].area} 像素")

        return levels

    def _extract_line_nested_contours(self, base_img: Image.Image) -> List[ContourLevel]:
        """通过线条膨胀检测嵌套轮廓（处理CAD线稿）

        思路：将CAD线稿逐步膨胀，每次膨胀后提取新的轮廓层。
        这样即使线稿是开口的或只有1像素宽，也能通过膨胀合并成闭合区域。
        """
        import cv2

        arr = np.array(base_img.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        # 检测所有深色像素（CAD线稿）
        _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # 反相：线稿变白
        inv = cv2.bitwise_not(binary)

        total_area = base_img.width * base_img.height
        found_levels: List[ContourLevel] = []

        # 逐步膨胀：检测不同尺度的嵌套结构
        max_iterations = 15
        for step in range(max_iterations):
            kernel_size = 3 + step * 3
            if kernel_size > 45:
                break

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            dilated = cv2.dilate(inv, kernel, iterations=1)
            closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                area = int(cv2.contourArea(contour))
                if area < total_area * 0.02:
                    continue

                x, y, w, h = cv2.boundingRect(contour)

                # 检查这个轮廓是否已被发现（位置相似）
                is_duplicate = False
                for existing in found_levels:
                    ex, ey, ew, eh = existing.bounding_rect
                    overlap = min(x + w, ex + ew) - max(x, ex)
                    overlap_ratio = overlap / min(w, ew) if min(w, ew) > 0 else 0
                    if overlap_ratio > 0.95 and abs(w - ew) / max(w, ew) < 0.1:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    mask_arr = np.zeros((base_img.height, base_img.width), dtype=np.uint8)
                    cv2.fillPoly(mask_arr, [contour], 255)
                    found_levels.append(ContourLevel(
                        level=len(found_levels),
                        mask=Image.fromarray(mask_arr, mode="L"),
                        bounding_rect=(x, y, w, h),
                        area=area,
                        hierarchy=step,
                    ))

        # 按面积排序
        found_levels.sort(key=lambda c: c.area, reverse=True)
        for i, level in enumerate(found_levels):
            level.level = i

        if found_levels:
            logger.info(f"线条检测提取到 {len(found_levels)} 层嵌套轮廓")

        return found_levels

    def _generate_virtual_contours(self, base_img: Image.Image,
                                     base_contour: ContourLevel,
                                     num_levels: int) -> List[ContourLevel]:
        """从单层基础轮廓生成多层虚拟轮廓

        通过系统腐蚀生成一系列内嵌轮廓层，
        每层都严格跟随外轮廓的形状。

        Args:
            base_contour: 基础轮廓（通常是最外层）
            num_levels: 生成的层数
        """
        import numpy as np

        levels = [base_contour]
        original_mask = base_contour.mask
        original_area = base_contour.area
        target_min_area = original_area * 0.05

        max_levels = min(num_levels, self.max_levels)
        min_dim = min(base_img.width, base_img.height)
        erosion_step = max(3, int(min_dim * 0.015))

        for i in range(1, max_levels):
            erosion_total = erosion_step * i

            # 每次从原始蒙版腐蚀（避免累积误差）
            eroded_mask = self.erode_mask(original_mask, erosion_total)
            eroded_arr = np.array(eroded_mask)
            area = int(np.sum(eroded_arr > 128))

            if area < target_min_area:
                break

            ys, xs = np.where(eroded_arr > 128)
            if len(xs) == 0:
                break

            x, y = int(xs.min()), int(ys.min())
            w, h = int(xs.max() - x + 1), int(ys.max() - y + 1)

            levels.append(ContourLevel(
                level=len(levels),
                mask=eroded_mask,
                bounding_rect=(x, y, w, h),
                area=area,
                hierarchy=i,
            ))

        logger.info(f"虚拟生成 {len(levels)} 层轮廓 (腐蚀步长={erosion_step}px)")
        return levels

    def _extract_single_contour(self, base_img: Image.Image) -> List[ContourLevel]:
        """回退方案：只提取单层外轮廓"""
        import numpy as np

        mask = self._create_single_mask(base_img)
        mask_arr = np.array(mask)
        ys, xs = np.where(mask_arr > 0)
        if len(xs) == 0:
            x, y, w, h = 0, 0, base_img.width, base_img.height
            area = base_img.width * base_img.height
        else:
            x, y = int(xs.min()), int(ys.min())
            w, h = int(xs.max() - x + 1), int(ys.max() - y + 1)
            area = int(np.sum(mask_arr > 0))

        return [ContourLevel(
            level=0,
            mask=mask,
            bounding_rect=(x, y, w, h),
            area=area,
            hierarchy=0,
        )]

    def _create_single_mask(self, base_img: Image.Image) -> Image.Image:
        """创建单层外轮廓蒙版"""
        try:
            import cv2
            import numpy as np

            arr = np.array(base_img.convert("RGB"))
            gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)

            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=3)
            inv = cv2.bitwise_not(closed)

            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            inv = cv2.dilate(inv, kernel_dilate, iterations=2)
            inv = cv2.erode(inv, kernel_dilate, iterations=1)

            contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest = max(contours, key=cv2.contourArea)
                mask_arr = np.zeros((base_img.height, base_img.width), dtype=np.uint8)
                cv2.fillPoly(mask_arr, [largest], 255)
                kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                mask_arr = cv2.morphologyEx(mask_arr, cv2.MORPH_OPEN, kernel_open, iterations=1)
                return Image.fromarray(mask_arr, mode="L")
        except ImportError:
            pass

        return Image.new("L", base_img.size, 255)

    def erode_mask(self, mask: Image.Image, erosion_px: int) -> Image.Image:
        """腐蚀蒙版"""
        if erosion_px <= 0:
            return mask.copy()

        try:
            import numpy as np
            from scipy import ndimage

            mask_arr = np.array(mask)
            binary = mask_arr > 128
            eroded_binary = ndimage.binary_erosion(binary, iterations=erosion_px)
            eroded = (eroded_binary.astype(np.uint8)) * 255
            return Image.fromarray(eroded, mode="L")
        except ImportError:
            try:
                import cv2
                import numpy as np

                mask_arr = np.array(mask)
                # Add 1px black border to ensure erosion works even on full-white masks
                bordered = np.pad(mask_arr, 1, mode="constant", constant_values=0)

                # Use iterative erosion for large values to avoid huge kernels
                if erosion_px > 20:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                    # Erode in chunks of 20 to avoid excessive iterations
                    remaining = erosion_px
                    result = bordered
                    while remaining > 0:
                        chunk = min(remaining, 20)
                        result = cv2.erode(result, kernel, iterations=chunk)
                        remaining -= chunk
                    eroded = result[1:-1, 1:-1]
                else:
                    kernel_size = max(1, erosion_px * 2 + 1)
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                    eroded = cv2.erode(bordered, kernel, iterations=1)
                    eroded = eroded[1:-1, 1:-1]

                return Image.fromarray(eroded, mode="L")
            except ImportError:
                return self._pil_erode(mask, erosion_px)

    def dilate_mask(self, mask: Image.Image, dilation_px: int) -> Image.Image:
        """膨胀蒙版"""
        if dilation_px <= 0:
            return mask.copy()

        try:
            import numpy as np
            from scipy import ndimage

            mask_arr = np.array(mask)
            binary = mask_arr > 128
            dilated_binary = ndimage.binary_dilation(binary, iterations=dilation_px)
            dilated = (dilated_binary.astype(np.uint8)) * 255
            return Image.fromarray(dilated, mode="L")
        except ImportError:
            try:
                import cv2
                import numpy as np

                mask_arr = np.array(mask)
                kernel_size = max(1, dilation_px * 2 + 1)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
                dilated = cv2.dilate(mask_arr, kernel, iterations=1)
                return Image.fromarray(dilated, mode="L")
            except ImportError:
                return self._pil_dilate(mask, dilation_px)

    def _pil_erode(self, mask: Image.Image, erosion_px: int) -> Image.Image:
        """PIL回退腐蚀"""
        from PIL import ImageFilter
        img = mask.copy()
        for _ in range(erosion_px):
            img = img.filter(ImageFilter.MinFilter(3))
        return img

    def _pil_dilate(self, mask: Image.Image, dilation_px: int) -> Image.Image:
        """PIL回退膨胀"""
        from PIL import ImageFilter
        img = mask.copy()
        for _ in range(dilation_px):
            img = img.filter(ImageFilter.MaxFilter(3))
        return img

    def create_border_ring_mask(self, outer_mask: Image.Image,
                                 inner_mask: Image.Image) -> Image.Image:
        """创建环形边框蒙版（两个蒙版的差集）

        用于生成"边框带"区域：外层蒙版 - 内层蒙版 = 边框环
        """
        import numpy as np

        outer_arr = np.array(outer_mask)
        inner_arr = np.array(inner_mask)

        ring_arr = np.where(
            (outer_arr > 128) & (inner_arr <= 128),
            255, 0
        ).astype(np.uint8)

        return Image.fromarray(ring_arr, mode="L")

    def compute_erosion_for_layer(self, outer_rect: Tuple[int, int, int, int],
                                   layer_rect: Tuple[int, int, int, int]) -> int:
        """计算PSD某层相对于外轮廓的腐蚀距离

        Args:
            outer_rect: 外轮廓边界 (x, y, w, h)
            layer_rect: PSD图层边界 (x, y, w, h)
        Returns:
            腐蚀像素数（取四边最小值）
        """
        ox, oy, ow, oh = outer_rect
        lx, ly, lw, lh = layer_rect

        # 各方向的距离（PSD层相对于外轮廓的内缩量）
        left_dist = lx - ox
        top_dist = ly - oy
        right_dist = (ox + ow) - (lx + lw)
        bottom_dist = (oy + oh) - (ly + lh)

        # 取最小值作为腐蚀量（保证不会超出边界）
        distances = [d for d in [left_dist, top_dist, right_dist, bottom_dist] if d > 0]
        if not distances:
            return 0

        # 取中位数避免极端值
        sorted_dists = sorted(distances)
        return int(sorted_dists[len(sorted_dists) // 2])
