#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PSD图层分类与映射模块

分析PSD图层的几何特征和位置，将其分类并映射到EPS轮廓层级。

分类依据：
1. 图层边界框位置（相对于PSD整体边界）
2. 图层像素密度与分布
3. 图层名称（如有）
4. 与EPS轮廓层级的面积匹配

核心思路：
- PSD图层按面积从大到小排序
- EPS轮廓按面积从大到小排序
- 一一对应：最大的PSD图层 → 最外层EPS轮廓
- 每个图层的边框间距通过腐蚀量保持与PSD一致
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from models import LayerInfo

logger = logging.getLogger(__name__)


@dataclass
class LayerClassification:
    """PSD图层分类结果"""
    layer: LayerInfo              # 原始PSD图层
    layer_index: int              # 在PSD中的索引
    bounding_rect: Tuple[int, int, int, int]  # (x, y, w, h) 在PSD坐标系中
    center: Tuple[float, float]   # 中心点
    area: int                     # 面积
    pixel_density: float          # 像素密度（非透明像素比例）
    graphic_type: str             # 分类: "outer_rect", "middle_rect", "inner_rect", "floral", "dots", "unknown"
    contour_level: int = 0       # 映射到的EPS轮廓层级
    erosion_px: int = 0          # 需要的腐蚀量（用于该层的蒙版）


class PSDLayerClassifier:
    """PSD图层分类器"""

    def __init__(self):
        self._type_names = {
            "outer_rect": "外层矩形",
            "middle_rect": "中层矩形",
            "inner_rect": "内层矩形",
            "floral": "花卉纹样",
            "dots": "点状装饰",
            "unknown": "未知",
        }

    def classify_layers(self, layers: List[LayerInfo],
                        psd_composite: Optional[Image.Image] = None) -> List[LayerClassification]:
        """分类PSD图层

        Args:
            layers: PSD图层列表
            psd_composite: PSD合成图（用于参考整体边界）
        Returns:
            分类结果列表，按面积从大到小排序
        """
        if not layers:
            return []

        results = []
        psd_bounds = self._compute_psd_bounds(layers, psd_composite)

        for idx, layer in enumerate(layers):
            cls = self._classify_single_layer(layer, idx, psd_bounds)
            results.append(cls)

        # 按面积从大到小排序
        results.sort(key=lambda r: r.area, reverse=True)

        # 分配轮廓层级：面积最大的映射到最外层(level=0)
        for i, cls in enumerate(results):
            cls.contour_level = min(i, 99)  # 最多99层

        # 计算腐蚀量（基于图层在PSD中的位置）
        self._compute_erosion_values(results, psd_bounds)

        # 日志输出
        for cls in results:
            type_name = self._type_names.get(cls.graphic_type, "未知")
            logger.info(f"图层 [{cls.layer_index}] '{cls.layer.name}': "
                         f"{type_name}, 面积={cls.area}, "
                         f"密度={cls.pixel_density:.2f}, "
                         f"轮廓层={cls.contour_level}, "
                         f"腐蚀={cls.erosion_px}px")

        return results

    def _compute_psd_bounds(self, layers: List[LayerInfo],
                             composite: Optional[Image.Image] = None) -> Tuple[int, int, int, int]:
        """计算PSD整体边界（所有图层的并集）"""
        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")

        for layer in layers:
            img = layer.image
            if img.mode != "RGBA":
                img = img.convert("RGBA")

            # 找到非透明像素的边界
            arr = np.array(img)
            alpha = arr[:, :, 3]
            non_transparent = np.where(alpha > 10)

            if len(non_transparent[0]) > 0:
                ly, lx = non_transparent[0], non_transparent[1]
                min_x = min(min_x, lx.min())
                min_y = min(min_y, ly.min())
                max_x = max(max_x, lx.max())
                max_y = max(max_y, ly.max())

        if min_x == float("inf"):
            if composite:
                return (0, 0, composite.width, composite.height)
            return (0, 0, 100, 100)

        w = max_x - min_x + 1
        h = max_y - min_y + 1
        return (int(min_x), int(min_y), int(w), int(h))

    def _classify_single_layer(self, layer: LayerInfo, index: int,
                                psd_bounds: Tuple[int, int, int, int]) -> LayerClassification:
        """分类单个PSD图层"""
        img = layer.image
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        arr = np.array(img)
        alpha = arr[:, :, 3]
        non_transparent = np.where(alpha > 10)

        if len(non_transparent[0]) == 0:
            return LayerClassification(
                layer=layer,
                layer_index=index,
                bounding_rect=(0, 0, 0, 0),
                center=(0, 0),
                area=0,
                pixel_density=0,
                graphic_type="unknown",
            )

        ys, xs = non_transparent
        x, y = int(xs.min()), int(ys.min())
        w, h = int(xs.max() - x + 1), int(ys.max() - y + 1)
        area = int(w * h)
        center = ((x + w / 2), (y + h / 2))

        # 计算像素密度
        total_pixels = w * h
        pixel_count = len(xs)
        density = pixel_count / total_pixels if total_pixels > 0 else 0

        # 计算该层相对于PSD边界的位置
        px, py, pw, ph = psd_bounds
        rel_x = x - px
        rel_y = y - py
        rel_w = w
        rel_h = h
        rel_area_ratio = (w * h) / (pw * ph) if pw * ph > 0 else 0

        # 几何特征分析
        graphic_type = self._determine_graphic_type(
            arr, alpha, x, y, w, h, density, rel_area_ratio,
            layer.name if hasattr(layer, 'name') else ""
        )

        return LayerClassification(
            layer=layer,
            layer_index=index,
            bounding_rect=(x, y, w, h),
            center=center,
            area=area,
            pixel_density=density,
            graphic_type=graphic_type,
        )

    def _determine_graphic_type(self, arr: np.ndarray, alpha: np.ndarray,
                                 x: int, y: int, w: int, h: int,
                                 density: float, area_ratio: float,
                                 name: str) -> str:
        """根据几何特征和像素分布判定图层类型

        判定规则：
        1. 点状装饰: 像素密度极低 (<0.05), 且面积小
        2. 花卉纹样: 像素密度中等 (0.1-0.5), 具有复杂纹理
        3. 矩形类: 像素密度高 (>0.5), 边界清晰
           - 外层矩形: 面积占比最大 (>0.6)
           - 中层矩形: 面积占比中等 (0.3-0.6)
           - 内层矩形: 面积占比较小 (<0.3)
        """
        # 名称匹配
        name_lower = name.lower() if name else ""
        if any(k in name_lower for k in ["dot", "点", "装饰"]):
            return "dots"
        if any(k in name_lower for k in ["floral", "花", "蔓", "pattern", "纹样"]):
            return "floral"
        if any(k in name_lower for k in ["outer", "外", "border", "边框"]):
            return "outer_rect"
        if any(k in name_lower for k in ["middle", "mid", "中"]):
            return "middle_rect"
        if any(k in name_lower for k in ["inner", "内"]):
            return "inner_rect"

        # 基于几何特征的判定
        if density < 0.05 and area_ratio < 0.3:
            return "dots"

        if 0.05 <= density <= 0.5:
            # 检查是否为复杂纹理（花卉）
            if self._has_complex_texture(arr, alpha, x, y, w, h):
                return "floral"

        # 矩形类判定
        if density > 0.5:
            if area_ratio > 0.6:
                return "outer_rect"
            elif area_ratio > 0.3:
                return "middle_rect"
            else:
                return "inner_rect"

        # 中等密度且非复杂纹理 → 可能是纹样
        if density > 0.1:
            return "floral"

        return "unknown"

    def _has_complex_texture(self, arr: np.ndarray, alpha: np.ndarray,
                              x: int, y: int, w: int, h: int) -> bool:
        """检测是否具有复杂纹理（高频变化）"""
        try:
            # 提取非透明区域的像素
            region = arr[y:y+h, x:x+w]
            region_alpha = alpha[y:y+h, x:x+w]
            mask = region_alpha > 10

            if np.sum(mask) == 0:
                return False

            # 计算局部方差（衡量纹理复杂度）
            gray_region = np.mean(region[mask], axis=0) if np.sum(mask) > 0 else 0
            if isinstance(gray_region, np.ndarray) and len(gray_region) > 10:
                variance = np.var(gray_region)
                return variance > 500  # 高方差 = 复杂纹理
        except Exception:
            pass

        return False

    def _compute_erosion_values(self, results: List[LayerClassification],
                                 psd_bounds: Tuple[int, int, int, int]):
        """计算每个图层的腐蚀量

        腐蚀量 = 图层相对于PSD外边界的内缩距离
        这确保渲染时各层的边框间距与PSD素材一致
        """
        px, py, pw, ph = psd_bounds

        for cls in results:
            lx, ly, lw, lh = cls.bounding_rect

            # 计算各方向内缩距离
            left = lx - px
            top = ly - py
            right = (px + pw) - (lx + lw)
            bottom = (py + ph) - (ly + lh)

            # 取最小正距离作为腐蚀量
            distances = [d for d in [left, top, right, bottom] if d > 0]
            if distances:
                # 使用中位数避免极端值
                sorted_dists = sorted(distances)
                cls.erosion_px = int(sorted_dists[len(sorted_dists) // 2])
            else:
                cls.erosion_px = 0
