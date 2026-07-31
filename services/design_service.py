#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
业务服务层 - 编排整个处理流程

核心流程（多层轮廓跟随渲染）：
  EPS栅格化 → 提取多层轮廓 → PSD图层分类 → 逐层轮廓对齐 → 逐层渲染 → 合成 → 边框叠加 → 色彩调整 → 保存

每层PSD素材独立处理，分别对齐到对应的EPS轮廓，
确保每个图层的边框线距离与PSD素材保持一致，
且每层都符合CAD轮廓的弧度和角度。
"""

import logging
import re
from pathlib import Path
from typing import Callable, Optional, Tuple, List

import numpy as np
from PIL import Image

from models import ProcessConfig, LayerInfo
from engines.base import ImageEngine
from engines.pillow_engine import get_eps_bbox
from processors.aligner import SmartAligner
from processors.color_adjuster import ColorAdjuster
from processors.compositor import ImageCompositor
from processors.contour_extractor import ContourExtractor, ContourLevel
from processors.layer_classifier import PSDLayerClassifier, LayerClassification

logger = logging.getLogger(__name__)


def _parse_psd_size_from_filename(psd_path: Path) -> Optional[Tuple[float, float]]:
    """从PSD文件名解析素材尺寸 (width_cm, height_cm)"""
    if psd_path is None:
        return None

    name = psd_path.stem
    has_portrait = bool(re.search(r'竖版|纵向|portrait', name, re.IGNORECASE))

    m = re.search(r'(?i)(\d+)[\-_xX×](\d+)', name)
    if m:
        num1, num2 = float(m.group(1)), float(m.group(2))
        if num1 > 0 and num2 > 0:
            if has_portrait:
                w, h = min(num1, num2), max(num1, num2)
                orientation = "竖版"
            else:
                w, h = max(num1, num2), min(num1, num2)
                orientation = "横版"
            logger.info(f"从文件名解析PSD尺寸: {name} -> {w:.0f}x{h:.0f}cm ({orientation})")
            return (w, h)
    return None


class DesignService:
    """设计自动化业务服务"""

    def __init__(
        self,
        engine: ImageEngine,
        aligner: Optional[SmartAligner] = None,
        color_adjuster: Optional[ColorAdjuster] = None,
        compositor: Optional[ImageCompositor] = None,
        contour_extractor: Optional[ContourExtractor] = None,
        layer_classifier: Optional[PSDLayerClassifier] = None,
    ):
        self.engine = engine
        self.aligner = aligner or SmartAligner()
        self.color_adjuster = color_adjuster or ColorAdjuster()
        self.compositor = compositor or ImageCompositor()
        self.contour_extractor = contour_extractor or ContourExtractor()
        self.layer_classifier = layer_classifier or PSDLayerClassifier()

    # ---------- 核心流程 ----------

    def process(self, config: ProcessConfig,
                progress_callback: Optional[Callable[[str], None]] = None) -> Path:
        """执行完整的设计处理流程

        新流程：多层轮廓跟随渲染
        1. EPS栅格化
        2. 提取多层嵌套轮廓
        3. 加载PSD各图层并分类
        4. 逐层渲染（每层对齐对应轮廓，保持边框间距）
        5. Z序合成
        6. CAD边框叠加
        7. 色彩调整 → 保存
        """
        def _notify(msg: str):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        eps_path = Path(config.eps_file)
        psd_path = Path(config.psd_file)
        out_path = Path(config.output_file)

        # 0. 画布尺寸解析
        width_cm, height_cm = self._resolve_canvas_size(eps_path, config)
        _notify(f"画布尺寸: {width_cm:.1f}x{height_cm:.1f}cm "
                f"({'横版' if width_cm > height_cm else '竖版'})")

        # 1. 栅格化 EPS
        _notify("正在打开EPS模板...")
        base_img = self.engine.open_eps(
            eps_path, dpi=config.dpi,
            width_cm=width_cm,
            height_cm=height_cm,
        )
        logger.info(f"EPS栅格化: {base_img.width}x{base_img.height}px")

        # 2. 提取多层轮廓
        _notify("正在提取CAD轮廓...")
        contour_levels = self.contour_extractor.extract_contours(base_img, num_levels=5)
        _notify(f"检测到 {len(contour_levels)} 层轮廓")

        # 3. 加载PSD各图层
        _notify("正在加载PSD图层...")
        psd_layers = self.engine.load_psd_layers(psd_path)
        if not psd_layers:
            raise RuntimeError("PSD中未找到可用图层")
        _notify(f"加载了 {len(psd_layers)} 个PSD图层")

        # 4. 获取PSD合成图（用于分类和回退）
        psd_composite = self._prepare_psd_composite(psd_path, psd_layers)

        # 5. 分类PSD图层
        _notify("正在分类PSD图层...")
        classifications = self.layer_classifier.classify_layers(psd_layers, psd_composite)

        # 6. 多层渲染（优先使用PSD合成图整体渲染，确保所有效果正确）
        if psd_composite:
            _notify("使用PSD合成图渲染（包含所有图层效果）...")
            result = self._render_with_psd_composite(
                base_img, contour_levels, psd_composite, config
            )
        else:
            _notify("正在逐层渲染...")
            result = self._render_multi_layer(
                base_img, contour_levels, classifications, psd_composite, config
            )

        # 8. 色彩调整
        _notify("正在应用色彩调整...")
        result = self.color_adjuster.adjust(
            result,
            brightness=config.brightness,
            contrast=config.contrast,
            saturation=config.saturation,
            hue_shift=config.hue_shift,
            warmth=config.warmth,
        )

        # 9. 保存
        _notify("正在保存JPG...")
        self.engine.save_jpg(result, out_path, quality=config.jpg_quality)
        _notify(f"处理完成: {out_path}")

        return out_path

    def generate_preview(self, config: ProcessConfig, max_width: int = 400) -> Image.Image:
        """生成预览图（复刻 process 流程，使用低DPI + 缩放显示）"""
        preview_dpi = min(config.dpi, 72)
        logger.info(f"生成预览: EPS @ {preview_dpi}dpi, 目标宽度 {max_width}px")

        eps_path = Path(config.eps_file)
        width_cm, height_cm = self._resolve_canvas_size(eps_path, config)

        # 1. 栅格化 EPS
        base = self.engine.open_eps(
            eps_path, dpi=preview_dpi,
            width_cm=width_cm,
            height_cm=height_cm,
        )

        # 2. 提取轮廓
        contour_levels = self.contour_extractor.extract_contours(base, num_levels=5)

        # 3. 加载PSD
        psd_path = Path(config.psd_file)
        psd_layers = self.engine.load_psd_layers(psd_path)
        if not psd_layers:
            raise RuntimeError("无可用图层")

        psd_composite = self._prepare_psd_composite(psd_path, psd_layers)
        classifications = self.layer_classifier.classify_layers(psd_layers, psd_composite)

        # 4. 多层渲染（优先使用PSD合成图整体渲染）
        if psd_composite:
            result = self._render_with_psd_composite(
                base, contour_levels, psd_composite, config
            )
        else:
            result = self._render_multi_layer(
                base, contour_levels, classifications, psd_composite, config
            )

        # 5. 色彩调整
        result = self.color_adjuster.adjust(
            result,
            brightness=config.brightness,
            contrast=config.contrast,
            saturation=config.saturation,
            hue_shift=config.hue_shift,
            warmth=config.warmth,
        )

        # 6. 缩放到显示尺寸
        if result.width > max_width:
            ratio = max_width / result.width
            display_size = (max_width, int(result.height * ratio))
            result = result.resize(display_size, Image.LANCZOS)

        return result

    # ---------- 多层渲染核心 ----------

    def _render_multi_layer(self, base_img: Image.Image,
                             contour_levels: List[ContourLevel],
                             classifications: List[LayerClassification],
                             psd_composite: Image.Image,
                             config: ProcessConfig) -> Image.Image:
        """多层轮廓跟随渲染（正确版v2）

        核心思路：
        1. 使用最外层EPS轮廓作为基础蒙版
        2. 按PSD图层内缩距离从小到大排序
        3. 为每层计算腐蚀值（确保层间有足够间距）
        4. 每层蒙版 = 基础蒙版腐蚀(腐蚀值)
        5. 描边 = 相邻两层蒙版的差集（统一深棕色）
        6. Z序合成：最外层先渲染，内层后渲染覆盖
        """
        w, h = base_img.size
        canvas = Image.new("RGBA", (w, h), (255, 255, 255, 255))

        # 使用最外层轮廓作为基础（所有层都基于此腐蚀）
        if not contour_levels:
            base_mask = Image.new("L", (w, h), 255)
            base_rect = (0, 0, w, h)
        else:
            base_mask = contour_levels[0].mask
            base_rect = contour_levels[0].bounding_rect

        bx, by, bw, bh = base_rect
        min_dim = min(bw, bh)

        # 1. 计算PSD整体边界
        psd_bounds = self._get_psd_bounds(classifications)
        px, py, pw, ph = psd_bounds

        # 2. 计算PSD→EPS缩放比
        scale_x = bw / pw if pw > 0 else 1.0
        scale_y = bh / ph if ph > 0 else 1.0

        # 3. 为每层计算PSD内缩距离（PSD像素→EPS像素）
        layer_data = []  # (orig_idx, erosion_eps, cls)
        for orig_idx, cls in enumerate(classifications):
            if cls.area <= 0:
                continue

            lx, ly, lw, lh = cls.bounding_rect
            left_inset = lx - px
            right_inset = (px + pw) - (lx + lw)
            top_inset = ly - py
            bottom_inset = (py + ph) - (ly + lh)

            insets = [d for d in [left_inset, top_inset, right_inset, bottom_inset] if d > 0]
            if insets and scale_x > 0:
                median_inset = sorted(insets)[len(insets) // 2]
                erosion_eps = int(median_inset * scale_x)
            else:
                erosion_eps = 0

            layer_data.append((orig_idx, erosion_eps, cls))

        # 4. 按腐蚀值从小到大排序（外层先）
        layer_data.sort(key=lambda x: x[1])

        # 5. 确保腐蚀值单调递增且层间距足够大
        # 每层间距至少为 min_dim * 0.04（约4%的最小边）
        min_layer_gap = max(5, int(min_dim * 0.04))
        max_erosion = int(min_dim * 0.4)  # 最大腐蚀40%的最小维度

        final_erosions = []
        for orig_idx, erosion, cls in layer_data:
            if final_erosions:
                prev_erosion = final_erosions[-1][1]
                if erosion <= prev_erosion:
                    erosion = prev_erosion + min_layer_gap
                elif erosion - prev_erosion < min_layer_gap:
                    erosion = prev_erosion + min_layer_gap
            erosion = min(erosion, max_erosion)
            final_erosions.append((orig_idx, erosion))

        logger.info(f"各层腐蚀值(EPS像素): "
                     f"{[(classifications[i].layer.name, e) for i, e in final_erosions]}")

        # 6. 为每层创建蒙版
        layer_masks = {}
        for orig_idx, erosion in final_erosions:
            if erosion > 0:
                mask = self.contour_extractor.erode_mask(base_mask, erosion)
            else:
                mask = base_mask.copy()
            layer_masks[orig_idx] = mask

        # 7. 计算描边环（统一深棕色描边）
        stroke_rings = {}
        sorted_orig_indices = [idx for idx, _ in final_erosions]

        # 最外层描边：base_mask - 最外层腐蚀后蒙版
        if sorted_orig_indices:
            first_idx = sorted_orig_indices[0]
            first_mask = layer_masks[first_idx]
            if final_erosions[0][1] > 0:
                outer_stroke = self.contour_extractor.create_border_ring_mask(
                    base_mask, first_mask
                )
                stroke_rings[first_idx] = outer_stroke

        # 层间描边：外层蒙版 - 内层蒙版
        for i in range(1, len(sorted_orig_indices)):
            outer_idx = sorted_orig_indices[i - 1]
            inner_idx = sorted_orig_indices[i]
            outer_mask = layer_masks[outer_idx]
            inner_mask = layer_masks[inner_idx]
            stroke_ring = self.contour_extractor.create_border_ring_mask(
                outer_mask, inner_mask
            )
            stroke_rings[inner_idx] = stroke_ring

        # 8. 第一遍：Z序合成（从最外层到最内层，所有图层填充）
        for orig_idx, erosion in final_erosions:
            cls = classifications[orig_idx]
            layer_img = cls.layer.image
            if layer_img.mode != "RGBA":
                layer_img = layer_img.convert("RGBA")

            layer_mask = layer_masks[orig_idx]

            # 蒙版边界
            mask_arr = np.array(layer_mask)
            mask_ys, mask_xs = np.where(mask_arr > 128)
            if len(mask_xs) > 0:
                mx, my = int(mask_xs.min()), int(mask_ys.min())
                mw, mh = int(mask_xs.max() - mx + 1), int(mask_ys.max() - my + 1)
            else:
                mx, my, mw, mh = 0, 0, w, h

            # PSD图层实际内容边界（alpha > 1）
            arr = np.array(layer_img)
            alpha = arr[:, :, 3]
            non_transparent = np.where(alpha > 1)
            if len(non_transparent[0]) == 0:
                logger.warning(f"图层 '{cls.layer.name}' 无可见内容，跳过")
                continue

            ys, xs = non_transparent
            content_x, content_y = int(xs.min()), int(ys.min())
            content_w, content_h = int(xs.max() - content_x + 1), int(ys.max() - content_y + 1)

            # COVER缩放
            if content_w > 0 and content_h > 0 and mw > 0 and mh > 0:
                scale = max(mw / content_w, mh / content_h)
            else:
                scale = 1.0

            new_w = max(1, int(layer_img.width * scale))
            new_h = max(1, int(layer_img.height * scale))

            # 中心对齐
            offset_x = int((mx + mw / 2) - (content_x + content_w / 2) * scale)
            offset_y = int((my + mh / 2) - (content_y + content_h / 2) * scale)

            # 渲染（仅填充，不应用描边）
            scaled_layer = layer_img.resize((new_w, new_h), Image.LANCZOS)
            layer_canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
            layer_canvas.paste(scaled_layer, (offset_x, offset_y), scaled_layer)

            # 蒙版裁剪
            layer_arr = np.array(layer_canvas)
            mask_bin = (mask_arr > 128)
            for c in range(3):
                layer_arr[:, :, c] = np.where(mask_bin, layer_arr[:, :, c], 0)
            layer_arr[:, :, 3] = np.where(mask_bin, layer_arr[:, :, 3], 0)

            rendered = Image.fromarray(layer_arr, mode="RGBA")
            canvas = Image.alpha_composite(canvas, rendered)

            logger.info(f"渲染图层 '{cls.layer.name}': "
                         f"类型={cls.graphic_type}, 腐蚀={erosion}px, "
                         f"缩放={scale:.4f}, 偏移=({offset_x},{offset_y})")

        # 第二遍：应用所有描边（从最外层到最内层，确保描边在图层之上）
        for orig_idx in sorted_orig_indices:
            if orig_idx in stroke_rings:
                canvas = self._apply_stroke_ring(canvas, stroke_rings[orig_idx])

        # 9. 叠加CAD边框线
        canvas = self._overlay_cad_border(canvas, base_img, base_mask)

        return canvas.convert("RGB")

    def _render_with_psd_composite(self, base_img: Image.Image,
                                    contour_levels: List[ContourLevel],
                                    psd_composite: Image.Image,
                                    config: ProcessConfig) -> Image.Image:
        """使用PSD合成图直接渲染

        当PSD包含矢量智能对象等无法单独加载的图层时，
        使用PSD合成图作为整体图案源，确保所有视觉效果正确呈现。

        核心思路：
        1. 获取PSD合成图（已包含所有图层效果、矢量智能对象、混合模式）
        2. 使用最外层EPS轮廓作为基础蒙版
        3. Cover模式缩放合成图适配EPS形状
        4. 应用EPS蒙版裁剪
        5. 叠加CAD边框线
        """
        w, h = base_img.size

        # 使用最外层轮廓作为基础蒙版
        if not contour_levels:
            base_mask = Image.new("L", (w, h), 255)
            base_rect = (0, 0, w, h)
        else:
            base_mask = contour_levels[0].mask
            base_rect = contour_levels[0].bounding_rect

        bx, by, bw, bh = base_rect

        # PSD合成图转为RGBA
        composite = psd_composite.convert("RGBA")
        cw, ch = composite.size

        # 计算Cover模式缩放（确保PSD填满EPS形状）
        scale_x = bw / cw if cw > 0 else 1.0
        scale_y = bh / ch if ch > 0 else 1.0
        scale = max(scale_x, scale_y)

        new_w = max(1, int(cw * scale))
        new_h = max(1, int(ch * scale))

        # 缩放PSD合成图
        scaled_composite = composite.resize((new_w, new_h), Image.LANCZOS)

        # 中心对齐到EPS形状
        offset_x = int(bx + (bw - new_w) / 2)
        offset_y = int(by + (bh - new_h) / 2)

        # 合成到画布
        canvas = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        canvas.paste(scaled_composite, (offset_x, offset_y), scaled_composite)

        # 应用EPS蒙版裁剪（蒙版外保持白色背景，避免convert RGB时变黑）
        mask_arr = np.array(base_mask)
        mask_bin = (mask_arr > 128)

        canvas_arr = np.array(canvas)
        for c in range(3):
            canvas_arr[:, :, c] = np.where(mask_bin, canvas_arr[:, :, c], 255)
        canvas_arr[:, :, 3] = 255

        canvas = Image.fromarray(canvas_arr, mode="RGBA")

        logger.info(f"PSD合成图渲染: 缩放={scale:.4f}, 偏移=({offset_x},{offset_y}), "
                     f"画布={w}x{h}")

        # 叠加CAD边框线
        canvas = self._overlay_cad_border(canvas, base_img, base_mask)

        return canvas.convert("RGB")

    def _apply_stroke_ring(self, canvas: Image.Image,
                            stroke_ring_mask: Image.Image) -> Image.Image:
        """应用描边环效果

        在描边环区域填充深色描边。使用统一的深棕色描边色。
        """
        canvas_arr = np.array(canvas)
        mask_arr = np.array(stroke_ring_mask)
        mask_bin = (mask_arr > 128)

        if not np.any(mask_bin):
            return canvas

        # 统一深棕色描边（与效果图一致）
        stroke_color = (90, 65, 30)

        stroke_arr = np.array(canvas.copy())
        for c in range(3):
            stroke_arr[:, :, c] = np.where(
                mask_bin,
                stroke_color[c],
                canvas_arr[:, :, c]
            )
        stroke_arr[:, :, 3] = np.where(mask_bin, 255, canvas_arr[:, :, 3])

        return Image.fromarray(stroke_arr, mode="RGBA")

    def _get_psd_bounds(self, classifications: List[LayerClassification]) -> Tuple[int, int, int, int]:
        """计算PSD所有图层的整体边界"""
        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")

        for cls in classifications:
            lx, ly, lw, lh = cls.bounding_rect
            if lw > 0 and lh > 0:
                min_x = min(min_x, lx)
                min_y = min(min_y, ly)
                max_x = max(max_x, lx + lw)
                max_y = max(max_y, ly + lh)

        if min_x == float("inf"):
            return (0, 0, 100, 100)

        return (int(min_x), int(min_y), int(max_x - min_x), int(max_y - min_y))

    def _render_single_layer(self, base_img: Image.Image,
                              psd_composite: Image.Image,
                              config: ProcessConfig) -> Image.Image:
        """单层渲染（回退方案）"""
        # 计算对齐
        scale, ox, oy = self._compute_alignment(base_img, psd_composite, config)

        # 缩放图案
        new_size = (int(psd_composite.width * scale), int(psd_composite.height * scale))
        pattern_img = psd_composite.resize(new_size, Image.LANCZOS)

        # 创建EPS蒙版
        eps_mask = self._create_eps_mask(base_img)

        # 合成
        result = self._compose_with_eps_mask(base_img, pattern_img, ox, oy, eps_mask)
        return result

    # ---------- 辅助方法 ----------

    def _prepare_psd_composite(self, psd_path: Path, layers) -> Image.Image:
        """获取PSD合成图

        优先使用引擎已导出的完整合成图（__psd_composite__），
        其次兼容旧版回退层，最后尝试psd-tools或单图层选择。
        """
        # 1. 优先使用引擎已导出的PSD完整合成图
        for layer in layers:
            if layer.name == "__psd_composite__":
                logger.info(f"使用PSD完整合成图: {layer.image.width}x{layer.image.height}")
                return layer.image.convert("RGBA")

        # 2. 兼容旧版回退层
        for layer in layers:
            if layer.name == "__psd_composite_fallback__":
                logger.info(f"使用已加载的PSD合成图: {layer.image.width}x{layer.image.height}")
                return layer.image.convert("RGBA")

        # 3. 尝试用psd-tools渲染完整合成图
        composite = self._render_psd_composite(psd_path)
        if composite is not None:
            return composite

        # 4. 回退：选择最佳花纹图层
        best = self._select_pattern_layer(layers)
        if best is not None:
            return best

        logger.warning("未找到合适的花纹图层，使用第一个图层")
        return layers[0].image.convert("RGBA")

    def _render_psd_composite(self, psd_path: Path) -> Optional[Image.Image]:
        """用 psd-tools 渲染 PSD 完整合成图"""
        try:
            from psd_tools import PSDImage
            psd = PSDImage.open(str(psd_path))
            img = psd.composite()
            if img is not None:
                if img.mode != "RGBA":
                    img = img.convert("RGBA")
                logger.info(f"使用 psd-tools 完整合成: {img.width}x{img.height}")
                return img
        except Exception as e:
            logger.debug(f"psd-tools 合成失败: {e}")
        return None

    def _select_pattern_layer(self, layers) -> Optional[Image.Image]:
        """回退方案：选择最佳花纹图层"""
        best = None
        best_score = -1
        for layer in layers:
            img = layer.image
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            small = img.resize((100, 100), Image.LANCZOS)
            pixels = list(small.getdata())
            total = len(pixels)
            non_transparent = sum(1 for p in pixels if p[3] > 10)
            non_white = sum(
                1 for p in pixels
                if p[3] > 10 and (p[0] < 250 or p[1] < 250 or p[2] < 250)
            )
            if non_transparent == 0 or non_white == 0:
                score = 0
            else:
                graphic_density = non_white / total
                score = graphic_density * 1000
            if score > best_score:
                best_score = score
                best = img
        return best if best_score > 0 else None

    def _overlay_cad_border(self, canvas: Image.Image, base_img: Image.Image,
                              contour_mask: Image.Image) -> Image.Image:
        """叠加CAD边框线

        从base中提取深色像素（CAD线稿），叠加在渲染结果上层
        边框线自然跟随EPS外轮廓的弧度
        """
        base_arr = np.array(base_img.convert("RGBA"))
        canvas_arr = np.array(canvas)
        mask_arr = np.array(contour_mask)
        mask_bin = (mask_arr > 128)

        # 检测base中的深色像素（CAD线稿）
        is_dark = np.any(base_arr[:, :, :3] < 120, axis=2)
        is_border = is_dark & mask_bin

        if np.any(is_border):
            for c in range(3):
                canvas_arr[:, :, c] = np.where(
                    is_border, base_arr[:, :, c], canvas_arr[:, :, c]
                )
            # 边框区域保持不透明
            canvas_arr[:, :, 3] = np.where(is_border, 255, canvas_arr[:, :, 3])
            logger.info(f"叠加CAD边框线: {np.sum(is_border)} 像素")

        return Image.fromarray(canvas_arr, mode="RGBA")

    def _compute_alignment(self, base_img: Image.Image, pattern_img: Image.Image,
                           config: ProcessConfig):
        """计算对齐参数（回退方案）"""
        if config.smart_align and config.auto_scale:
            return self.aligner.align(base_img, pattern_img, fill_mode="cover")
        else:
            scale = config.pattern_scale if config.pattern_scale != 1.0 else (
                max(base_img.width / pattern_img.width,
                    base_img.height / pattern_img.height) * 1.02
            )
            ox = config.pattern_offset_x or (base_img.width - int(pattern_img.width * scale)) // 2
            oy = config.pattern_offset_y or (base_img.height - int(pattern_img.height * scale)) // 2
            return scale, ox, oy

    def _create_eps_mask(self, base_img: Image.Image) -> Image.Image:
        """创建EPS外轮廓蒙版（回退方案）"""
        return self.contour_extractor._create_single_mask(base_img)

    def _compose_with_eps_mask(self, base: Image.Image, pattern: Image.Image,
                                offset_x: int, offset_y: int,
                                eps_mask: Image.Image) -> Image.Image:
        """使用EPS蒙版合成图案（回退方案）"""
        w, h = base.size
        canvas = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        pattern_rgba = pattern.convert("RGBA")
        canvas.paste(pattern_rgba, (offset_x, offset_y), pattern_rgba)

        canvas_arr = np.array(canvas)
        mask_arr = np.array(eps_mask)
        mask_bin = (mask_arr > 128).astype(np.uint8)

        for c in range(3):
            canvas_arr[:, :, c] = np.where(mask_bin, canvas_arr[:, :, c], 255)
        canvas_arr[:, :, 3] = 255

        # CAD边框叠加
        base_arr = np.array(base.convert("RGBA"))
        is_dark = np.any(base_arr[:, :, :3] < 120, axis=2)
        is_border = is_dark & (mask_bin > 0)
        if np.any(is_border):
            for c in range(3):
                canvas_arr[:, :, c] = np.where(
                    is_border, base_arr[:, :, c], canvas_arr[:, :, c]
                )

        return Image.fromarray(canvas_arr, mode="RGBA").convert("RGB")

    # ---------- 画布尺寸解析 ----------

    def _resolve_canvas_size(self, eps_path: Path, config: ProcessConfig) -> tuple:
        """解析画布尺寸：以EPS文件尺寸为准"""
        bbox = get_eps_bbox(eps_path)

        if bbox is not None:
            eps_w, eps_h = bbox
            bbox_is_landscape = eps_w > eps_h

            psd_path = Path(config.psd_file) if config.psd_file else None
            psd_size = _parse_psd_size_from_filename(psd_path) if psd_path else None
            if psd_size is not None:
                psd_w, psd_h = psd_size
                psd_is_landscape = psd_w > psd_h
                if psd_is_landscape != bbox_is_landscape:
                    logger.info(f"PSD方向与EPS不匹配，交换画布方向")
                    return round(eps_h, 1), round(eps_w, 1)

            logger.info(f"使用EPS尺寸作为画布: {eps_w:.1f}x{eps_h:.1f}cm")
            return round(eps_w, 1), round(eps_h, 1)

        configured_w = config.canvas_width_cm
        configured_h = config.canvas_height_cm
        logger.warning(f"无法解析EPS BoundingBox，使用配置尺寸: {configured_w}x{configured_h}cm")
        return configured_w, configured_h
