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

    def generate_preview(self, config: ProcessConfig, max_width_cm: float = 15.0) -> Image.Image:
        """生成预览图

        使用中等DPI栅格化EPS，高分辨率加载PSD，合成后缩放到显示尺寸。
        max_width_cm: 预览在GUI中显示的最大宽度（厘米），用于确定显示缩放
        """
        # 预览用中等DPI（平衡质量和速度）
        preview_dpi = 150
        logger.info(f"=== 开始生成预览: EPS @ {preview_dpi}dpi ===")

        eps_path = Path(config.eps_file)
        width_cm, height_cm = self._resolve_canvas_size(eps_path, config)

        # 1. 栅格化 EPS（中等DPI）
        base = self.engine.open_eps(
            eps_path, dpi=preview_dpi,
            width_cm=width_cm,
            height_cm=height_cm,
        )

        w_px, h_px = base.size
        logger.info(f"[Step1] EPS栅格化: {w_px}x{h_px}px ({width_cm}x{height_cm}cm @ {preview_dpi}dpi)")

        # 2. 提取轮廓
        logger.info(f"[Step2] 开始提取轮廓 (num_levels=5) ...")
        contour_levels = self.contour_extractor.extract_contours(base, num_levels=5)
        if contour_levels:
            top = contour_levels[0]
            tx, ty, tw, th = top.bounding_rect
            sol = top.area / max(tw * th, 1)
            logger.info(f"[Step2] 轮廓提取完成: {len(contour_levels)}层, "
                        f"第0层 area={top.area}, rect=({tx},{ty},{tw},{th}), solidity={sol:.3f}")
        else:
            logger.warning("[Step2] 未提取到任何轮廓！")

        # 3. 加载PSD（PSD以原生分辨率加载，保持高质量）
        psd_path = Path(config.psd_file)
        psd_layers = self.engine.load_psd_layers(psd_path)
        if not psd_layers:
            raise RuntimeError("无可用图层")

        psd_composite = self._prepare_psd_composite(psd_path, psd_layers)
        logger.info(f"[Step3] 素材加载完成: {psd_path.suffix}, "
                    f"psd_composite={psd_composite.size if psd_composite else None}")
        classifications = self.layer_classifier.classify_layers(psd_layers, psd_composite)

        # 4. 多层渲染（优先使用PSD合成图整体渲染）
        if psd_composite:
            logger.info("[Step4] 使用 _render_with_psd_composite 路径")
            result = self._render_with_psd_composite(
                base, contour_levels, psd_composite, config
            )
        else:
            logger.info("[Step4] 使用 _render_multi_layer 路径")
            result = self._render_multi_layer(
                base, contour_levels, classifications, psd_composite, config
            )
        logger.info(f"[Step4] 渲染完成: {result.size}")

        # 5. 色彩调整
        result = self.color_adjuster.adjust(
            result,
            brightness=config.brightness,
            contrast=config.contrast,
            saturation=config.saturation,
            hue_shift=config.hue_shift,
            warmth=config.warmth,
        )
        logger.info(f"[Step5] 色彩调整完成")

        # 6. 缩放到GUI显示尺寸（以厘米为单位）
        display_dpi = 96  # 屏幕显示DPI (1cm ≈ 37.8px)
        target_width_px = int(max_width_cm / 2.54 * display_dpi)
        target_height_px = int(target_width_px * height_cm / width_cm)

        if result.width > target_width_px:
            ratio = target_width_px / result.width
            result = result.resize((target_width_px, int(result.height * ratio)), Image.LANCZOS)

        disp_w_cm = result.width / display_dpi * 2.54
        disp_h_cm = result.height / display_dpi * 2.54
        logger.info(f"[Step6] 预览生成: {result.width}x{result.height}px "
                     f"(显示约 {disp_w_cm:.1f}x{disp_h_cm:.1f}cm, "
                     f"实际 {width_cm:.1f}x{height_cm:.1f}cm)")

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
        5. 描边 = 相邻两层蒙版的差集（统一米黄色）
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

        # 7. 计算描边环（统一米黄色描边）
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

        # ★ 生成同心边框（跟随EPS轮廓，等比例缩放内层边框）
        canvas = self._generate_concentric_borders(
            canvas, base_mask, base_img, config
        )

        # 叠加CAD边框线
        canvas = self._overlay_cad_border(canvas, base_img, base_mask)

        return canvas.convert("RGB")

    def _render_with_psd_composite(self, base_img: Image.Image,
                                    contour_levels: List[ContourLevel],
                                    psd_composite: Image.Image,
                                    config: ProcessConfig) -> Image.Image:
        """使用PSD合成图直接渲染（v3：带结果校验+多轮fallback）

        主流程：独立X/Y缩放将PSD适配到EPS形状，用蒙版裁剪，叠加CAD边框。
        校验：若渲染结果中心区域几乎全白，说明蒙版选错/反了 → 依次fallback：
          1) 使用第2层轮廓 (levels[1])
          2) 反转蒙版 (mask_inv)
          3) 调用泛洪法重算蒙版 (ContourExtractor._extract_by_flood_fill)
        """
        w, h = base_img.size

        # 准备"基础蒙版+边界"的候选列表
        mask_candidates = []  # 每项: (desc, mask_img, bounding_rect)

        # 候选0: 最外层轮廓（原默认逻辑）
        if contour_levels:
            l0 = contour_levels[0]
            mask_candidates.append(("levels[0]（默认最外层轮廓）", l0.mask, l0.bounding_rect))

        # 候选1: 第2层轮廓 (如果存在)
        if len(contour_levels) >= 2:
            l1 = contour_levels[1]
            # 只在第2层面积足够大时考虑 (>40% 第0层)
            if l1.area > l0.area * 0.40:
                mask_candidates.append(("levels[1]（内层轮廓，可能是成品线）",
                                        l1.mask, l1.bounding_rect))

        # 候选2: 反转第0层蒙版 (防止里外搞反)
        if contour_levels:
            inv_mask = Image.fromarray(
                255 - np.array(l0.mask), mode="L"
            )
            # 计算反转后蒙版的 bounding_rect 和 area
            inv_arr = np.array(inv_mask)
            ys, xs = np.where(inv_arr > 128)
            if len(xs) > 0:
                ix, iy = int(xs.min()), int(ys.min())
                iw, ih = int(xs.max() - ix + 1), int(ys.max() - iy + 1)
                iarea = int(np.sum(inv_arr > 128))
                # 反转后面积也要合理（不能是整个画布的99%也不能<3%）
                total = w * h
                if 0.03 * total < iarea < 0.99 * total:
                    mask_candidates.append((
                        f"反转levels[0]蒙版（面积={iarea}）",
                        inv_mask, (ix, iy, iw, ih)
                    ))

        # 候选3: 泛洪法直接重算（独立于原contour_levels结果）
        try:
            flood = self.contour_extractor._extract_by_flood_fill(base_img)
            if flood is not None:
                mask_candidates.append((
                    f"泛洪法蒙版（FloodFill, 面积={flood.area}）",
                    flood.mask, flood.bounding_rect
                ))
        except Exception as e:
            logger.debug(f"泛洪法候选失败: {e}")

        logger.info(f"渲染蒙版候选: {len(mask_candidates)} 个")
        for i, (desc, m, r) in enumerate(mask_candidates):
            b = r
            sol = b[2] * b[3]
            logger.info(f"  [{i}] {desc}: rect({b[0]},{b[1]},{b[2]},{b[3]})")

        # 依次尝试每个蒙版候选
        for idx, (desc, mask, rect) in enumerate(mask_candidates):
            rendered = self._do_render_psd_with_mask(
                base_img, psd_composite, mask, rect, config
            )
            # 校验：中心10%区域是否>95%白色（"白屏"特征）
            white_ratio = self._center_white_ratio(rendered, radius=0.10)
            if white_ratio <= 0.95:
                logger.info(f"采用蒙版[{idx}] {desc} （中心白色占比={white_ratio:.2%}，合理）")
                return rendered
            else:
                logger.warning(
                    f"蒙版[{idx}] {desc} 渲染疑似失败：中心白色占比={white_ratio:.2%}，尝试下一个"
                )

        # 所有候选都失败，返回最后一个（至少比报错好）
        logger.warning("所有蒙版候选的中心都呈白色，使用默认第一个作为兜底")
        return self._do_render_psd_with_mask(
            base_img, psd_composite,
            mask_candidates[0][1] if mask_candidates else Image.new("L", (w, h), 255),
            mask_candidates[0][2] if mask_candidates else (0, 0, w, h),
            config,
        )

    def _do_render_psd_with_mask(self, base_img: Image.Image,
                                  psd_composite: Image.Image,
                                  mask: Image.Image,
                                  bounding_rect: Tuple[int, int, int, int],
                                  config: ProcessConfig) -> Image.Image:
        """核心渲染：指定蒙版+边界，缩放PSD合成图，裁剪，叠加细线边框。

        优化v6：
        - 生成多层同心细线边框（米黄色，跟随EPS轮廓）
        - CAD边框只在轮廓边界区域绘制，排除设计元素内部深色
        """
        w, h = base_img.size
        bx, by, bw, bh = bounding_rect

        # 验证蒙版有效性
        mask_arr_check = np.array(mask)
        mask_white_count = np.sum(mask_arr_check > 128)
        mask_total = w * h
        mask_ratio = mask_white_count / mask_total
        logger.info(f"蒙版检查: 白色像素={mask_white_count} ({mask_ratio*100:.1f}%), "
                     f"rect=({bx},{by},{bw},{bh})")

        # 如果蒙版几乎全黑（<1%白色），可能导致结果空白
        if mask_ratio < 0.01:
            logger.warning(f"蒙版几乎全黑({mask_ratio*100:.2f}%)，可能导致空白结果。尝试反转蒙版")
            mask_arr_check = 255 - mask_arr_check
            mask_white_count = np.sum(mask_arr_check > 128)
            mask_ratio = mask_white_count / mask_total
            logger.info(f"  反转后: 白色像素={mask_white_count} ({mask_ratio*100:.1f}%)")
            if mask_ratio >= 0.01:
                mask = Image.fromarray(mask_arr_check, mode="L")
                # 重新计算bounding_rect
                ys, xs = np.where(mask_arr_check > 128)
                if len(xs) > 0:
                    bx, by = int(xs.min()), int(ys.min())
                    bw, bh = int(xs.max() - bx + 1), int(ys.max() - by + 1)

        # PSD合成图转为RGBA
        composite = psd_composite.convert("RGBA")
        cw, ch = composite.size

        logger.info(f"PSD合成图: {cw}x{ch}px")

        # 独立计算X/Y缩放比（精确适配EPS形状，不保持比例）
        scale_x = bw / cw if cw > 0 else 1.0
        scale_y = bh / ch if ch > 0 else 1.0

        # 边界保护：缩放比例限制在合理范围
        max_scale = 20.0
        scale_x = min(scale_x, max_scale)
        scale_y = min(scale_y, max_scale)
        min_scale = 0.05
        scale_x = max(scale_x, min_scale)
        scale_y = max(scale_y, min_scale)

        new_w = max(1, int(cw * scale_x))
        new_h = max(1, int(ch * scale_y))

        logger.info(f"PSD适配: PSD={cw}x{ch}, 蒙版rect={bw}x{bh}, "
                     f"缩放X={scale_x:.4f}, Y={scale_y:.4f} -> {new_w}x{new_h}")

        # 使用独立X/Y缩放将PSD精确适配到EPS形状
        scaled_composite = composite.resize((new_w, new_h), Image.LANCZOS)

        # 对齐到EPS边界起点 (bounding_rect的左上角)
        offset_x = int(bx)
        offset_y = int(by)

        # 创建白色画布，合成PSD
        canvas = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        canvas.paste(scaled_composite, (offset_x, offset_y), scaled_composite)

        # 应用蒙版裁剪（蒙版外保持白色背景）
        mask_arr = np.array(mask)
        mask_bin = (mask_arr > 128)

        canvas_arr = np.array(canvas)
        for c in range(3):
            canvas_arr[:, :, c] = np.where(mask_bin, canvas_arr[:, :, c], 255)
        canvas_arr[:, :, 3] = 255

        canvas = Image.fromarray(canvas_arr, mode="RGBA")

        # ★ 生成多层同心边框（跟随EPS轮廓形状）
        canvas = self._generate_concentric_borders(
            canvas, mask, base_img, config
        )

        # 叠加CAD边框线（在最上层）
        canvas = self._overlay_cad_border(canvas, base_img, mask)

        # 验证渲染结果
        result_arr = np.array(canvas.convert("RGB"))
        non_white = np.sum(np.any(result_arr < 250, axis=2))
        total_pixels = w * h
        logger.info(f"渲染结果: 非白色像素={non_white} ({non_white/total_pixels*100:.1f}%)")

        return canvas.convert("RGB")

    def _center_white_ratio(self, img: Image.Image, radius: float = 0.10) -> float:
        """计算图像中心区域（边长=radius*2 的正方形）中白色像素的占比。

        用于检测"蒙版选错/反导致白屏"的典型症状。
        返回值: 0~1，越大表示越白；>0.95基本就是白屏失败。
        """
        arr = np.array(img.convert("RGB"))
        h, w = arr.shape[:2]
        cx, cy = w // 2, h // 2
        rx = max(3, int(w * radius))
        ry = max(3, int(h * radius))
        region = arr[cy - ry:cy + ry, cx - rx:cx + rx, :]
        if region.size == 0:
            return 0.0
        # 白色判定：RGB三个通道都>=250
        is_white = np.all(region >= 250, axis=-1)
        return float(is_white.sum()) / float(is_white.size)

    def _apply_stroke_ring(self, canvas: Image.Image,
                            stroke_ring_mask: Image.Image) -> Image.Image:
        """应用描边环效果

        在描边环区域填充细线描边。使用统一的米黄色调。
        """
        canvas_arr = np.array(canvas)
        mask_arr = np.array(stroke_ring_mask)
        mask_bin = (mask_arr > 128)

        if not np.any(mask_bin):
            return canvas

        # 统一米黄色描边（与设计风格一致）
        stroke_color = (205, 188, 148)

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

        从base中提取深色像素（CAD线稿），叠加在渲染结果上层。
        边框线自然跟随EPS外轮廓的弧度。

        增强v3：只在蒙版边界区域绘制CAD线稿，排除设计元素内部的深色区域。
        通过腐蚀蒙版创建"边界环"，只在环内绘制CAD线条。
        """
        import numpy as np

        base_arr = np.array(base_img.convert("RGBA"))
        canvas_arr = np.array(canvas)
        mask_arr = np.array(contour_mask)
        mask_bin = (mask_arr > 128)

        # 创建"边界环"区域：原蒙版 - 腐蚀后的蒙版
        # 这样只在轮廓边界附近绘制CAD线稿
        edge_width = max(3, int(min(base_img.width, base_img.height) * 0.005))
        eroded_mask = self.contour_extractor.erode_mask(contour_mask, edge_width)
        eroded_arr = np.array(eroded_mask)
        eroded_bin = (eroded_arr > 128)

        # 边界区域 = 在原蒙版内但不在腐蚀后蒙版内
        boundary_zone = mask_bin & (~eroded_bin)

        # 检测base中的深色像素（CAD线稿）
        # 高阈值(80)确保只捕获真正的深色线条
        is_dark = np.any(base_arr[:, :, :3] < 80, axis=2)

        # 边框 = 深色像素 AND 在边界区域内
        is_border = is_dark & boundary_zone

        # 如果边界像素太少，回退：扩大边界区域和阈值
        if np.sum(is_border) < 20:
            # 回退1：扩大边界宽度
            edge_width2 = edge_width * 3
            eroded_mask2 = self.contour_extractor.erode_mask(contour_mask, edge_width2)
            eroded_arr2 = np.array(eroded_mask2)
            eroded_bin2 = (eroded_arr2 > 128)
            boundary_zone2 = mask_bin & (~eroded_bin2)
            is_border = is_dark & boundary_zone2

        if np.sum(is_border) < 20:
            # 回退2：使用蒙版内全部深色像素（但提高阈值）
            is_dark = np.any(base_arr[:, :, :3] < 120, axis=2)
            is_border = is_dark & mask_bin
            logger.info(f"CAD边框: 使用回退方案，边框像素={np.sum(is_border)}")

        if np.any(is_border):
            # 边框线使用原CAD颜色（保持原汁原味）
            for c in range(3):
                canvas_arr[:, :, c] = np.where(
                    is_border, base_arr[:, :, c], canvas_arr[:, :, c]
                )
            canvas_arr[:, :, 3] = np.where(is_border, 255, canvas_arr[:, :, 3])
            logger.info(f"叠加CAD边框线: {np.sum(is_border)} 像素")

        return Image.fromarray(canvas_arr, mode="RGBA")

    def _generate_concentric_borders(self, canvas: Image.Image,
                                       outer_mask: Image.Image,
                                       base_img: Image.Image,
                                       config: ProcessConfig) -> Image.Image:
        """生成多层同心细线边框（跟随EPS轮廓形状）

        从外轮廓蒙版向内依次腐蚀，生成等比例缩放的细线边框。
        所有边框线都严格跟随EPS轮廓的弧度和形状。

        设计风格（与图二一致）：
        - 细线条（1-2px），不是粗带
        - 统一米黄色调，不渐深
        - 3层细线：外层、中层、内层
        """
        w, h = canvas.size

        # Get bounding rect from mask
        mask_arr = np.array(outer_mask)
        ys, xs = np.where(mask_arr > 128)
        if len(xs) == 0:
            return canvas

        bx, by = int(xs.min()), int(ys.min())
        bw, bh = int(xs.max() - bx + 1), int(ys.max() - by + 1)
        min_dim = min(bw, bh)
        if min_dim <= 0:
            return canvas

        # 边框线参数（细线风格）
        # 线条宽度：1-2px（细、优雅）
        line_width = max(1, int(min_dim * 0.003))
        # 线条位置（距外轮廓的内缩距离）
        line_insets = [
            max(2, int(min_dim * 0.010)),   # 第1条细线：离外轮廓 ~1%
            max(4, int(min_dim * 0.025)),   # 第2条细线：离外轮廓 ~2.5%
            max(6, int(min_dim * 0.045)),   # 第3条细线：离外轮廓 ~4.5%
        ]

        # 统一米黄色边框（与设计风格一致）
        border_color = (205, 188, 148)  # 米黄色

        canvas_arr = np.array(canvas)

        # 生成每条细线边框
        for inset in line_insets:
            # 腐蚀到该线条的位置
            eroded_outer = self.contour_extractor.erode_mask(outer_mask, inset)
            # 再腐蚀一个线条宽度作为内层边界
            eroded_inner = self.contour_extractor.erode_mask(outer_mask, inset + line_width)

            # 检查蒙版是否有效
            eo_arr = np.array(eroded_outer)
            ei_arr = np.array(eroded_inner)
            if np.sum(eo_arr > 128) < 10 or np.sum(ei_arr > 128) < 10:
                logger.info(f"边框线 inset={inset}px 蒙版过小，跳过")
                break

            # 生成细线环 = 外层腐蚀 - 内层腐蚀
            thin_line = self.contour_extractor.create_border_ring_mask(
                eroded_outer, eroded_inner
            )

            # 检查线环是否有效
            tl_arr = np.array(thin_line)
            if np.sum(tl_arr > 128) >= 1:
                canvas_arr = self._apply_ring_to_canvas(
                    canvas_arr, thin_line, border_color
                )

        logger.info(f"生成细线边框: {len(line_insets)}层, 颜色={border_color}")
        return Image.fromarray(canvas_arr, mode="RGBA")

    def _apply_ring_to_canvas(self, canvas_arr: np.ndarray,
                               ring_mask: Image.Image,
                               color: Tuple[int, int, int]) -> np.ndarray:
        """将边框环应用到画布数组"""
        mask_arr = np.array(ring_mask)
        mask_bin = (mask_arr > 128)

        if not np.any(mask_bin):
            return canvas_arr

        for c in range(3):
            canvas_arr[:, :, c] = np.where(
                mask_bin,
                color[c],
                canvas_arr[:, :, c]
            )
        canvas_arr[:, :, 3] = np.where(mask_bin, 255, canvas_arr[:, :, 3])

        return canvas_arr

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
