#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
业务服务层 - 编排整个处理流程

通过依赖注入接收引擎和处理器，本身不包含任何图像算法。

依赖方向：services 依赖 engines + processors + models，不依赖 gui
"""

import logging
import re
from pathlib import Path
from typing import Callable, Optional, Tuple

from PIL import Image

from models import ProcessConfig
from engines.base import ImageEngine
from engines.pillow_engine import get_eps_bbox
from processors.aligner import SmartAligner
from processors.color_adjuster import ColorAdjuster
from processors.compositor import ImageCompositor

logger = logging.getLogger(__name__)


def _parse_psd_size_from_filename(psd_path: Path) -> Optional[Tuple[float, float]]:
    """从PSD文件名解析素材尺寸 (width_cm, height_cm)

    支持格式:
        蔓生花80-140.psd -> (80, 140)
        pattern_80x140.psd -> (80, 140)
        design_80_140.psd -> (80, 140)
    """
    name = psd_path.stem
    # 尝试匹配 80-140, 80x140, 80_140 格式
    m = re.search(r'(?i)(\d+)[\-_xX](\d+)', name)
    if m:
        w, h = float(m.group(1)), float(m.group(2))
        if w > 0 and h > 0:
            logger.info(f"从文件名解析PSD尺寸: {name} -> {w:.0f}x{h:.0f}cm")
            return (w, h)
    return None


class DesignService:
    """
    设计自动化业务服务
    职责：编排处理流程（打开->对齐->合成->调整->保存）
    不直接处理图像，全部委托给注入的组件
    """

    def __init__(
        self,
        engine: ImageEngine,
        aligner: Optional[SmartAligner] = None,
        color_adjuster: Optional[ColorAdjuster] = None,
        compositor: Optional[ImageCompositor] = None,
    ):
        self.engine = engine
        self.aligner = aligner or SmartAligner()
        self.color_adjuster = color_adjuster or ColorAdjuster()
        self.compositor = compositor or ImageCompositor()

    # ---------- 核心流程 ----------

    def process(self, config: ProcessConfig,
                progress_callback: Optional[Callable[[str], None]] = None) -> Path:
        """
        执行完整的设计处理流程

        流程：EPS方向检测 → 栅格化 → PSD合成 → 对齐 → EPS轮廓蒙版裁剪 → 合成 → 色彩调整 → 保存
        """
        def _notify(msg: str):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        eps_path = Path(config.eps_file)
        psd_path = Path(config.psd_file)
        out_path = Path(config.output_file)

        # 0. 自动检测EPS方向并调整画布尺寸
        width_cm, height_cm = self._resolve_canvas_size(eps_path, config)
        _notify(f"画布尺寸: {width_cm:.1f}x{height_cm:.1f}cm ({'横版' if width_cm > height_cm else '竖版'})")

        # 1. 栅格化 EPS
        _notify("正在打开EPS模板...")
        base_img = self.engine.open_eps(
            eps_path, dpi=config.dpi,
            width_cm=width_cm,
            height_cm=height_cm,
        )
        logger.info(f"EPS栅格化: {base_img.width}x{base_img.height}px "
                     f"({width_cm:.1f}x{height_cm:.1f}cm @ {config.dpi}dpi)")

        # 2. 加载 PSD 图层
        _notify("正在加载PSD图层...")
        layers = self.engine.load_psd_layers(psd_path)
        if not layers:
            raise RuntimeError("PSD中未找到可用图层")

        # 3. 获取 PSD 完整合成图作为花纹源
        _notify("正在合成PSD花纹...")
        pattern_img = self._prepare_pattern(psd_path, layers)
        logger.info(f"花纹源: {pattern_img.width}x{pattern_img.height}, mode={pattern_img.mode}")

        # 4. 计算对齐参数
        scale, ox, oy = self._compute_alignment(base_img, pattern_img, config)
        logger.info(f"对齐: 缩放={scale:.4f}, 偏移=({ox},{oy})")

        # 5. 缩放花纹图案
        new_size = (int(pattern_img.width * scale), int(pattern_img.height * scale))
        pattern_img = pattern_img.resize(new_size, Image.LANCZOS)
        logger.info(f"缩放后花纹: {pattern_img.width}x{pattern_img.height}")

        # 6. 生成 EPS 轮廓蒙版（从 base 中提取非白色像素）
        eps_mask = self._create_eps_mask(base_img)
        logger.info(f"EPS轮廓蒙版: {eps_mask.width}x{eps_mask.height}")

        # 7. 使用蒙版裁剪花纹并合成
        _notify("正在合成图像...")
        result = self._compose_with_eps_mask(base_img, pattern_img, ox, oy, eps_mask)

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
        """
        生成预览图（复刻 process 流程，使用低 DPI + 缩放显示）
        """
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
        logger.info(f"EPS预览: {base.width}x{base.height} ({width_cm:.1f}x{height_cm:.1f}cm)")

        # 2. PSD 合成
        psd_path = Path(config.psd_file)
        layers = self.engine.load_psd_layers(psd_path)
        if not layers:
            raise RuntimeError("无可用图层")
        pattern_img = self._prepare_pattern(psd_path, layers)

        # 3. 对齐
        scale, ox, oy = self._compute_alignment(base, pattern_img, config)

        # 4. 缩放花纹
        new_size = (int(pattern_img.width * scale), int(pattern_img.height * scale))
        pattern_img = pattern_img.resize(new_size, Image.LANCZOS)

        # 5. EPS 轮廓蒙版
        eps_mask = self._create_eps_mask(base)

        # 6. 合成
        result = self._compose_with_eps_mask(base, pattern_img, ox, oy, eps_mask)

        # 7. 色彩调整
        result = self.color_adjuster.adjust(
            result,
            brightness=config.brightness,
            contrast=config.contrast,
            saturation=config.saturation,
            hue_shift=config.hue_shift,
            warmth=config.warmth,
        )

        # 8. 缩放到显示尺寸
        if result.width > max_width:
            ratio = max_width / result.width
            display_size = (max_width, int(result.height * ratio))
            result = result.resize(display_size, Image.LANCZOS)
            logger.info(f"预览缩放: {result.width}x{result.height}")

        return result

    # ---------- 核心辅助方法 ----------

    def _compute_alignment(self, base_img: Image.Image, pattern_img: Image.Image,
                           config: ProcessConfig):
        """计算对齐参数（缩放 + 偏移）

        使用 cover 模式（max缩放），确保图案完全覆盖边框区域，无留白。
        """
        if config.smart_align and config.auto_scale:
            logger.info("使用智能对齐[cover]...")
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
        """从 EPS 栅格化图像中提取外轮廓作为填充蒙版

        使用轮廓检测找到EPS形状的外边界，填充整个内部区域作为蒙版。
        这样PSD素材能正确填充到EPS形状内部，且边框弧度与外轮廓一致。
        返回 L 模式蒙版：255=填充区域内, 0=外部
        """
        import numpy as np

        try:
            import cv2

            gray = cv2.cvtColor(np.array(base_img), cv2.COLOR_RGB2GRAY)

            # 二值化：背景变白(255)，CAD线框/图形变黑(0)
            _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)

            # 形态学闭运算：闭合线框的微小缺口，形成完整轮廓
            kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close, iterations=3)

            # 转换为线为白色、背景为黑色的图以便轮廓检测
            # 背景=0(黑), 线框=255(白)
            inv = cv2.bitwise_not(closed)

            # 再做一次膨胀腐蚀，确保轮廓完整
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            inv = cv2.dilate(inv, kernel_dilate, iterations=2)
            inv = cv2.erode(inv, kernel_dilate, iterations=1)

            # 查找外轮廓（使用RETR_EXTERNAL只获取最外层轮廓）
            contours, hierarchy = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                logger.warning("未检测到任何轮廓，尝试回退方案")
                return self._create_eps_mask_fallback(base_img)

            # 找到面积最大的轮廓（即EPS形状的外轮廓）
            largest_contour = max(contours, key=cv2.contourArea)

            # 创建蒙版：填充最大轮廓的内部区域
            mask_img = np.zeros((base_img.height, base_img.width), dtype=np.uint8)
            cv2.fillPoly(mask_img, [largest_contour], 255)

            # 形态学开运算去除边缘毛刺
            kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            mask_img = cv2.morphologyEx(mask_img, cv2.MORPH_OPEN, kernel_open, iterations=1)

            total_pixels = base_img.width * base_img.height
            mask_pixels = int(np.sum(mask_img > 0))
            ratio = mask_pixels / total_pixels * 100
            logger.info(f"EPS蒙版(外轮廓填充): {mask_pixels}/{total_pixels} ({ratio:.2f}%)")

            if mask_pixels > 0:
                return Image.fromarray(mask_img, mode="L")
            else:
                logger.warning("轮廓面积为0，尝试回退方案")
        except ImportError:
            logger.warning("OpenCV未安装，使用回退方案")
        except Exception as e:
            logger.warning(f"外轮廓蒙版失败: {e}")

        return self._create_eps_mask_fallback(base_img)

    def _create_eps_mask_fallback(self, base_img: Image.Image) -> Image.Image:
        """回退方案：使用边框矩形检测"""
        import numpy as np

        try:
            import cv2

            gray = cv2.cvtColor(np.array(base_img), cv2.COLOR_RGB2GRAY)
            _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)

            # 膨胀连接所有线条
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
            dilated = cv2.dilate(binary, kernel, iterations=3)
            eroded = cv2.erode(dilated, kernel, iterations=2)

            # 从四角做floodFill标记背景
            h, w = eroded.shape
            flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
            flood_img = eroded.copy()

            seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
            for sx, sy in seeds:
                if 0 <= sx < w and 0 <= sy < h and flood_img[sy, sx] == 255:
                    cv2.floodFill(flood_img, flood_mask, (sx, sy), 128)

            # 封闭区域：值仍为255
            region_mask = np.where(flood_img == 255, 255, 0).astype(np.uint8)

            if np.sum(region_mask > 0) > 0:
                logger.info("EPS蒙版(回退floodFill)")
                return Image.fromarray(region_mask, mode="L")
        except Exception:
            pass

        # 最终回退：全图蒙版
        logger.warning("所有蒙版检测失败，使用全图蒙版")
        return Image.new("L", base_img.size, 255)

    def _compose_with_eps_mask(self, base: Image.Image, pattern: Image.Image,
                                offset_x: int, offset_y: int,
                                eps_mask: Image.Image,
                                overlay_border: bool = True) -> Image.Image:
        """使用 EPS 轮廓蒙版合成图案

        合成逻辑：
        1. 在画布上放置缩放后的PSD花纹图案
        2. 用EPS蒙版裁剪（图案仅在EPS形状内部可见）
        3. 叠加CAD边框线（确保边框线在花纹上层）
        4. 边框弧度与EPS外轮廓保持一致

        Args:
            base: 原始EPS栅格化图像（含CAD线稿）
            pattern: 缩放后的PSD花纹图像
            offset_x, offset_y: 花纹放置位置
            eps_mask: EPS轮廓蒙版（L模式）
            overlay_border: 是否叠加CAD边框线
        """
        import numpy as np

        w, h = base.size

        # 1. 创建画布并放置花纹图案
        canvas = Image.new("RGBA", (w, h), (255, 255, 255, 255))
        pattern_rgba = pattern.convert("RGBA")
        canvas.paste(pattern_rgba, (offset_x, offset_y), pattern_rgba)

        canvas_arr = np.array(canvas)
        mask_arr = np.array(eps_mask)
        mask_bin = (mask_arr > 128).astype(np.uint8)

        # 2. 用蒙版裁剪：形状内部保留花纹，外部设为白色
        for c in range(3):
            canvas_arr[:, :, c] = np.where(mask_bin, canvas_arr[:, :, c], 255)
        canvas_arr[:, :, 3] = 255

        # 3. 叠加 CAD 边框线（从base中提取深色像素，叠加在花纹上层）
        #    边框线会自然跟随EPS外轮廓的弧度
        if overlay_border:
            base_arr = np.array(base.convert("RGBA"))
            # 检测base中的深色像素（CAD线稿）
            is_dark = np.any(base_arr[:, :, :3] < 120, axis=2)
            # 仅保留形状内部的深色像素作为边框
            is_border = is_dark & (mask_bin > 0)

            if np.any(is_border):
                for c in range(3):
                    canvas_arr[:, :, c] = np.where(
                        is_border, base_arr[:, :, c], canvas_arr[:, :, c]
                    )
                logger.info(f"叠加CAD边框线: {np.sum(is_border)} 像素")

        return Image.fromarray(canvas_arr, mode="RGBA").convert("RGB")

    # ---------- 辅助方法 ----------

    def _resolve_canvas_size(self, eps_path: Path, config: ProcessConfig) -> tuple:
        """解析画布尺寸：以EPS文件尺寸为准，PSD仅作方向参考

        逻辑：
        1. 优先从EPS BoundingBox获取物理尺寸作为画布大小
        2. 若EPS尺寸获取失败，回退到配置的画布尺寸
        3. 若PSD方向与EPS不匹配，自动交换画布宽高
        """
        bbox = get_eps_bbox(eps_path)

        if bbox is not None:
            eps_w, eps_h = bbox
            bbox_is_landscape = eps_w > eps_h

            # 检查PSD方向是否与EPS匹配
            psd_path = Path(config.psd_file) if config.psd_file else None
            psd_size = _parse_psd_size_from_filename(psd_path) if psd_path else None
            if psd_size is not None:
                psd_w, psd_h = psd_size
                psd_is_landscape = psd_w > psd_h
                if psd_is_landscape != bbox_is_landscape:
                    logger.info(f"PSD方向与EPS不匹配，交换画布方向: "
                                f"EPS={eps_w:.1f}x{eps_h:.1f}cm -> "
                                f"{eps_h:.1f}x{eps_w:.1f}cm")
                    return round(eps_h, 1), round(eps_w, 1)

            logger.info(f"使用EPS尺寸作为画布: {eps_w:.1f}x{eps_h:.1f}cm "
                        f"({'横版' if bbox_is_landscape else '竖版'})")
            return round(eps_w, 1), round(eps_h, 1)

        # 回退：使用配置的尺寸
        configured_w = config.canvas_width_cm
        configured_h = config.canvas_height_cm
        logger.warning(f"无法解析EPS BoundingBox，使用配置尺寸: {configured_w}x{configured_h}cm")
        return configured_w, configured_h

    def _prepare_pattern(self, psd_path: Path, layers) -> Image.Image:
        """获取花纹源图像

        优先使用 psd-tools 的 composite() 获得完整合成图，
        回退到智能选择最佳图层。
        """
        composite = self._render_psd_composite(psd_path)
        if composite is not None:
            return composite

        best = self._select_pattern_layer(layers)
        if best is not None:
            return best

        logger.warning("未找到合适的花纹图层，使用第一个图层")
        return layers[0].image.convert("RGBA")

    def _render_psd_composite(self, psd_path: Path) -> Optional[Image.Image]:
        """尝试用 psd-tools 渲染 PSD 完整合成图"""
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
        """回退方案：从图层中选择最佳花纹图层"""
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


