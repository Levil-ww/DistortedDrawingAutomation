#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
业务服务层 - 编排整个处理流程

通过依赖注入接收引擎和处理器，本身不包含任何图像算法。

依赖方向：services 依赖 engines + processors + models，不依赖 gui
"""

import logging
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from models import ProcessConfig
from engines.base import ImageEngine
from engines.pillow_engine import get_eps_bbox
from processors.aligner import SmartAligner
from processors.color_adjuster import ColorAdjuster
from processors.compositor import ImageCompositor

logger = logging.getLogger(__name__)


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
        """计算对齐参数（缩放 + 偏移）"""
        if config.smart_align and config.auto_scale:
            logger.info("使用智能对齐...")
            return self.aligner.align(base_img, pattern_img)
        else:
            scale = config.pattern_scale if config.pattern_scale != 1.0 else (
                min(base_img.width / pattern_img.width,
                    base_img.height / pattern_img.height) * 0.98
            )
            ox = config.pattern_offset_x or (base_img.width - int(pattern_img.width * scale)) // 2
            oy = config.pattern_offset_y or (base_img.height - int(pattern_img.height * scale)) // 2
            return scale, ox, oy

    def _create_eps_mask(self, base_img: Image.Image) -> Image.Image:
        """从 EPS 栅格化图像中提取 CAD 轮廓并填充为蒙版

        使用 OpenCV 检测非白色像素（CAD 线），找到最大轮廓并填充。
        返回 L 模式蒙版：255=轮廓区域内, 0=外部
        """
        try:
            import cv2
            import numpy as np

            cv_img = cv2.cvtColor(np.array(base_img), cv2.COLOR_RGB2GRAY)

            _, binary = cv2.threshold(cv_img, 250, 255, cv2.THRESH_BINARY_INV)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            binary = cv2.dilate(binary, kernel, iterations=1)
            binary = cv2.erode(binary, kernel, iterations=1)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                mask_img = np.zeros((base_img.height, base_img.width), dtype=np.uint8)
                cv2.drawContours(mask_img, contours, -1, 255, thickness=cv2.FILLED)
                total_pixels = base_img.width * base_img.height
                mask_pixels = int(np.sum(mask_img > 0))
                logger.info(f"EPS蒙版(轮廓填充): {mask_pixels}/{total_pixels} ({mask_pixels/total_pixels*100:.2f}%)")
                return Image.fromarray(mask_img, mode="L")
        except ImportError:
            logger.warning("OpenCV不可用，使用简化蒙版")

        # 回退：简单的非白色像素检测
        gray = base_img.convert("L")
        w, h = gray.size
        mask = Image.new("L", (w, h), 0)
        from PIL import Image as PILImage
        import numpy as np
        arr = np.array(gray)
        mask_arr = np.where(arr < 250, 255, 0).astype(np.uint8)
        mask = Image.fromarray(mask_arr, mode="L")
        return mask

    def _compose_with_eps_mask(self, base: Image.Image, pattern: Image.Image,
                                offset_x: int, offset_y: int,
                                eps_mask: Image.Image) -> Image.Image:
        """使用 EPS 轮廓蒙版合成图案

        将花纹粘贴到 base，然后用 EPS 轮廓蒙版裁剪 alpha 通道。
        """
        import numpy as np

        result = base.convert("RGBA")
        pattern_rgba = pattern.convert("RGBA")

        result.paste(pattern_rgba, (offset_x, offset_y), pattern_rgba)

        # 用 EPS 蒙版合成 alpha 通道
        result_arr = np.array(result)
        mask_arr = np.array(eps_mask)
        result_alpha = result_arr[:, :, 3].astype(np.uint16)
        mask_bin = (mask_arr > 128).astype(np.uint16)
        new_alpha = (result_alpha * mask_bin).astype(np.uint8)
        result_arr[:, :, 3] = new_alpha

        return Image.fromarray(result_arr, mode="RGBA").convert("RGB")

    # ---------- 辅助方法 ----------

    def _resolve_canvas_size(self, eps_path: Path, config: ProcessConfig) -> tuple:
        """解析画布尺寸：自动检测EPS bounding box，确保方向匹配

        逻辑：
        1. 解析EPS bounding box获取自然尺寸
        2. 如果用户配置的方向与EPS不一致 → 使用EPS方向的尺寸
        3. 如果一致 → 保持用户配置的尺寸
        """
        bbox = get_eps_bbox(eps_path)

        configured_w = config.canvas_width_cm
        configured_h = config.canvas_height_cm
        configured_is_landscape = configured_w > configured_h

        if bbox is not None:
            bbox_w, bbox_h = bbox
            bbox_is_landscape = bbox_w > bbox_h
            bbox_ratio = bbox_w / bbox_h if bbox_h > 0 else 1

            if configured_is_landscape != bbox_is_landscape:
                logger.info(f"画布方向不匹配: 配置={configured_w}x{configured_h}cm, "
                            f"EPS={bbox_w:.1f}x{bbox_h:.1f}cm, 自动调整方向")
                # 根据EPS方向调整，保持配置的面积近似
                area = configured_w * configured_h
                if bbox_is_landscape:
                    new_h = max(1.0, (area / bbox_ratio) ** 0.5)
                    new_w = new_h * bbox_ratio
                else:
                    new_w = max(1.0, (area * bbox_ratio) ** 0.5)
                    new_h = new_w / bbox_ratio
                return round(new_w, 1), round(new_h, 1)
            else:
                logger.info(f"画布方向匹配: {configured_w}x{configured_h}cm")
                return configured_w, configured_h
        else:
            logger.warning("无法解析EPS BoundingBox，使用配置尺寸")
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


