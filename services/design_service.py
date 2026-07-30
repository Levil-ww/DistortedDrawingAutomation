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

from models import ProcessConfig, LayerInfo
from engines.base import ImageEngine
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

    def process(self, config: ProcessConfig, progress_callback: Optional[Callable[[str], None]] = None) -> Path:
        """
        执行完整的设计处理流程
        Args:
            config: 处理配置
            progress_callback: 进度回调函数，接收状态字符串
        Returns:
            输出文件路径
        """
        def _notify(msg: str):
            logger.info(msg)
            if progress_callback:
                progress_callback(msg)

        eps_path = Path(config.eps_file)
        psd_path = Path(config.psd_file)
        out_path = Path(config.output_file)

        _notify("正在打开EPS模板...")
        base_img = self.engine.open_eps(eps_path, dpi=config.dpi)

        _notify("正在加载PSD图层...")
        layers = self.engine.load_psd_layers(psd_path)
        if not layers:
            raise RuntimeError("PSD中未找到可用图层")

        pattern_layer = layers[0]
        pattern_img = pattern_layer.image
        if pattern_img.mode != "RGBA":
            pattern_img = pattern_img.convert("RGBA")

        _notify(f"使用花纹图层: {pattern_layer.name}")

        # 智能对齐 / 手动参数
        if config.smart_align and config.auto_scale:
            _notify("正在进行智能对齐...")
            scale, ox, oy = self.aligner.align(base_img, pattern_img)
            new_size = (int(pattern_img.width * scale), int(pattern_img.height * scale))
            pattern_img = pattern_img.resize(new_size, Image.LANCZOS)
        else:
            if config.pattern_scale != 1.0:
                new_size = (
                    int(pattern_img.width * config.pattern_scale),
                    int(pattern_img.height * config.pattern_scale),
                )
                pattern_img = pattern_img.resize(new_size, Image.LANCZOS)
            ox = config.pattern_offset_x or (base_img.width - pattern_img.width) // 2
            oy = config.pattern_offset_y or (base_img.height - pattern_img.height) // 2

        # 查找蒙版
        mask = self._find_mask(layers, base_img.size)

        _notify("正在合成图像...")
        result = self.compositor.compose(base_img, pattern_img, ox, oy, mask)

        _notify("正在应用色彩调整...")
        result = self.color_adjuster.adjust(
            result,
            brightness=config.brightness,
            contrast=config.contrast,
            saturation=config.saturation,
            hue_shift=config.hue_shift,
            warmth=config.warmth,
        )

        _notify("正在保存JPG...")
        self.engine.save_jpg(result, out_path, quality=config.jpg_quality)

        _notify(f"处理完成: {out_path}")
        return out_path

    def generate_preview(self, config: ProcessConfig, max_width: int = 400) -> Image.Image:
        """
        生成低分辨率预览图（用于GUI快速显示）

        策略：
        1. 优先尝试用 psd-tools 渲染完整 PSD 合成图
        2. 将 PSD 合成图与 EPS 模板居中叠加
        """
        base = self.engine.open_eps(Path(config.eps_file), dpi=72)
        logger.info(f"EPS原始尺寸: {base.width}x{base.height}")

        psd_path = Path(config.psd_file)
        layers = self.engine.load_psd_layers(psd_path)
        if not layers:
            raise RuntimeError("无可用图层")

        # 优先尝试渲染完整 PSD 合成图（保留图层叠加效果）
        psd_composite = self._render_psd_composite(psd_path, layers)
        if psd_composite is None:
            # 回退：智能选择花纹图层
            psd_composite = self._select_pattern_layer(layers)
            if psd_composite is None:
                psd_composite = layers[0].image.convert("RGBA")
        logger.info(f"PSD合成图: {psd_composite.width}x{psd_composite.height}")

        if base.width < 50 or base.height < 50:
            logger.warning(f"EPS栅格化结果异常({base.width}x{base.height})，使用默认画布")
            default_w, default_h = max_width, int(max_width * 1.5)
            base_small = base.resize((default_w, default_h), Image.LANCZOS)
        else:
            ratio = max_width / base.width
            base_small = base.resize((max_width, int(base.height * ratio)), Image.LANCZOS)

        target_ratio = min(
            base_small.width / psd_composite.width,
            base_small.height / psd_composite.height
        ) * 0.95
        target_ratio = min(target_ratio, 1.0)
        psd_small = psd_composite.resize(
            (max(1, int(psd_composite.width * target_ratio)),
             max(1, int(psd_composite.height * target_ratio))),
            Image.LANCZOS
        )

        result = base_small.convert("RGBA")
        ox = max(0, (result.width - psd_small.width) // 2)
        oy = max(0, (result.height - psd_small.height) // 2)
        logger.info(
            f"合成: base={result.width}x{result.height}, "
            f"psd={psd_small.width}x{psd_small.height}, offset=({ox},{oy})"
        )
        result.paste(psd_small, (ox, oy), psd_small)
        return result.convert("RGB")

    def _render_psd_composite(self, psd_path: Path, layers) -> Optional[Image.Image]:
        """尝试用 psd-tools 渲染 PSD 完整合成图

        使用 psd.composite() 直接得到所有图层叠加后的完整图。
        """
        try:
            from psd_tools import PSDImage
            psd = PSDImage.open(str(psd_path))
            # psd.composite() 会渲染所有可见图层叠加后的完整图
            img = psd.composite()
            if img is not None:
                if img.mode != "RGBA":
                    img = img.convert("RGBA")
                logger.info(f"使用 psd-tools 完整合成: {img.width}x{img.height}")
                return img
        except Exception as e:
            logger.debug(f"psd-tools 合成失败，回退到单图层: {e}")
        return None

    def _select_pattern_layer(self, layers) -> Optional[Image.Image]:
        """从PSD图层中选择最合适的花纹图层

        优先选择"图形"层而非"填充"层：
        - 图形层特征：部分透明背景，且非透明像素中图形（非白/非纯色）占比较高
        - 填充层特征：完全不透明或大面积纯色
        """
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

            # 评分：优先"半透明 + 图形密集"图层
            # 完全不透明的填充层（透明率=0）评分低
            # 全透明的空层评分=0
            transparency_ratio = (total - non_transparent) / total
            if non_transparent == 0 or non_white == 0:
                score = 0
            else:
                # 图形密度 = 非白像素 / 总像素
                graphic_density = non_white / total
                # 透明率（花纹层通常有透明区域）
                # 优先：图形密度高 + 有适当透明率（10%-70%）
                transparency_bonus = 0
                if 0.1 < transparency_ratio < 0.7:
                    transparency_bonus = 5
                score = graphic_density * 1000 + transparency_bonus

            logger.debug(
                f"图层 {layer.name}: 非白={non_white}, 非透明={non_transparent}/{total}, "
                f"透明率={transparency_ratio:.2f}, 得分={score:.1f}"
            )
            if score > best_score:
                best_score = score
                best = img
        if best_score <= 0:
            return None
        return best

    def _find_mask(self, layers, target_size) -> Optional[Image.Image]:
        """从图层列表中查找可能的蒙版图层"""
        for layer in layers[1:]:
            gray = layer.image.convert("L")
            avg = sum(gray.getdata()) / (gray.width * gray.height)
            if avg < 200:
                return gray.resize(target_size, Image.LANCZOS)
        return None
