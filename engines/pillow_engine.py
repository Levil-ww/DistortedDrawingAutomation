#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pillow 纯Python引擎实现

无需外部依赖，使用Pillow/PIL处理所有图像操作。

重要：本模块只实现引擎接口，所有色彩调整等后处理逻辑
应交由上层的 DesignService 统一调用 processors 处理，
避免引擎层反向依赖 processors 包。
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List

from PIL import Image

from .base import ImageEngine
from models import EngineCapabilities, LayerInfo

logger = logging.getLogger(__name__)


class PillowEngine(ImageEngine):
    """Pillow 纯Python引擎"""

    def __init__(self):
        self._caps = EngineCapabilities(
            name="Pillow",
            supports_photoshop_native=False,
            supports_clipping_mask=False,
            supports_smart_align=True,
            supports_color_adjust=False,  # 由上层 processors 处理
            requires_external_app=False,
        )

    @property
    def capabilities(self) -> EngineCapabilities:
        return self._caps

    def open_eps(self, path: Path, dpi: int = 300) -> Image.Image:
        """栅格化EPS文件"""
        if not path.exists():
            raise FileNotFoundError(f"EPS文件不存在: {path}")

        if path.stat().st_size < 100:
            raise ValueError(f"EPS文件过小({path.stat().st_size}字节)，可能已损坏: {path}")

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        try:
            gs_cmd = [
                "gswin64c", "-dNOPAUSE", "-dBATCH", "-sDEVICE=png16m",
                f"-r{dpi}", f"-sOutputFile={tmp.name}", str(path)
            ]
            result = subprocess.run(gs_cmd, capture_output=True, timeout=60)
            if result.returncode == 0:
                img = Image.open(tmp.name).convert("RGB")
                logger.info(f"Ghostscript栅格化成功: {img.width}x{img.height}")
                if img.width > 10 and img.height > 10:
                    return img
                logger.warning(f"Ghostscript输出尺寸过小({img.width}x{img.height})，尝试回退")
            else:
                err_msg = result.stderr.decode(errors='ignore').strip()
                logger.warning(f"Ghostscript返回错误码 {result.returncode}: {err_msg[:200]}")
        except FileNotFoundError:
            logger.warning("Ghostscript未安装，回退到PIL直接打开")
        except Exception as e:
            logger.warning(f"Ghostscript执行失败: {e}")
        finally:
            Path(tmp.name).unlink(missing_ok=True)

        # 回退：PIL直接打开（仅支持嵌入预览，效果有限）
        try:
            img = Image.open(str(path))
            logger.warning(f"PIL回退打开EPS: 原始尺寸{img.width}x{img.height}, 模式{img.mode}")
            if img.mode != "RGB":
                img = img.convert("RGB")
            if img.width < 50 or img.height < 50:
                raise RuntimeError(
                    f"EPS栅格化失败: Ghostscript无法处理该文件，"
                    f"PIL回退仅获得{img.width}x{img.height}的图像。"
                    f"请确认EPS文件格式正确，或尝试使用其他格式"
                )
            return img
        except Exception as e:
            if "EPS栅格化失败" in str(e):
                raise
            raise RuntimeError(f"无法打开EPS文件: {e}") from e

    def load_psd_layers(self, path: Path) -> List[LayerInfo]:
        """加载PSD图层"""
        layers: List[LayerInfo] = []

        # 尝试psd-tools
        try:
            from psd_tools import PSDImage
            psd = PSDImage.open(str(path))
            for layer in psd:
                if layer.is_visible() and layer.has_pixels():
                    img = layer.composite()
                    if img:
                        layers.append(LayerInfo(layer.name, img))
            if layers:
                return layers
        except ImportError:
            logger.debug("psd-tools未安装")

        # 回退：使用目录中的PNG图层
        work_dir = path.parent
        for png in sorted(work_dir.glob("*_000*.png")):
            layers.append(LayerInfo(png.stem, Image.open(str(png))))

        if not layers:
            raise RuntimeError(f"无法从 {path} 加载任何图层")
        return layers

    def save_jpg(self, image: Image.Image, path: Path, quality: int = 95) -> None:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(str(path), "JPEG", quality=quality, optimize=True)
