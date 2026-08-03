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
import re
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image

from .base import ImageEngine
from models import EngineCapabilities, LayerInfo

logger = logging.getLogger(__name__)


def get_eps_bbox(path: Path) -> Optional[Tuple[float, float]]:
    """解析EPS文件的BoundingBox，返回 (宽度cm, 高度cm) 或 None

    PostScript点数转厘米: points / 72 inches_per_point * 2.54 cm/inch
    """
    try:
        with open(path, 'rb') as f:
            raw = f.read(4096)
        text = raw.decode('latin-1', errors='ignore')
        m = re.search(r'%%BoundingBox:\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)', text)
        if not m:
            logger.warning("未找到BoundingBox")
            return None
        x1, y1, x2, y2 = [float(m.group(i)) for i in range(1, 5)]
        w_pts = abs(x2 - x1)
        h_pts = abs(y2 - y1)
        w_cm = w_pts / 72.0 * 2.54
        h_cm = h_pts / 72.0 * 2.54
        logger.info(f"EPS BoundingBox: {w_pts:.0f}x{h_pts:.0f}pt = {w_cm:.1f}x{h_cm:.1f}cm "
                     f"({'横版' if w_cm > h_cm else '竖版'})")
        return (w_cm, h_cm)
    except Exception as e:
        logger.warning(f"解析BoundingBox失败: {e}")
        return None


class PillowEngine(ImageEngine):
    """Pillow 纯Python引擎"""

    def __init__(self):
        self._caps = EngineCapabilities(
            name="Pillow",
            supports_photoshop_native=False,
            supports_clipping_mask=False,
            supports_smart_align=True,
            supports_color_adjust=False,
            requires_external_app=False,
        )
        Image.MAX_IMAGE_PIXELS = None

    @property
    def capabilities(self) -> EngineCapabilities:
        return self._caps

    def open_eps(self, path: Path, dpi: int = 300,
                 width_cm: float = None, height_cm: float = None) -> Image.Image:
        """栅格化EPS文件

        当指定 width_cm / height_cm 时，按物理尺寸 + DPI 计算像素，
        通过 Ghostscript 的 -dDEVICEWIDTHPOINTS / -dDEVICEHEIGHTPOINTS 强制输出尺寸。
        """
        if not path.exists():
            raise FileNotFoundError(f"EPS文件不存在: {path}")

        if path.stat().st_size < 100:
            raise ValueError(f"EPS文件过小({path.stat().st_size}字节)，可能已损坏: {path}")

        # 计算目标像素尺寸
        if width_cm and height_cm:
            target_w = max(1, int(width_cm / 2.54 * dpi))
            target_h = max(1, int(height_cm / 2.54 * dpi))
            logger.info(f"目标物理尺寸: {width_cm}x{height_cm}cm @ {dpi}dpi = {target_w}x{target_h}px")
        else:
            target_w = target_h = None

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        try:
            gs_cmd = [
                "gswin64c", "-dNOPAUSE", "-dBATCH", "-sDEVICE=png16m",
                f"-r{dpi}",
            ]
            # 强制设备尺寸（点数 = cm / 2.54 * 72）
            if width_cm and height_cm:
                pts_w = max(1, int(width_cm / 2.54 * 72))
                pts_h = max(1, int(height_cm / 2.54 * 72))
                gs_cmd += [f"-dDEVICEWIDTHPOINTS={pts_w}", f"-dDEVICEHEIGHTPOINTS={pts_h}"]
                gs_cmd += ["-dFitPage"]
            gs_cmd += [f"-sOutputFile={tmp.name}", str(path)]

            result = subprocess.run(gs_cmd, capture_output=True, timeout=120)
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
            if target_w and target_h and (img.width != target_w or img.height != target_h):
                img = img.resize((target_w, target_h), Image.LANCZOS)
                logger.info(f"PIL回退: 缩放到目标尺寸 {target_w}x{target_h}")
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

    def _load_plain_image_as_composite(self, path: Path) -> List[LayerInfo]:
        """将普通图片（JPG/PNG/BMP等）包装为PSD合成图层格式

        直接加载整张图，包装为 __psd_composite__ 图层，
        这样上层可以复用 _render_with_psd_composite 的渲染逻辑。
        """
        try:
            img = Image.open(str(path))
            # 转为RGBA（如果是普通JPG则添加不透明alpha通道）
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            layers = [LayerInfo("__psd_composite__", img)]
            logger.info(f"加载普通图片素材: {path.suffix} {img.width}x{img.height} (作为合成图层)")
            return layers
        except Exception as e:
            raise RuntimeError(f"无法加载图片 {path}: {e}") from e

    def load_psd_layers(self, path: Path) -> List[LayerInfo]:
        """加载素材图层（智能识别格式）

        - .psd: 使用 psd-tools 分图层加载，并附带完整合成图
        - .jpg/.jpeg/.png/.bmp/.tif/.tiff/.webp 等普通图片: 直接包装为合成图层
        """
        suffix = path.suffix.lower()

        # 非PSD格式走普通图片加载
        psd_exts = {".psd", ".psb"}
        if suffix not in psd_exts:
            return self._load_plain_image_as_composite(path)

        # PSD 格式按原逻辑分图层加载
        layers: List[LayerInfo] = []
        skipped_count = 0
        total_count = 0

        try:
            from psd_tools import PSDImage
            psd = PSDImage.open(str(path))
            total_count = len(list(psd))

            for layer in psd:
                if not layer.is_visible():
                    continue

                # 处理矢量智能对象
                if hasattr(layer, 'smart_object') and layer.smart_object is not None:
                    try:
                        img = layer.composite()
                        if img:
                            layers.append(LayerInfo(layer.name, img))
                            logger.info(f"加载智能对象图层 '{layer.name}': {img.width}x{img.height}")
                            continue
                    except Exception as e:
                        logger.warning(f"智能对象图层 '{layer.name}' composite() 失败: {e}")

                    # 尝试导出智能对象内容
                    try:
                        import tempfile
                        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                        tmp.close()
                        layer.smart_object.export(tmp.name)
                        img = Image.open(tmp.name)
                        img = img.convert("RGBA")
                        Path(tmp.name).unlink(missing_ok=True)
                        layers.append(LayerInfo(layer.name, img))
                        logger.info(f"通过export()加载智能对象 '{layer.name}': {img.width}x{img.height}")
                        continue
                    except Exception as e2:
                        logger.warning(f"智能对象图层 '{layer.name}' export() 也失败: {e2}")
                        skipped_count += 1
                        continue

                # 普通图层
                if layer.has_pixels():
                    img = layer.composite()
                    if img:
                        layers.append(LayerInfo(layer.name, img))
                else:
                    skipped_count += 1

            # 总是追加PSD合成图（确保包含所有效果、智能对象的正确渲染）
            try:
                composite = psd.composite()
                if composite:
                    if composite.mode != "RGBA":
                        composite = composite.convert("RGBA")
                    layers.append(LayerInfo("__psd_composite__", composite))
                    logger.info(f"PSD合成图: {composite.width}x{composite.height}")
            except Exception as e:
                logger.warning(f"PSD合成图导出失败: {e}")

            if layers:
                logger.info(f"psd-tools加载完成: {len(layers)} 个图层 (跳过 {skipped_count})")
                return layers
        except ImportError:
            logger.debug("psd-tools未安装")
        except Exception as e:
            logger.warning(f"psd-tools加载PSD图层出错: {e}")

        # 回退：使用目录中的PNG图层
        work_dir = path.parent
        png_layers = sorted(work_dir.glob("*_000*.png"))
        if png_layers:
            layers = [LayerInfo(png.stem, Image.open(str(png))) for png in png_layers]
            logger.info(f"从PNG回退加载: {len(layers)} 个图层")
            return layers

        if not layers:
            raise RuntimeError(f"无法从 {path} 加载任何图层")
        return layers

    def save_jpg(self, image: Image.Image, path: Path, quality: int = 95) -> None:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(str(path), "JPEG", quality=quality, optimize=True)
