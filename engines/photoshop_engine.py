#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Photoshop COM 引擎实现

通过 win32com 与 Photoshop 通信，利用 PS 的原生能力处理。

重要：本模块只实现引擎接口，所有色彩调整等后处理逻辑
应交由上层的 DesignService 统一调用 processors 处理，
避免引擎层反向依赖 processors 包。
"""

import logging
import tempfile
from pathlib import Path
from typing import List

from PIL import Image

from .base import ImageEngine
from models import EngineCapabilities, LayerInfo

logger = logging.getLogger(__name__)


class PhotoshopEngine(ImageEngine):
    """Photoshop COM 引擎"""

    def __init__(self):
        self._ps_app = self._connect()
        self._caps = EngineCapabilities(
            name="Photoshop COM",
            supports_photoshop_native=True,
            supports_clipping_mask=True,
            supports_smart_align=True,
            supports_color_adjust=False,  # 由上层 processors 处理
            requires_external_app=True,
        )

    @property
    def capabilities(self) -> EngineCapabilities:
        return self._caps

    def _connect(self):
        """连接 Photoshop COM"""
        try:
            import win32com.client
            ps = win32com.client.Dispatch("Photoshop.Application")
            ps.displayDialogs = 3  # DisplayNoDialogs
            logger.info(f"Photoshop COM 已连接 (v{ps.version})")
            return ps
        except ImportError as e:
            raise RuntimeError("未安装 pywin32，无法使用 Photoshop 引擎") from e
        except Exception as e:
            raise RuntimeError(f"连接 Photoshop 失败: {e}") from e

    # ---------- 文件操作 ----------

    def open_eps(self, path: Path, dpi: int = 300,
                 width_cm: float = None, height_cm: float = None) -> Image.Image:
        path_ps = str(path).replace("\\", "/")
        extra = ""
        if width_cm and height_cm:
            width_in = width_cm / 2.54
            height_in = height_cm / 2.54
            extra = f'''
        opt.width = {width_in};
        opt.height = {height_in};'''
        js = f'''
        var f = new File("{path_ps}");
        var opt = new EPSOpenOptions();
        opt.antiAlias = true;
        opt.resolution = {dpi};
        opt.mode = OpenDocumentMode.RGB;
        opt.cropTo = CropToType.MediaBox;{extra}
        app.open(f, opt);
        '''
        self._ps_app.doJavaScript(js)
        return self._active_doc_to_pil()

    def _load_plain_image_as_composite(self, path: Path) -> List[LayerInfo]:
        """将普通图片（JPG/PNG/BMP等）包装为PSD合成图层格式"""
        try:
            img = Image.open(str(path))
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            layers = [LayerInfo("__psd_composite__", img)]
            logger.info(f"加载普通图片素材 (PS引擎): {path.suffix} {img.width}x{img.height}")
            return layers
        except Exception as e:
            raise RuntimeError(f"无法加载图片 {path}: {e}") from e

    def load_psd_layers(self, path: Path) -> List[LayerInfo]:
        suffix = path.suffix.lower()
        psd_exts = {".psd", ".psb"}
        if suffix not in psd_exts:
            # 非PSD格式，直接用Pillow加载为合成图层（无需启动PS）
            return self._load_plain_image_as_composite(path)

        doc = self._ps_app.Open(str(path))
        layers: List[LayerInfo] = []
        try:
            all_layers = list(doc.layers)
            total = len(all_layers)
            skipped = 0

            for layer in all_layers:
                if not layer.visible:
                    continue

                typename = layer.typename
                if typename not in ("ArtLayer", "SmartObjectLayer"):
                    # 尝试其他图层类型（如文字层、调整层等）
                    if typename not in ("TextLayer", "AdjustmentLayer"):
                        skipped += 1
                        continue

                try:
                    self._ps_app.activeDocument = doc
                    doc.activeLayer = layer

                    if typename == "SmartObjectLayer":
                        # 智能对象：通过导出方式获取内容
                        img = self._export_smart_object(doc, layer)
                        if img:
                            layers.append(LayerInfo(layer.name, img))
                            logger.info(f"加载智能对象图层 '{layer.name}': {img.width}x{img.height}")
                        else:
                            skipped += 1
                        continue

                    # 普通图层：复制到新文档再导出
                    layer.copy()
                    temp_doc = self._ps_app.documents.add(
                        doc.width, doc.height, doc.resolution, "temp", 2  # RGB
                    )
                    self._ps_app.activeDocument = temp_doc
                    pasted = temp_doc.paste()
                    img = self._doc_to_pil(temp_doc)
                    layers.append(LayerInfo(layer.name, img))
                    temp_doc.close(False)
                except Exception as e:
                    logger.warning(f"加载图层 '{layer.name}' 失败: {e}")
                    skipped += 1

            # 总是导出PSD合成图（所有可见图层的最终渲染结果）
            try:
                composite = self._doc_to_pil(doc)
                if composite:
                    if composite.mode != "RGBA":
                        composite = composite.convert("RGBA")
                    layers.append(LayerInfo("__psd_composite__", composite))
                    logger.info(f"PSD合成图: {composite.width}x{composite.height}")
            except Exception as e:
                logger.warning(f"PSD合成图导出失败: {e}")

            logger.info(f"Photoshop COM加载完成: {len(layers)} 个图层 (含合成图)")
        finally:
            doc.close(False)
        return layers

    def _export_smart_object(self, doc, layer) -> Image.Image:
        """导出智能对象图层的内容

        方法：临时将该图层转为可编辑像素，导出后再还原。
        这确保智能对象中的矢量内容被正确栅格化。
        """
        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        try:
            # 仅显示当前图层，导出其内容
            self._ps_app.activeDocument = doc

            # 隐藏所有图层，只保留当前智能对象
            original_visibility = {}
            for l in doc.layers:
                original_visibility[l.name] = l.visible
                l.visible = False
            layer.visible = True

            # 导出当前文档（只包含当前图层内容）
            export_doc = self._ps_app.activeDocument
            opt = self._ps_app.PNGSaveOptions()
            export_doc.saveAs(tmp.name, opt, True)

            # 恢复所有图层可见性
            for l in doc.layers:
                if l.name in original_visibility:
                    l.visible = original_visibility[l.name]

            img = Image.open(tmp.name)
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            return img
        except Exception as e:
            logger.warning(f"智能对象导出失败: {e}")
            return None
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    def save_jpg(self, image: Image.Image, path: Path, quality: int = 95) -> None:
        # 使用 Pillow 保存即可，无需经过 PS
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(str(path), "JPEG", quality=quality, optimize=True)

    # ---------- 内部工具 ----------

    def _active_doc_to_pil(self) -> Image.Image:
        """将当前活动PS文档转为PIL Image"""
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        try:
            doc = self._ps_app.activeDocument
            opt = self._ps_app.PNGSaveOptions()
            doc.saveAs(tmp.name, opt, True)
            return Image.open(tmp.name).convert("RGB")
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    def _doc_to_pil(self, doc) -> Image.Image:
        """将指定PS文档转为PIL Image"""
        self._ps_app.activeDocument = doc
        return self._active_doc_to_pil()

    def cleanup(self) -> None:
        logger.info("PhotoshopEngine 资源清理完成")
