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

    def load_psd_layers(self, path: Path) -> List[LayerInfo]:
        doc = self._ps_app.Open(str(path))
        layers: List[LayerInfo] = []
        try:
            for layer in doc.layers:
                if not layer.visible or layer.typename != "ArtLayer":
                    continue
                self._ps_app.activeDocument = doc
                doc.activeLayer = layer
                layer.copy()
                # 粘贴到新文档再导出为PIL
                temp_doc = self._ps_app.documents.add(
                    doc.width, doc.height, doc.resolution, "temp", 2  # RGB
                )
                self._ps_app.activeDocument = temp_doc
                pasted = temp_doc.paste()
                img = self._doc_to_pil(temp_doc)
                layers.append(LayerInfo(layer.name, img))
                temp_doc.close(False)
        finally:
            doc.close(False)
        return layers

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
