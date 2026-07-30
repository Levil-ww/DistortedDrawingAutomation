#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
引擎抽象基类 - 定义所有引擎必须实现的接口

这是策略模式的核心，新增引擎只需继承此类并实现方法。

重要：本模块只依赖 models.py，不依赖 processors 或 services，
确保引擎层不会反向依赖上层模块。
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from PIL import Image

from models import EngineCapabilities, LayerInfo


class ImageEngine(ABC):
    """
    图像处理引擎抽象基类
    所有具体引擎（Photoshop/Pillow/未来GIMP等）必须实现这些方法
    """

    @property
    @abstractmethod
    def capabilities(self) -> EngineCapabilities:
        """返回引擎能力描述"""
        ...

    @abstractmethod
    def open_eps(self, path: Path, dpi: int = 300,
                 width_cm: float = None, height_cm: float = None) -> Image.Image:
        """
        打开并栅格化EPS文件，返回PIL Image
        所有引擎统一返回PIL Image，上层无需关心底层实现

        Args:
            path: EPS文件路径
            dpi: 栅格化分辨率
            width_cm: 目标宽度（厘米），用于覆盖EPS原始尺寸
            height_cm: 目标高度（厘米），用于覆盖EPS原始尺寸
        """
        ...

    @abstractmethod
    def load_psd_layers(self, path: Path) -> List[LayerInfo]:
        """
        加载PSD文件的所有可见图层
        返回统一的 LayerInfo 列表
        """
        ...

    @abstractmethod
    def save_jpg(self, image: Image.Image, path: Path, quality: int = 95) -> None:
        """保存图像为JPG格式"""
        ...

    def cleanup(self) -> None:
        """清理资源（可选重写）"""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
