#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据模型层 - 仅定义数据结构，无任何业务逻辑

本模块是项目中最底层的依赖，其他所有模块都可以安全导入它
而不会产生循环依赖。
"""

from dataclasses import dataclass, asdict, fields
from typing import Dict, Any

from PIL import Image


@dataclass
class EngineCapabilities:
    """引擎能力声明"""
    name: str
    supports_photoshop_native: bool = False
    supports_clipping_mask: bool = False
    supports_smart_align: bool = True
    supports_color_adjust: bool = True
    requires_external_app: bool = False


@dataclass
class LayerInfo:
    """统一的图层信息描述"""
    name: str
    image: Image.Image


@dataclass
class ProcessConfig:
    """处理配置数据类 - 所有默认值集中在此定义"""
    # 文件路径
    eps_file: str = ""
    psd_file: str = ""
    output_file: str = "output.jpg"

    # 基础参数
    dpi: int = 300
    canvas_width_cm: float = 80.0
    canvas_height_cm: float = 150.0

    # 手动变换
    pattern_scale: float = 1.0
    pattern_offset_x: int = 0
    pattern_offset_y: int = 0
    fill_mode: str = "clip"

    # 智能对齐
    smart_align: bool = True
    auto_scale: bool = True
    margin_percent: float = 2.0

    # 色彩调整
    brightness: int = 0          # -100 ~ 100
    contrast: int = 0            # -100 ~ 100
    saturation: int = 0          # -100 ~ 100
    hue_shift: int = 0           # -180 ~ 180
    warmth: int = 0              # -100 ~ 100

    # 输出质量
    jpg_quality: int = 95        # 1-100

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessConfig":
        """安全地从字典创建，忽略未知字段"""
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def clone(self) -> "ProcessConfig":
        """创建深拷贝"""
        return ProcessConfig.from_dict(self.to_dict())
