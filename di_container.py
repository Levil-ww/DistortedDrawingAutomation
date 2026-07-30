#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依赖注入容器

集中管理所有组件的创建和生命周期，消除各层之间的直接耦合。

使用方式：
    container = DIContainer()
    container.register_engine("auto")
    service = container.create_service()

依赖方向：di_container 依赖所有下层模块，但没有任何模块依赖它
（除了入口文件 main.py 和 gui/app.py）
"""

import logging
from typing import Optional

from models import ProcessConfig
from engines import ImageEngine, create_engine
from processors.aligner import SmartAligner
from processors.color_adjuster import ColorAdjuster
from processors.compositor import ImageCompositor
from services.design_service import DesignService
from config_manager import ConfigManager

logger = logging.getLogger(__name__)


class DIContainer:
    """
    依赖注入容器
    负责：
    1. 引擎的创建和生命周期管理
    2. 处理器的创建
    3. 服务层的组装
    4. 配置的加载
    """

    def __init__(self):
        self._engine: Optional[ImageEngine] = None
        self._config_manager = ConfigManager()

    # ---------- 引擎管理 ----------

    def register_engine(self, preferred: str = "auto") -> ImageEngine:
        """
        注册并创建引擎
        Args:
            preferred: "auto" | "photoshop" | "pillow"
        Returns:
            创建的引擎实例
        """
        self._engine = create_engine(preferred)
        logger.info(f"DI容器：注册引擎 {self._engine.capabilities.name}")
        return self._engine

    @property
    def engine(self) -> ImageEngine:
        if self._engine is None:
            raise RuntimeError("引擎未注册，请先调用 register_engine()")
        return self._engine

    # ---------- 处理器创建 ----------

    def create_aligner(self, margin_percent: float = 2.0) -> SmartAligner:
        return SmartAligner(margin_percent=margin_percent)

    def create_color_adjuster(self) -> ColorAdjuster:
        return ColorAdjuster()

    def create_compositor(self) -> ImageCompositor:
        return ImageCompositor()

    # ---------- 服务组装 ----------

    def create_service(
        self,
        aligner: Optional[SmartAligner] = None,
        color_adjuster: Optional[ColorAdjuster] = None,
        compositor: Optional[ImageCompositor] = None,
    ) -> DesignService:
        """
        创建完整配置的设计服务
        所有未提供的处理器将自动创建
        """
        return DesignService(
            engine=self.engine,
            aligner=aligner or self.create_aligner(),
            color_adjuster=color_adjuster or self.create_color_adjuster(),
            compositor=compositor or self.create_compositor(),
        )

    # ---------- 配置管理 ----------

    def load_config(self, path: Optional[str] = None) -> ProcessConfig:
        return self._config_manager.load(path)

    def save_config(self, config: ProcessConfig, path: Optional[str] = None) -> None:
        self._config_manager.save(config, path)

    # ---------- 资源清理 ----------

    def cleanup(self) -> None:
        if self._engine:
            self._engine.cleanup()
            self._engine = None
            logger.info("DI容器：引擎已清理")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False
