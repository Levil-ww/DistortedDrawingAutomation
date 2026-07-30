#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理层 - 负责配置的加载、保存、验证
与业务逻辑完全解耦
"""

import json
import os
from pathlib import Path
from typing import Optional

from models import ProcessConfig


class ConfigManager:
    """配置管理器"""

    DEFAULT_FILENAME = "config.json"

    def __init__(self, work_dir: Optional[Path] = None):
        self.work_dir = work_dir or Path(__file__).parent
        self._config_path: Optional[Path] = None

    def load(self, path: Optional[str] = None) -> ProcessConfig:
        """加载配置，支持指定路径或自动查找"""
        if path:
            self._config_path = Path(path)
        else:
            self._config_path = self._find_default_config()

        if self._config_path and self._config_path.exists():
            try:
                with open(self._config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return ProcessConfig.from_dict(data)
            except (json.JSONDecodeError, TypeError) as e:
                raise ValueError(f"配置文件格式错误: {e}")

        return ProcessConfig()

    def save(self, config: ProcessConfig, path: Optional[str] = None) -> None:
        """保存配置到JSON文件"""
        target = Path(path) if path else (self._config_path or self.work_dir / self.DEFAULT_FILENAME)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, ensure_ascii=False, indent=2)

    def _find_default_config(self) -> Optional[Path]:
        """按优先级查找默认配置文件"""
        candidates = [
            self.work_dir / self.DEFAULT_FILENAME,
            self.work_dir / "config_enhanced.json",
        ]
        for c in candidates:
            if c.exists():
                return c
        return None
