#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
设计自动化工具 v2.0

分层架构：
    models       - 数据模型（最底层，无依赖）
    processors   - 图像算法（依赖 models）
    engines      - 图像引擎（依赖 models）
    services     - 业务编排（依赖 engines + processors + models）
    gui          - 用户界面（依赖 services + models）
    di_container - 依赖注入容器（组装所有层）

依赖规则：
    - 上层可以导入下层
    - 下层严禁导入上层
    - models 不依赖任何内部模块
"""

__version__ = "2.0.0"
