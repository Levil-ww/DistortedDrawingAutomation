#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI 入口

命令行调用示例:
    python main.py --config config.json
    python main.py --engine pillow --eps template.eps --psd material.psd

设计：入口文件只负责解析参数、初始化容器、调用服务，
不包含任何业务逻辑。
"""

import argparse
import logging
import sys
from pathlib import Path

from di_container import DIContainer


def setup_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    parser = argparse.ArgumentParser(description="设计自动化工具 v2.0")
    parser.add_argument("--config", "-c", help="配置文件路径")
    parser.add_argument("--engine", "-e", choices=["auto", "photoshop", "pillow"], default="auto", help="引擎选择")
    parser.add_argument("--eps", help="EPS模板文件路径")
    parser.add_argument("--psd", help="PSD素材文件路径")
    parser.add_argument("--out", help="输出文件路径")
    args = parser.parse_args()

    setup_logging()

    # 使用 DI 容器管理所有依赖
    with DIContainer() as container:
        # 加载配置
        config = container.load_config(args.config)

        # 命令行参数覆盖配置文件
        if args.eps:
            config.eps_file = args.eps
        if args.psd:
            config.psd_file = args.psd
        if args.out:
            config.output_file = args.out

        # 注册引擎并创建服务
        engine = container.register_engine(preferred=args.engine)
        print(f"使用引擎: {engine.capabilities.name}")

        service = container.create_service()

        # 执行处理
        try:
            out_path = service.process(config)
            print(f"\n成功: {out_path}")
        except Exception as e:
            print(f"\n失败: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
