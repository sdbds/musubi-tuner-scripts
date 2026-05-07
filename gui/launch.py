#!/usr/bin/env python3
"""Command-line launcher for the Musubi Tuner GUI."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


project_root = Path(__file__).parent.parent.resolve()
gui_root = project_root / "gui"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(gui_root) not in sys.path:
    sys.path.insert(0, str(gui_root))

os.chdir(project_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动 Musubi Tuner GUI")
    parser.add_argument("--host", type=str, default=None, help="绑定地址 (默认: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="端口")
    parser.add_argument("--cloud", action="store_true", help="云模式，绑定 0.0.0.0")
    parser.add_argument("--native", action="store_true", help="原生窗口模式")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器/窗口")
    return parser.parse_args()


def apply_env(args: argparse.Namespace) -> None:
    if args.cloud:
        os.environ["MUSUBI_GUI_HOST"] = "0.0.0.0"
    elif args.host:
        os.environ["MUSUBI_GUI_HOST"] = args.host

    if args.port is not None:
        os.environ["MUSUBI_GUI_PORT"] = str(args.port)

    if args.native:
        os.environ["MUSUBI_GUI_NATIVE"] = "1"

    if args.no_browser:
        os.environ["MUSUBI_GUI_NO_BROWSER"] = "1"


if __name__ == "__main__":
    options = parse_args()
    apply_env(options)

    from main import main

    main()
