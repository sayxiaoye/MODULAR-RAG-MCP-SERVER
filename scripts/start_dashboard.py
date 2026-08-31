#!/usr/bin/env python3
"""Dashboard 启动脚本：以 Streamlit 运行多页面管理平台。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
_APP = _SRC_ROOT / "observability" / "dashboard" / "app.py"


def main() -> int:
    """启动 Streamlit Dashboard，并把 src/ 加入 PYTHONPATH。"""
    if not _APP.is_file():
        print(f"找不到 Dashboard 入口: {_APP}", file=sys.stderr)
        return 1
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print(
            "未安装 streamlit。请执行: python -m pip install '.[dashboard]'",
            file=sys.stderr,
        )
        return 1

    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(_SRC_ROOT) if not existing else str(_SRC_ROOT) + os.pathsep + existing
    )
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(_APP),
        "--server.headless",
        "true",
    ]
    return subprocess.call(command, env=env, cwd=str(_REPO_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
