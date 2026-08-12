"""结构化日志占位模块：启动阶段向 stderr 输出，避免污染 MCP stdio。"""

from __future__ import annotations

import logging
import sys
from typing import Optional

# 默认 logger 名称，与 observability 层职责一致
_LOGGER_NAME = "modular_rag"


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """获取配置好的根 logger，输出到 stderr（MCP Server 要求 stdout 仅用于协议消息）。"""
    logger_name = name or _LOGGER_NAME
    logger = logging.getLogger(logger_name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
