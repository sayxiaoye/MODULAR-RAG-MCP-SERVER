"""结构化日志：stderr 文本日志 + JSON Lines trace 持久化。"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from core.settings import SettingsError, load_settings, resolve_path

# 默认 logger 名称，与 observability 层职责一致
_LOGGER_NAME = "modular_rag"
_TRACE_LOGGER_NAME = "modular_rag.trace"
_DEFAULT_TRACE_FILE = "./logs/traces.jsonl"


class JSONFormatter(logging.Formatter):
    """将 LogRecord 格式化为单行 JSON，供 JSON Lines 文件输出。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # 允许 extra 注入 trace_type 等字段，便于与 Trace 记录对齐
        extra_type = getattr(record, "trace_type", None)
        if extra_type is not None:
            payload["trace_type"] = extra_type
        return json.dumps(payload, ensure_ascii=False, default=str)


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


def resolve_trace_file(path: str | Path | None = None) -> Path:
    """解析 traces.jsonl 路径：显式参数优先，否则读 settings.observability.trace_file。"""
    if path is not None:
        candidate = Path(path)
        return candidate if candidate.is_absolute() else resolve_path(candidate)
    try:
        settings = load_settings()
        return resolve_path(settings.observability.trace_file)
    except (SettingsError, OSError, TypeError):
        return resolve_path(_DEFAULT_TRACE_FILE)


def get_trace_logger(trace_file: str | Path | None = None) -> logging.Logger:
    """
    获取写入 JSON Lines 的 trace logger（FileHandler + JSONFormatter）。

    不向 stdout/stderr 传播，避免污染 MCP stdio。
    """
    logger = logging.getLogger(_TRACE_LOGGER_NAME)
    if not logger.handlers:
        file_path = resolve_trace_file(trace_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(file_path, encoding="utf-8")
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def write_trace(trace_dict: Mapping[str, Any], path: str | Path | None = None) -> None:
    """
    将一条 trace 字典追加写入 traces.jsonl（一行一个 JSON 对象）。

    Args:
        trace_dict: 通常来自 TraceContext.to_dict()。
        path: 可选覆盖路径，测试时可指向临时文件。
    """
    payload = dict(trace_dict)
    file_path = resolve_trace_file(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, default=str)
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def reset_trace_logger() -> None:
    """关闭并移除 trace logger handlers，供测试隔离文件路径。"""
    logger = logging.getLogger(_TRACE_LOGGER_NAME)
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
