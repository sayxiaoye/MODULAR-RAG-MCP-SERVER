"""Splitter 抽象层：定义文本切分接口，供 DocumentChunker 与摄取链路复用。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class SplitterError(Exception):
    """切分输入校验或策略执行失败时抛出。"""


class BaseSplitter(ABC):
    """Splitter 抽象基类：将长文本切分为若干语义片段（str 列表）。"""

    @abstractmethod
    def split_text(self, text: str, trace: Any | None = None) -> list[str]:
        """
        将输入文本切分为多个 chunk 文本片段。

        Args:
            text: 待切分的原始文本（通常为 Markdown 文档内容）。
            trace: 可选追踪上下文（F 阶段注入，用于记录策略与耗时）。

        Returns:
            切分后的文本片段列表，顺序与原文位置一致。
        """

    def _validate_text(self, text: str) -> str:
        """校验输入为合法非空字符串。"""
        if not isinstance(text, str) or not text.strip():
            raise SplitterError("text 必须是非空字符串")
        return text
