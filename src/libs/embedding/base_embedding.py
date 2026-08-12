"""Embedding 抽象层：定义批量向量化接口，供 Dense 检索与摄取链路复用。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence


class EmbeddingError(Exception):
    """Embedding 调用或输入校验失败时抛出。"""


class BaseEmbedding(ABC):
    """Embedding 抽象基类：屏蔽各 Provider 的批量请求格式差异。"""

    @abstractmethod
    def embed(
        self,
        texts: Sequence[str],
        trace: Any | None = None,
    ) -> list[list[float]]:
        """
        将文本批量编码为稠密向量。

        Args:
            texts: 待编码文本列表，顺序与返回向量一一对应。
            trace: 可选追踪上下文（F 阶段注入，用于记录 provider/耗时）。

        Returns:
            与 texts 等长的向量列表，每个元素为 float 维度向量。
        """

    def _validate_texts(self, texts: Sequence[str]) -> list[str]:
        """校验输入非空且每条文本为合法字符串。"""
        if not texts:
            raise EmbeddingError("texts 不能为空")
        normalized: list[str] = []
        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise EmbeddingError(f"texts[{index}] 必须是非空字符串")
            normalized.append(text)
        return normalized
