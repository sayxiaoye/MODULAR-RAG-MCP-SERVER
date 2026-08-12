"""Transform 抽象层：Chunk 级增强与去噪的统一接口。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from core.types import Chunk


class TransformError(Exception):
    """Transform 处理失败时抛出。"""


class BaseTransform(ABC):
    """Transform 抽象基类：对 Chunk 列表进行批量增强。"""

    @abstractmethod
    def transform(
        self,
        chunks: Sequence[Chunk],
        trace: Any | None = None,
    ) -> list[Chunk]:
        """
        对 Chunk 列表执行变换并返回新列表。

        Args:
            chunks: DocumentChunker 产出的 Chunk 序列。
            trace: 可选 TraceContext，用于记录阶段耗时。

        Returns:
            变换后的 Chunk 列表，顺序与输入一致。
        """
        ...
