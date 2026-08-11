"""DenseEncoder：将 Chunk 文本批量编码为稠密向量。"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from core.settings import Settings
from core.types import Chunk
from libs.embedding.base_embedding import BaseEmbedding, EmbeddingError
from libs.embedding.embedding_factory import EmbeddingFactory

logger = logging.getLogger(__name__)


class DenseEncoderError(Exception):
    """稠密向量编码失败或输出契约不满足时抛出。"""


class DenseEncoder:
    """摄取链路 Dense 编码器：委托 libs.embedding 完成批量向量化。"""

    def __init__(
        self,
        settings: Settings,
        embedding: BaseEmbedding | None = None,
    ) -> None:
        self._settings = settings
        # 测试可注入 FakeEmbedding，生产环境通过 EmbeddingFactory 创建
        self._embedding = embedding or EmbeddingFactory.create(settings)

    def encode(
        self,
        chunks: Sequence[Chunk],
        trace: Any | None = None,
    ) -> list[list[float]]:
        """
        将 Chunk 文本批量编码为稠密向量。

        Args:
            chunks: 待编码的 Chunk 序列，顺序与返回向量一一对应。
            trace: 可选 TraceContext，供 Embedding 记录 provider/耗时。

        Returns:
            与 chunks 等长的向量列表，且每条向量维度一致。

        Raises:
            DenseEncoderError: Embedding 调用失败或返回数量/维度不一致。
        """
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]
        try:
            vectors = self._embedding.embed(texts, trace=trace)
        except EmbeddingError as exc:
            raise DenseEncoderError(f"Dense 编码失败: {exc}") from exc

        if len(vectors) != len(chunks):
            raise DenseEncoderError(
                f"向量数量({len(vectors)})与 Chunk 数量({len(chunks)})不一致"
            )

        expected_dim = self._settings.embedding.dimensions
        for index, vector in enumerate(vectors):
            if not isinstance(vector, list) or not vector:
                raise DenseEncoderError(f"vectors[{index}] 必须是非空 float 列表")
            if len(vector) != expected_dim:
                raise DenseEncoderError(
                    f"vectors[{index}] 维度 {len(vector)} 与配置 dimensions {expected_dim} 不一致"
                )

        return vectors
