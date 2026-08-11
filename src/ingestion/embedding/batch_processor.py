"""BatchProcessor：按 batch_size 分批驱动 Dense/Sparse 编码。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.embedding.dense_encoder import DenseEncoder
from ingestion.embedding.sparse_encoder import SparseChunkStats, SparseEncoder


@dataclass(frozen=True)
class BatchEncodingResult:
    """批量编码汇总结果，向量与稀疏统计顺序与输入 chunks 一致。"""

    dense_vectors: list[list[float]]
    sparse_stats: list[SparseChunkStats]
    batch_count: int


class BatchProcessorError(Exception):
    """批处理编排失败时抛出。"""


class BatchProcessor:
    """将 Chunk 列表按配置分批，驱动 DenseEncoder 与 SparseEncoder。"""

    def __init__(
        self,
        settings: Settings,
        dense_encoder: DenseEncoder | None = None,
        sparse_encoder: SparseEncoder | None = None,
        batch_size: int | None = None,
    ) -> None:
        self._settings = settings
        self._dense_encoder = dense_encoder or DenseEncoder(settings)
        self._sparse_encoder = sparse_encoder or SparseEncoder()
        ingestion = settings.ingestion
        if ingestion is None:
            raise BatchProcessorError("缺少 ingestion 配置，无法确定 batch_size")
        configured = batch_size if batch_size is not None else ingestion.batch_size
        if configured <= 0:
            raise BatchProcessorError("batch_size 必须大于 0")
        self._batch_size = configured

    def process(
        self,
        chunks: Sequence[Chunk],
        trace: Any | None = None,
    ) -> BatchEncodingResult:
        """
        分批编码全部 Chunk，并合并为与输入等长的结果列表。

        Args:
            chunks: 待编码 Chunk 序列。
            trace: 可选 TraceContext，记录每批耗时。

        Returns:
            Dense 向量与 Sparse 统计，顺序与 chunks 一致。
        """
        if not chunks:
            return BatchEncodingResult(dense_vectors=[], sparse_stats=[], batch_count=0)

        dense_vectors: list[list[float]] = []
        sparse_stats: list[SparseChunkStats] = []
        batch_count = 0

        for batch_index, batch in enumerate(self._iter_batches(chunks)):
            batch_count += 1
            start = time.perf_counter()
            dense_vectors.extend(self._dense_encoder.encode(batch, trace=trace))
            sparse_stats.extend(self._sparse_encoder.encode(batch))
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._record_trace(
                trace,
                "embedding_batch",
                elapsed_ms=elapsed_ms,
                batch_index=batch_index,
                batch_size=len(batch),
            )

        if len(dense_vectors) != len(chunks) or len(sparse_stats) != len(chunks):
            raise BatchProcessorError("批处理编码结果数量与 Chunk 数量不一致")

        return BatchEncodingResult(
            dense_vectors=dense_vectors,
            sparse_stats=sparse_stats,
            batch_count=batch_count,
        )

    def batch_sizes_for_count(self, count: int) -> list[int]:
        """根据总数量计算各批大小（用于验收 batch 切分逻辑）。"""
        if count <= 0:
            return []
        sizes: list[int] = []
        remaining = count
        while remaining > 0:
            size = min(self._batch_size, remaining)
            sizes.append(size)
            remaining -= size
        return sizes

    def _iter_batches(self, chunks: Sequence[Chunk]) -> Iterator[Sequence[Chunk]]:
        """按 batch_size 切分 chunks，保持原始顺序。"""
        for start in range(0, len(chunks), self._batch_size):
            yield chunks[start : start + self._batch_size]

    @staticmethod
    def _record_trace(
        trace: Any | None,
        name: str,
        *,
        elapsed_ms: float,
        **details: Any,
    ) -> None:
        if isinstance(trace, TraceContext):
            trace.record_stage(name, elapsed_ms=elapsed_ms, **details)
