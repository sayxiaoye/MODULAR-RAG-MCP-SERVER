"""DenseRetriever：query 向量化 + VectorStore 语义召回编排。"""

from __future__ import annotations

import time
from typing import Any, Mapping

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult
from libs.embedding.base_embedding import BaseEmbedding, EmbeddingError
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.vector_store.base_vector_store import BaseVectorStore, VectorStoreError
from libs.vector_store.vector_store_factory import VectorStoreFactory


class DenseRetrieverError(Exception):
    """稠密检索编排失败时抛出。"""


class DenseRetriever:
    """
    语义召回器：组合 Embedding（query 向量化）与 VectorStore.query。

    规范中的 EmbeddingClient 在本项目对应 ``libs.embedding.BaseEmbedding``。
    """

    def __init__(
        self,
        settings: Settings,
        embedding: BaseEmbedding | None = None,
        vector_store: BaseVectorStore | None = None,
    ) -> None:
        self._settings = settings
        self._embedding = embedding or EmbeddingFactory.create(settings)
        self._vector_store = vector_store or VectorStoreFactory.create(settings)

    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[RetrievalResult]:
        """
        对 query 执行稠密向量检索。

        Args:
            query: 自然语言查询（通常为 ProcessedQuery.dense_query）。
            top_k: 返回条数上限。
            filters: 可选 metadata 硬过滤（如 collection）。
            trace: 可选 TraceContext。

        Returns:
            RetrievalResult 列表，按相似度降序。
        """
        if not isinstance(query, str) or not query.strip():
            raise DenseRetrieverError("query 必须是非空字符串")
        if top_k <= 0:
            raise DenseRetrieverError("top_k 必须大于 0")

        start = time.perf_counter()
        try:
            vectors = self._embedding.embed([query.strip()], trace=trace)
        except EmbeddingError as exc:
            raise DenseRetrieverError(f"query 向量化失败: {exc}") from exc

        if not vectors:
            raise DenseRetrieverError("Embedding 未返回 query 向量")

        query_vector = vectors[0]
        embed_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        try:
            raw_results = self._vector_store.query(
                query_vector,
                top_k,
                filters=filters,
                trace=trace,
            )
        except VectorStoreError as exc:
            raise DenseRetrieverError(f"向量库检索失败: {exc}") from exc

        query_ms = (time.perf_counter() - start) * 1000

        if isinstance(trace, TraceContext):
            trace.record_stage(
                "dense_retriever",
                elapsed_ms=embed_ms + query_ms,
                embed_ms=round(embed_ms, 3),
                query_ms=round(query_ms, 3),
                top_k=top_k,
                result_count=len(raw_results),
            )

        return [_normalize_result(item) for item in raw_results]


def _normalize_result(item: Mapping[str, Any]) -> RetrievalResult:
    """将 VectorStore 契约结果规范化为 RetrievalResult。"""
    try:
        return RetrievalResult.from_dict(item)
    except Exception as exc:
        raise DenseRetrieverError(f"检索结果契约非法: {exc}") from exc
