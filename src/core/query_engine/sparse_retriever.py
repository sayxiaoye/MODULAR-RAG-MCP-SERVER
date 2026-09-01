"""SparseRetriever：BM25 关键词召回 + VectorStore 正文回填编排。"""

from __future__ import annotations

import time
from typing import Any, Sequence

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult
from ingestion.storage.bm25_indexer import BM25Indexer, BM25IndexerError
from libs.vector_store.base_vector_store import BaseVectorStore, VectorStoreError
from libs.vector_store.vector_store_factory import VectorStoreFactory


class SparseRetrieverError(Exception):
    """稀疏检索编排失败时抛出。"""


class SparseRetriever:
    """
    关键词召回器：BM25 倒排索引打分后，经 VectorStore 批量取回 text/metadata。

    keywords 通常来自 ``QueryProcessor.process()`` 的 ``ProcessedQuery.keywords``。
    """

    def __init__(
        self,
        settings: Settings,
        bm25_indexer: BM25Indexer | None = None,
        vector_store: BaseVectorStore | None = None,
    ) -> None:
        self._settings = settings
        if bm25_indexer is not None:
            self._bm25_indexer = bm25_indexer
        else:
            # 与摄取管线一致：按 vector_store.collection_name 定位 BM25 索引文件
            indexer = BM25Indexer(collection=settings.vector_store.collection_name)
            indexer.load()
            self._bm25_indexer = indexer
        self._vector_store = vector_store or VectorStoreFactory.create(settings)

    def retrieve(
        self,
        keywords: Sequence[str],
        top_k: int,
        trace: Any | None = None,
    ) -> list[RetrievalResult]:
        """
        对关键词列表执行 BM25 检索并组装 RetrievalResult。

        Args:
            keywords: 稀疏检索词项（去停用词后）。
            top_k: 返回条数上限。
            trace: 可选 TraceContext。

        Returns:
            RetrievalResult 列表，按 BM25 分数降序。
        """
        normalized_keywords = [kw.strip() for kw in keywords if isinstance(kw, str) and kw.strip()]
        if not normalized_keywords:
            raise SparseRetrieverError("keywords 不能为空")
        if top_k <= 0:
            raise SparseRetrieverError("top_k 必须大于 0")

        query_text = " ".join(normalized_keywords)
        start = time.perf_counter()
        try:
            ranked = self._bm25_indexer.query(query_text, top_k)
        except BM25IndexerError as exc:
            raise SparseRetrieverError(f"BM25 查询失败: {exc}") from exc

        bm25_ms = (time.perf_counter() - start) * 1000
        if not ranked:
            if isinstance(trace, TraceContext):
                trace.record_stage(
                    "sparse_retriever",
                    elapsed_ms=bm25_ms,
                    bm25_ms=round(bm25_ms, 3),
                    top_k=top_k,
                    result_count=0,
                )
            return []

        chunk_ids = [chunk_id for chunk_id, _ in ranked]
        score_by_id = {chunk_id: float(score) for chunk_id, score in ranked}

        start = time.perf_counter()
        try:
            records = self._vector_store.get_by_ids(
                chunk_ids,
                trace=trace,
                collection=self._settings.vector_store.collection_name,
            )
        except VectorStoreError as exc:
            raise SparseRetrieverError(f"向量库批量读取失败: {exc}") from exc

        lookup_ms = (time.perf_counter() - start) * 1000
        record_by_id = {str(item["id"]): item for item in records}

        # 保持 BM25 排名顺序，跳过向量库中缺失的 chunk
        results: list[RetrievalResult] = []
        for chunk_id in chunk_ids:
            record = record_by_id.get(chunk_id)
            if record is None:
                continue
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=score_by_id[chunk_id],
                    text=str(record.get("text", "")),
                    metadata=dict(record.get("metadata", {})),
                )
            )

        if isinstance(trace, TraceContext):
            trace.record_stage(
                "sparse_retriever",
                elapsed_ms=bm25_ms + lookup_ms,
                bm25_ms=round(bm25_ms, 3),
                lookup_ms=round(lookup_ms, 3),
                top_k=top_k,
                result_count=len(results),
            )

        return results
