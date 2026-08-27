"""HybridSearch：编排 QueryProcessor、Dense/Sparse 召回、RRF 融合与 Metadata 过滤。

F3 在编排层写入 query_processing / dense_retrieval / sparse_retrieval 阶段；
fusion 阶段由 RRFFusion 通过同一 TraceContext 记录。
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Mapping, Sequence

from core.query_engine.dense_retriever import DenseRetriever, DenseRetrieverError
from core.query_engine.fusion import RRFFusion
from core.query_engine.query_processor import ProcessedQuery, QueryProcessor, QueryProcessorError
from core.query_engine.sparse_retriever import SparseRetriever, SparseRetrieverError
from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult


class HybridSearchError(Exception):
    """混合检索编排失败时抛出。"""


class HybridSearch:
    """
    混合检索编排器：并行 Dense + Sparse 召回，RRF 融合后做 Metadata 后置过滤。

    对应 spec D5：``query_processor.process() → 并行召回 → fusion.fuse() → metadata_filter``。
    F3 在编排层写入 ``query_processing`` / ``dense_retrieval`` / ``sparse_retrieval``。
    """

    def __init__(
        self,
        settings: Settings,
        query_processor: QueryProcessor | None = None,
        dense_retriever: DenseRetriever | None = None,
        sparse_retriever: SparseRetriever | None = None,
        fusion: RRFFusion | None = None,
    ) -> None:
        self._settings = settings
        self._query_processor = query_processor or QueryProcessor()
        self._dense_retriever = dense_retriever or DenseRetriever(settings)
        self._sparse_retriever = sparse_retriever or SparseRetriever(settings)
        self._fusion = fusion or RRFFusion(settings)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: Mapping[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[RetrievalResult]:
        """
        执行完整混合检索流程。

        Args:
            query: 用户自然语言查询。
            top_k: 最终返回条数；默认 ``settings.retrieval.fusion_top_k``。
            filters: 可选 metadata 过滤（与 query 内嵌约束合并）。
            trace: 可选 TraceContext。

        Returns:
            融合并过滤后的 RetrievalResult 列表。
        """
        limit = top_k if top_k is not None else self._settings.retrieval.fusion_top_k
        if limit <= 0:
            raise HybridSearchError("top_k 必须大于 0")

        start = time.perf_counter()
        # F3：编排层记录规范阶段名，保证 Fake 检索器也能产出完整 query trace
        qp_start = time.perf_counter()
        try:
            processed = self._query_processor.process(query, filters=filters, trace=trace)
        except QueryProcessorError as exc:
            raise HybridSearchError(f"查询预处理失败: {exc}") from exc

        if isinstance(trace, TraceContext):
            trace.record_stage(
                "query_processing",
                elapsed_ms=(time.perf_counter() - qp_start) * 1000,
                method="keyword",
                keyword_count=len(processed.keywords),
            )

        dense_results, sparse_results, dense_ms, sparse_ms = self._retrieve_parallel(
            processed, trace=trace
        )
        if isinstance(trace, TraceContext):
            # 并行结束后再写入，避免双线程同时 append stages
            trace.record_stage(
                "dense_retrieval",
                elapsed_ms=dense_ms,
                method="vector",
                provider=self._settings.embedding.provider,
                result_count=len(dense_results),
            )
            trace.record_stage(
                "sparse_retrieval",
                elapsed_ms=sparse_ms,
                method="bm25",
                result_count=len(sparse_results),
            )

        fused = self._fusion.fuse(
            [dense_results, sparse_results],
            top_k=limit,
            trace=trace,
        )
        filtered = self._apply_metadata_filters(fused, processed.filters)

        if isinstance(trace, TraceContext):
            trace.record_stage(
                "hybrid_search",
                elapsed_ms=(time.perf_counter() - start) * 1000,
                method="hybrid",
                top_k=limit,
                dense_count=len(dense_results),
                sparse_count=len(sparse_results),
                result_count=len(filtered[:limit]),
            )

        return filtered[:limit]

    def _retrieve_parallel(
        self,
        processed: ProcessedQuery,
        trace: Any | None = None,
    ) -> tuple[list[RetrievalResult], list[RetrievalResult], float, float]:
        """并行执行 Dense/Sparse 召回；任一路失败时降级为空列表，并返回各路耗时。"""
        dense_top_k = self._settings.retrieval.dense_top_k
        sparse_top_k = self._settings.retrieval.sparse_top_k
        merged_filters = processed.filters

        def _run_dense() -> tuple[list[RetrievalResult], float]:
            start = time.perf_counter()
            try:
                results = self._dense_retriever.retrieve(
                    processed.dense_query,
                    dense_top_k,
                    filters=merged_filters or None,
                    trace=trace,
                )
            except DenseRetrieverError:
                results = []
            return results, (time.perf_counter() - start) * 1000

        def _run_sparse() -> tuple[list[RetrievalResult], float]:
            start = time.perf_counter()
            try:
                results = self._sparse_retriever.retrieve(
                    processed.keywords,
                    sparse_top_k,
                    trace=trace,
                )
            except SparseRetrieverError:
                results = []
            return results, (time.perf_counter() - start) * 1000

        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(_run_dense)
            sparse_future = executor.submit(_run_sparse)
            dense_results, dense_ms = dense_future.result()
            sparse_results, sparse_ms = sparse_future.result()
            return dense_results, sparse_results, dense_ms, sparse_ms

    @staticmethod
    def _apply_metadata_filters(
        candidates: Sequence[RetrievalResult],
        filters: Mapping[str, Any] | None,
    ) -> list[RetrievalResult]:
        """
        后置 metadata 过滤兜底：缺失字段采用 missing->include，避免误杀召回。
        """
        if not filters:
            return list(candidates)

        filtered: list[RetrievalResult] = []
        for item in candidates:
            metadata = item.metadata
            if all(
                key not in metadata or metadata[key] == value
                for key, value in filters.items()
            ):
                filtered.append(item)
        return filtered
