"""查询流水线：HybridSearch + Rerank，供 CLI 与 MCP Tool 复用。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from core.query_engine.dense_retriever import DenseRetriever
from core.query_engine.fusion import RRFFusion
from core.query_engine.hybrid_search import HybridSearch
from core.query_engine.query_processor import QueryProcessor
from core.query_engine.reranker import RerankResult, Reranker
from core.query_engine.sparse_retriever import SparseRetriever
from core.settings import Settings, VectorStoreSettings, resolve_path
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult
from ingestion.storage.bm25_indexer import BM25Indexer, BM25IndexerError

DEFAULT_TOP_K = 10


@dataclass(frozen=True)
class QueryPipelineResult:
    """一次查询流水线各阶段结果，供 CLI / MCP 输出复用。"""

    fusion_results: list[RetrievalResult]
    final_results: list[RetrievalResult]
    dense_results: list[RetrievalResult]
    sparse_results: list[RetrievalResult]
    rerank_result: RerankResult | None = None


def resolve_data_paths(data_root: str | Path | None) -> dict[str, Path]:
    """根据 data 根目录计算 Chroma/BM25 等子路径。"""
    base = resolve_path(data_root) if data_root is not None else resolve_path("data")
    db_root = base / "db"
    return {
        "chroma": db_root / "chroma",
        "bm25": db_root / "bm25",
    }


def settings_for_query(
    base_settings: Settings,
    collection: str | None,
    data_root: str | Path | None,
) -> tuple[Settings, Path]:
    """
    按 collection / data_root 构造查询用 Settings。

    collection 会覆盖 vector_store.collection_name（与 ingest 行为对齐）。
    data_root 会覆盖 Chroma/BM25 根目录。
    """
    data_paths = resolve_data_paths(data_root)
    vector_store = base_settings.vector_store
    collection_name = collection.strip() if collection else vector_store.collection_name
    persist_directory = (
        str(data_paths["chroma"]) if data_root is not None else vector_store.persist_directory
    )
    updated_vector_store = VectorStoreSettings(
        provider=vector_store.provider,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    updated_settings = Settings(
        llm=base_settings.llm,
        embedding=base_settings.embedding,
        vector_store=updated_vector_store,
        retrieval=base_settings.retrieval,
        rerank=base_settings.rerank,
        evaluation=base_settings.evaluation,
        observability=base_settings.observability,
        ingestion=base_settings.ingestion,
        vision_llm=base_settings.vision_llm,
    )
    return updated_settings, data_paths["bm25"]


def _build_pipeline_components(
    settings: Settings,
    bm25_root: Path,
) -> tuple[QueryProcessor, DenseRetriever, SparseRetriever, RRFFusion, Reranker]:
    """组装查询流水线组件（Dense/Sparse 共享 VectorStore）。"""
    from libs.vector_store.vector_store_factory import VectorStoreFactory

    vector_store = VectorStoreFactory.create(settings)
    indexer = BM25Indexer(
        collection=settings.vector_store.collection_name,
        index_root=bm25_root,
    )
    try:
        indexer.load()
    except BM25IndexerError:
        # BM25 索引尚未构建时仍允许仅 Dense 查询
        pass

    return (
        QueryProcessor(),
        DenseRetriever(settings, vector_store=vector_store),
        SparseRetriever(settings, bm25_indexer=indexer, vector_store=vector_store),
        RRFFusion(settings),
        Reranker(settings),
    )


def execute_query_pipeline(
    settings: Settings,
    query: str,
    top_k: int,
    filters: Mapping[str, Any] | None = None,
    *,
    no_rerank: bool = False,
    trace: TraceContext | None = None,
    query_processor: QueryProcessor | None = None,
    dense_retriever: DenseRetriever | None = None,
    sparse_retriever: SparseRetriever | None = None,
    fusion: RRFFusion | None = None,
    reranker: Reranker | None = None,
    bm25_root: Path | None = None,
) -> QueryPipelineResult:
    """
    执行完整查询流水线，返回各阶段结果。

    便于 CLI 与单元测试注入 mock 组件。
    """
    if query_processor is None or dense_retriever is None or sparse_retriever is None:
        root = bm25_root or resolve_data_paths(None)["bm25"]
        (
            query_processor,
            dense_retriever,
            sparse_retriever,
            fusion,
            reranker,
        ) = _build_pipeline_components(settings, root)
    else:
        fusion = fusion or RRFFusion(settings)
        reranker = reranker or Reranker(settings)

    processed = query_processor.process(query, filters=filters, trace=trace)
    merged_filters = processed.filters or None

    def _run_dense() -> list[RetrievalResult]:
        try:
            return dense_retriever.retrieve(
                processed.dense_query,
                settings.retrieval.dense_top_k,
                filters=merged_filters,
                trace=trace,
            )
        except Exception:
            return []

    def _run_sparse() -> list[RetrievalResult]:
        try:
            return sparse_retriever.retrieve(
                processed.keywords,
                settings.retrieval.sparse_top_k,
                trace=trace,
            )
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(_run_dense)
        sparse_future = executor.submit(_run_sparse)
        dense_results = dense_future.result()
        sparse_results = sparse_future.result()

    fusion_results = fusion.fuse(
        [dense_results, sparse_results],
        top_k=top_k,
        trace=trace,
    )
    filtered = HybridSearch._apply_metadata_filters(fusion_results, processed.filters)

    if no_rerank:
        return QueryPipelineResult(
            dense_results=dense_results,
            sparse_results=sparse_results,
            fusion_results=filtered,
            final_results=filtered[:top_k],
            rerank_result=None,
        )

    rerank_result = reranker.rerank(query, filtered, top_k=top_k, trace=trace)
    return QueryPipelineResult(
        dense_results=dense_results,
        sparse_results=sparse_results,
        fusion_results=filtered,
        final_results=rerank_result.results,
        rerank_result=rerank_result,
    )
