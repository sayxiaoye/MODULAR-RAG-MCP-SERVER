#!/usr/bin/env python3
"""在线查询 CLI：调用 HybridSearch + Reranker 并格式化输出检索结果。"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, TextIO

# 支持直接执行 scripts/query.py 时能找到 src 包
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from core.query_engine.dense_retriever import DenseRetriever
from core.query_engine.fusion import RRFFusion
from core.query_engine.hybrid_search import HybridSearch, HybridSearchError
from core.query_engine.query_processor import QueryProcessor, QueryProcessorError
from core.query_engine.reranker import RerankResult, Reranker
from core.query_engine.sparse_retriever import SparseRetriever
from core.settings import Settings, SettingsError, VectorStoreSettings, load_settings, resolve_path
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult
from ingestion.storage.bm25_indexer import BM25Indexer, BM25IndexerError
from observability.logger import get_logger

DEFAULT_TOP_K = 10
_TEXT_PREVIEW_LEN = 120
_EMPTY_DATA_HINT = "未找到相关文档，请先运行 ingest.py 摄取数据"


@dataclass(frozen=True)
class QueryPipelineResult:
    """一次查询流水线各阶段结果，供默认/verbose 输出复用。"""

    fusion_results: list[RetrievalResult]
    final_results: list[RetrievalResult]
    dense_results: list[RetrievalResult]
    sparse_results: list[RetrievalResult]
    rerank_result: RerankResult | None = None


def build_arg_parser() -> argparse.ArgumentParser:
    """构建 query 命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="Modular RAG MCP Server — 在线混合检索查询入口",
    )
    parser.add_argument(
        "--query",
        required=True,
        help="查询文本（必填）",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"返回结果数量（默认 {DEFAULT_TOP_K}）",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="限定检索集合（覆盖 vector_store.collection_name）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示 Dense/Sparse/Fusion/Rerank 各阶段中间结果",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="跳过 Reranker 阶段，直接返回融合结果",
    )
    parser.add_argument(
        "--settings",
        default=None,
        help="可选 settings.yaml 路径，默认使用 config/settings.yaml",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="覆盖 data 根目录（默认 data/），测试时可指向临时目录",
    )
    return parser


def _resolve_data_paths(data_root: str | Path | None) -> dict[str, Path]:
    """根据 data 根目录计算 Chroma/BM25 等子路径。"""
    base = resolve_path(data_root) if data_root is not None else resolve_path("data")
    db_root = base / "db"
    return {
        "chroma": db_root / "chroma",
        "bm25": db_root / "bm25",
    }


def _settings_for_query(
    base_settings: Settings,
    collection: str | None,
    data_root: str | Path | None,
) -> tuple[Settings, Path]:
    """
    按 CLI 参数构造查询用 Settings。

    --collection 会覆盖 vector_store.collection_name（与 ingest 行为对齐）。
    --data-root 会覆盖 Chroma/BM25 根目录。
    """
    data_paths = _resolve_data_paths(data_root)
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
        root = bm25_root or _resolve_data_paths(None)["bm25"]
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


def _preview_text(text: str, limit: int = _TEXT_PREVIEW_LEN) -> str:
    """生成单行文本摘要，避免终端输出过长。"""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[: limit - 3]}..."


def _source_label(metadata: Mapping[str, Any]) -> str:
    """从 metadata 提取来源文件标识。"""
    for key in ("source_path", "source_file", "file_name"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _format_result_block(title: str, results: list[RetrievalResult], out: TextIO) -> None:
    """打印单个阶段的检索结果块。"""
    print(title, file=out)
    if not results:
        print("  (empty)", file=out)
        return
    for index, item in enumerate(results, start=1):
        page = item.metadata.get("page")
        page_suffix = f" | page={page}" if page is not None else ""
        print(
            f"  {index}. score={item.score:.4f} | id={item.chunk_id} | "
            f"source={_source_label(item.metadata)}{page_suffix}",
            file=out,
        )
        print(f"     {_preview_text(item.text)}", file=out)


def render_query_output(
    pipeline_result: QueryPipelineResult,
    *,
    verbose: bool = False,
    out: TextIO | None = None,
) -> None:
    """将流水线结果格式化为人类可读输出。"""
    sink = out or sys.stdout
    final_results = pipeline_result.final_results

    if verbose:
        _format_result_block("=== Dense Results ===", pipeline_result.dense_results, sink)
        _format_result_block("=== Sparse Results ===", pipeline_result.sparse_results, sink)
        _format_result_block("=== Fusion Results ===", pipeline_result.fusion_results, sink)
        if pipeline_result.rerank_result is None:
            print("=== Rerank Results ===", file=sink)
            print("  (skipped)", file=sink)
        else:
            title = "=== Rerank Results ==="
            if pipeline_result.rerank_result.fallback:
                reason = pipeline_result.rerank_result.fallback_reason or "unknown"
                print(f"=== Rerank Results (fallback: {reason}) ===", file=sink)
            else:
                print(title, file=sink)
            _format_result_block("", pipeline_result.final_results, sink)

    print("=== Final Top-K ===", file=sink)
    if not final_results:
        print(_EMPTY_DATA_HINT, file=sink)
        return

    for index, item in enumerate(final_results, start=1):
        page = item.metadata.get("page")
        page_suffix = f" | page={page}" if page is not None else ""
        print(
            f"{index}. score={item.score:.4f} | source={_source_label(item.metadata)}{page_suffix}",
            file=sink,
        )
        print(f"   {_preview_text(item.text)}", file=sink)


def run_query(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    collection: str | None = None,
    verbose: bool = False,
    no_rerank: bool = False,
    settings_path: str | None = None,
    data_root: str | None = None,
    *,
    out: TextIO | None = None,
    query_processor: QueryProcessor | None = None,
    dense_retriever: DenseRetriever | None = None,
    sparse_retriever: SparseRetriever | None = None,
    fusion: RRFFusion | None = None,
    reranker: Reranker | None = None,
) -> int:
    """
    执行查询 CLI 主流程。

    Returns:
        进程退出码：0 成功，1 配置或查询失败。
    """
    logger = get_logger("query")
    sink = out or sys.stdout

    if not isinstance(query, str) or not query.strip():
        print("查询文本不能为空。", file=sink)
        return 0

    if top_k <= 0:
        print("top-k 必须大于 0。", file=sink)
        return 1

    try:
        base_settings = load_settings(settings_path)
    except SettingsError as exc:
        logger.error("配置加载失败: %s", exc)
        return 1

    settings, bm25_root = _settings_for_query(base_settings, collection, data_root)
    trace = (
        TraceContext(trace_type="query")
        if settings.observability.trace_enabled
        else None
    )

    filters = {"collection": collection} if collection else None

    try:
        pipeline_result = execute_query_pipeline(
            settings,
            query.strip(),
            top_k,
            filters=filters,
            no_rerank=no_rerank,
            trace=trace,
            query_processor=query_processor,
            dense_retriever=dense_retriever,
            sparse_retriever=sparse_retriever,
            fusion=fusion,
            reranker=reranker,
            bm25_root=bm25_root,
        )
    except (QueryProcessorError, HybridSearchError) as exc:
        logger.error("查询失败: %s", exc)
        return 1

    render_query_output(pipeline_result, verbose=verbose, out=sink)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：解析参数并调用 run_query。"""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_query(
        query=args.query,
        top_k=args.top_k,
        collection=args.collection,
        verbose=bool(args.verbose),
        no_rerank=bool(args.no_rerank),
        settings_path=args.settings,
        data_root=args.data_root,
    )


if __name__ == "__main__":
    sys.exit(main())
