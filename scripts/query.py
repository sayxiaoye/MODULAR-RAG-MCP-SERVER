#!/usr/bin/env python3
"""在线查询 CLI：调用 HybridSearch + Reranker 并格式化输出检索结果。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, TextIO

# 支持直接执行 scripts/query.py 时能找到 src 包
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from core.query_engine.dense_retriever import DenseRetriever
from core.query_engine.fusion import RRFFusion
from core.query_engine.hybrid_search import HybridSearchError
from core.query_engine.query_pipeline import (
    DEFAULT_TOP_K,
    QueryPipelineResult,
    execute_query_pipeline,
    settings_for_query,
)
from core.query_engine.query_processor import QueryProcessor, QueryProcessorError
from core.query_engine.reranker import Reranker
from core.query_engine.sparse_retriever import SparseRetriever
from core.settings import SettingsError, load_settings
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult
from observability.logger import get_logger

_TEXT_PREVIEW_LEN = 120
_EMPTY_DATA_HINT = "未找到相关文档，请先运行 ingest.py 摄取数据"


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

    settings, bm25_root = settings_for_query(base_settings, collection, data_root)
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
