#!/usr/bin/env python3
"""评估 CLI：加载黄金测试集，跑 HybridSearch + Evaluator，输出 metrics。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TextIO

# 支持直接执行 scripts/evaluate.py 时能找到 src 包
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from core.query_engine.dense_retriever import DenseRetriever
from core.query_engine.hybrid_search import HybridSearch
from core.query_engine.query_pipeline import settings_for_query
from core.query_engine.sparse_retriever import SparseRetriever
from core.settings import Settings, SettingsError, load_settings
from ingestion.storage.bm25_indexer import BM25Indexer, BM25IndexerError
from libs.evaluator.evaluator_factory import EvaluatorFactory
from libs.vector_store.vector_store_factory import VectorStoreFactory
from observability.evaluation.eval_runner import EvalReport, EvalRunner, EvalRunnerError
from observability.logger import get_logger

DEFAULT_TEST_SET = _REPO_ROOT / "tests" / "fixtures" / "golden_test_set.json"


def build_arg_parser() -> argparse.ArgumentParser:
    """构建 evaluate 命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="Modular RAG MCP Server — 黄金测试集评估入口",
    )
    parser.add_argument(
        "--test-set",
        default=None,
        help="黄金测试集 JSON 路径（默认 tests/fixtures/golden_test_set.json）",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="限定检索集合（覆盖 vector_store.collection_name）",
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
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 输出完整 EvalReport",
    )
    return parser


def _build_hybrid_search(settings: Settings, bm25_root: Path) -> HybridSearch:
    """按 collection 对齐 Chroma 与 BM25，组装 HybridSearch。"""
    vector_store = VectorStoreFactory.create(settings)
    indexer = BM25Indexer(
        collection=settings.vector_store.collection_name,
        index_root=bm25_root,
    )
    try:
        indexer.load()
    except BM25IndexerError:
        # 尚无稀疏索引时仍允许仅 Dense 检索
        pass
    return HybridSearch(
        settings,
        dense_retriever=DenseRetriever(settings, vector_store=vector_store),
        sparse_retriever=SparseRetriever(
            settings,
            bm25_indexer=indexer,
            vector_store=vector_store,
        ),
    )


def render_report(report: EvalReport, *, as_json: bool = False, out: TextIO | None = None) -> None:
    """把评估报告打印成可读文本或 JSON。"""
    sink = out or sys.stdout
    if as_json:
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), file=sink)
        return

    print("=== Evaluation Report ===", file=sink)
    print(f"cases: {report.case_count}", file=sink)
    print(f"hit_rate: {report.hit_rate:.4f}", file=sink)
    print(f"mrr: {report.mrr:.4f}", file=sink)
    extra = {
        key: value
        for key, value in report.metrics.items()
        if key not in {"hit_rate", "mrr"}
    }
    for key in sorted(extra):
        print(f"{key}: {extra[key]:.4f}", file=sink)

    for index, case in enumerate(report.cases, start=1):
        print(file=sink)
        print(f"--- query {index} ---", file=sink)
        print(f"query: {case.query}", file=sink)
        if case.error:
            print(f"error: {case.error}", file=sink)
            continue
        for key in ("hit_rate", "mrr"):
            if key in case.metrics:
                print(f"{key}: {case.metrics[key]:.4f}", file=sink)
        print(f"retrieved: {', '.join(case.retrieved_ids) or '(empty)'}", file=sink)
        print(f"golden: {', '.join(case.golden_ids)}", file=sink)


def run_evaluate(
    test_set: str | None = None,
    collection: str | None = None,
    settings_path: str | None = None,
    data_root: str | None = None,
    as_json: bool = False,
    *,
    out: TextIO | None = None,
    hybrid_search: Any | None = None,
    evaluator: Any | None = None,
) -> int:
    """
    执行评估 CLI 主流程。

    Returns:
        进程退出码：0 成功产出报告，1 配置或测试集失败。
    """
    logger = get_logger("evaluate")
    sink = out or sys.stdout
    try:
        base_settings = load_settings(settings_path)
    except SettingsError as exc:
        logger.error("配置加载失败: %s", exc)
        return 1

    settings, bm25_root = settings_for_query(base_settings, collection, data_root)
    test_set_path = Path(test_set) if test_set else DEFAULT_TEST_SET
    searcher = hybrid_search if hybrid_search is not None else _build_hybrid_search(settings, bm25_root)
    scorer = evaluator if evaluator is not None else EvaluatorFactory.create(settings)

    try:
        report = EvalRunner(settings, searcher, scorer).run(test_set_path)
    except EvalRunnerError as exc:
        logger.error("评估失败: %s", exc)
        print(f"评估失败: {exc}", file=sink)
        return 1

    render_report(report, as_json=as_json, out=sink)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：解析参数并调用 run_evaluate。"""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_evaluate(
        test_set=args.test_set,
        collection=args.collection,
        settings_path=args.settings,
        data_root=args.data_root,
        as_json=bool(args.json),
    )


if __name__ == "__main__":
    raise SystemExit(main())
