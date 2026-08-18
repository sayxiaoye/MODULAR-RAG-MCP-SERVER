"""query.py CLI 单元测试：参数解析、输出格式与 no-rerank 行为。"""

from __future__ import annotations

import io
from typing import Any, Sequence

import pytest

from core.query_engine.dense_retriever import DenseRetrieverError
from core.query_engine.fusion import RRFFusion
from core.query_engine.query_processor import QueryProcessor
from core.query_engine.reranker import RerankResult, Reranker
from core.settings import load_settings
from core.types import RetrievalResult

# scripts/ 不在默认 pythonpath，按 ingest 测试惯例手动加入
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from query import (  # noqa: E402
    _EMPTY_DATA_HINT,
    build_arg_parser,
    execute_query_pipeline,
    render_query_output,
    run_query,
)


def _result(chunk_id: str, score: float = 0.8, text: str = "hello world") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text,
        metadata={"source_path": f"{chunk_id}.pdf", "page": 1},
    )


class FakeDenseRetriever:
    def retrieve(self, query, top_k, filters=None, trace=None):
        return [_result("dense-1")]


class FakeSparseRetriever:
    def retrieve(self, keywords, top_k, trace=None):
        return [_result("sparse-1")]


class FailingDenseRetriever:
    def retrieve(self, query, top_k, filters=None, trace=None):
        raise DenseRetrieverError("dense down")


class ReverseReranker:
    def rerank(self, query, candidates, top_k=None, trace=None):
        ordered = list(reversed(candidates))
        limit = top_k if top_k is not None else len(ordered)
        return RerankResult(results=list(ordered[:limit]), fallback=False)


@pytest.mark.unit
class TestQueryCli:
    """验证 query CLI 编排与输出契约。"""

    def test_build_arg_parser_requires_query(self) -> None:
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_run_query_empty_query_returns_zero(self) -> None:
        buffer = io.StringIO()
        exit_code = run_query("   ", out=buffer)
        assert exit_code == 0
        assert "不能为空" in buffer.getvalue()

    def test_execute_query_pipeline_no_rerank(self) -> None:
        settings = load_settings()
        pipeline = execute_query_pipeline(
            settings,
            "Azure 配置",
            top_k=2,
            no_rerank=True,
            query_processor=QueryProcessor(),
            dense_retriever=FakeDenseRetriever(),
            sparse_retriever=FakeSparseRetriever(),
            fusion=RRFFusion(settings),
            reranker=Reranker(settings),
        )
        assert pipeline.rerank_result is None
        assert pipeline.final_results
        assert {item.chunk_id for item in pipeline.final_results} == {"dense-1", "sparse-1"}

    def test_execute_query_pipeline_with_rerank(self) -> None:
        settings = load_settings()
        pipeline = execute_query_pipeline(
            settings,
            "Azure",
            top_k=2,
            query_processor=QueryProcessor(),
            dense_retriever=FakeDenseRetriever(),
            sparse_retriever=FakeSparseRetriever(),
            fusion=RRFFusion(settings),
            reranker=ReverseReranker(),
        )
        assert pipeline.rerank_result is not None
        assert pipeline.rerank_result.fallback is False

    def test_render_query_output_shows_empty_hint(self) -> None:
        from query import QueryPipelineResult

        buffer = io.StringIO()
        render_query_output(
            QueryPipelineResult(
                dense_results=[],
                sparse_results=[],
                fusion_results=[],
                final_results=[],
            ),
            out=buffer,
        )
        assert _EMPTY_DATA_HINT in buffer.getvalue()

    def test_render_verbose_includes_stage_blocks(self) -> None:
        from query import QueryPipelineResult

        buffer = io.StringIO()
        render_query_output(
            QueryPipelineResult(
                dense_results=[_result("d1")],
                sparse_results=[_result("s1")],
                fusion_results=[_result("f1")],
                final_results=[_result("f1")],
                rerank_result=None,
            ),
            verbose=True,
            out=buffer,
        )
        text = buffer.getvalue()
        assert "Dense Results" in text
        assert "Sparse Results" in text
        assert "Fusion Results" in text
        assert "skipped" in text

    def test_run_query_formats_final_results(self) -> None:
        settings = load_settings()
        buffer = io.StringIO()
        exit_code = run_query(
            "Azure",
            top_k=1,
            no_rerank=True,
            out=buffer,
            query_processor=QueryProcessor(),
            dense_retriever=FakeDenseRetriever(),
            sparse_retriever=FakeSparseRetriever(),
            fusion=RRFFusion(settings),
            reranker=Reranker(settings),
        )
        assert exit_code == 0
        output = buffer.getvalue()
        assert "Final Top-K" in output
        assert "dense-1.pdf" in output

    def test_dense_failure_still_returns_sparse_results(self) -> None:
        settings = load_settings()
        pipeline = execute_query_pipeline(
            settings,
            "Azure",
            top_k=2,
            no_rerank=True,
            query_processor=QueryProcessor(),
            dense_retriever=FailingDenseRetriever(),
            sparse_retriever=FakeSparseRetriever(),
            fusion=RRFFusion(settings),
            reranker=Reranker(settings),
        )
        assert [item.chunk_id for item in pipeline.final_results] == ["sparse-1"]
