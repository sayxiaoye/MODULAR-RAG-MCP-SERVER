"""HybridSearch 集成测试：验证混合检索编排、过滤与单路降级。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pytest

from core.query_engine.dense_retriever import DenseRetrieverError
from core.query_engine.fusion import RRFFusion
from core.query_engine.hybrid_search import HybridSearch, HybridSearchError
from core.query_engine.query_processor import QueryProcessor
from core.query_engine.sparse_retriever import SparseRetrieverError
from core.settings import load_settings
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult


def _result(
    chunk_id: str,
    score: float = 0.9,
    text: str = "",
    metadata: dict[str, Any] | None = None,
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text or f"text-{chunk_id}",
        metadata=metadata or {"source_path": f"{chunk_id}.md", "collection": "docs"},
    )


class FakeDenseRetriever:
    """可控 Dense 召回结果，用于集成测试。"""

    def __init__(self, results: list[RetrievalResult] | None = None, *, fail: bool = False) -> None:
        self._results = results or []
        self._fail = fail
        self.last_query: str | None = None
        self.last_filters: Mapping[str, Any] | None = None

    def retrieve(
        self,
        query: str,
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[RetrievalResult]:
        self.last_query = query
        self.last_filters = dict(filters) if filters else None
        if self._fail:
            raise DenseRetrieverError("dense failed")
        return list(self._results[:top_k])


class FakeSparseRetriever:
    """可控 Sparse 召回结果，用于集成测试。"""

    def __init__(self, results: list[RetrievalResult] | None = None, *, fail: bool = False) -> None:
        self._results = results or []
        self._fail = fail
        self.last_keywords: list[str] | None = None

    def retrieve(
        self,
        keywords: Sequence[str],
        top_k: int,
        trace: Any | None = None,
    ) -> list[RetrievalResult]:
        self.last_keywords = list(keywords)
        if self._fail:
            raise SparseRetrieverError("sparse failed")
        return list(self._results[:top_k])


@pytest.fixture
def settings():
    return load_settings()


@pytest.mark.integration
class TestHybridSearch:
    """验证混合检索端到端编排行为。"""

    def test_search_returns_fused_top_k(self, settings) -> None:
        """fixtures 风格语料下应返回融合后的 Top-K 且包含正文与 metadata。"""
        dense = FakeDenseRetriever(
            [
                _result("chunk_a", text="Azure OpenAI 配置指南"),
                _result("chunk_b", text="BM25 检索与 RRF 融合"),
            ]
        )
        sparse = FakeSparseRetriever(
            [
                _result("chunk_b", text="BM25 检索与 RRF 融合"),
                _result("chunk_c", text="Azure 向量数据库部署"),
            ]
        )
        engine = HybridSearch(
            settings,
            query_processor=QueryProcessor(),
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=RRFFusion(settings, rrf_k=60),
        )

        results = engine.search("Azure 配置指南", top_k=2)

        assert len(results) == 2
        assert all(item.text and item.metadata.get("source_path") for item in results)
        assert results[0].chunk_id in {"chunk_a", "chunk_b", "chunk_c"}

    def test_metadata_filters_exclude_non_matching(self, settings) -> None:
        """后置 filters 应过滤掉 metadata 不匹配的候选。"""
        dense = FakeDenseRetriever(
            [
                _result("keep", metadata={"collection": "docs", "source_path": "a.md"}),
                _result("skip", metadata={"collection": "other", "source_path": "b.md"}),
            ]
        )
        sparse = FakeSparseRetriever([])
        engine = HybridSearch(
            settings,
            dense_retriever=dense,
            sparse_retriever=sparse,
            fusion=RRFFusion(settings),
        )

        results = engine.search("Azure", top_k=5, filters={"collection": "docs"})

        assert [item.chunk_id for item in results] == ["keep"]

    def test_missing_metadata_field_is_included(self, settings) -> None:
        """缺失 filter 字段的候选应被保留（missing->include）。"""
        dense = FakeDenseRetriever(
            [_result("no-collection", metadata={"source_path": "x.md"})]
        )
        engine = HybridSearch(
            settings,
            dense_retriever=dense,
            sparse_retriever=FakeSparseRetriever([]),
            fusion=RRFFusion(settings),
        )

        results = engine.search("Azure", top_k=1, filters={"collection": "docs"})
        assert len(results) == 1
        assert results[0].chunk_id == "no-collection"

    def test_dense_failure_falls_back_to_sparse(self, settings) -> None:
        """Dense 失败时应降级为仅 Sparse 结果。"""
        sparse = FakeSparseRetriever([_result("only-sparse")])
        engine = HybridSearch(
            settings,
            dense_retriever=FakeDenseRetriever(fail=True),
            sparse_retriever=sparse,
            fusion=RRFFusion(settings),
        )

        results = engine.search("Azure 配置", top_k=3)
        assert len(results) == 1
        assert results[0].chunk_id == "only-sparse"

    def test_sparse_failure_falls_back_to_dense(self, settings) -> None:
        """Sparse 失败时应降级为仅 Dense 结果。"""
        dense = FakeDenseRetriever([_result("only-dense")])
        engine = HybridSearch(
            settings,
            dense_retriever=dense,
            sparse_retriever=FakeSparseRetriever(fail=True),
            fusion=RRFFusion(settings),
        )

        results = engine.search("Azure 配置", top_k=3)
        assert len(results) == 1
        assert results[0].chunk_id == "only-dense"

    def test_records_hybrid_search_trace_stage(self, settings) -> None:
        """应记录 hybrid_search 阶段及子组件阶段。"""
        engine = HybridSearch(
            settings,
            dense_retriever=FakeDenseRetriever([_result("a")]),
            sparse_retriever=FakeSparseRetriever([_result("b")]),
            fusion=RRFFusion(settings),
        )
        trace = TraceContext(trace_type="query")
        engine.search("Azure 配置", top_k=1, trace=trace)

        stages = [stage["name"] for stage in trace.finish()["stages"]]
        assert "query_processor" in stages
        assert "fusion" in stages
        assert "hybrid_search" in stages

    def test_invalid_top_k_raises(self, settings) -> None:
        engine = HybridSearch(
            settings,
            dense_retriever=FakeDenseRetriever([]),
            sparse_retriever=FakeSparseRetriever([]),
            fusion=RRFFusion(settings),
        )
        with pytest.raises(HybridSearchError, match="top_k"):
            engine.search("Azure", top_k=0)

    def test_empty_query_raises(self, settings) -> None:
        engine = HybridSearch(
            settings,
            dense_retriever=FakeDenseRetriever([]),
            sparse_retriever=FakeSparseRetriever([]),
            fusion=RRFFusion(settings),
        )
        with pytest.raises(HybridSearchError, match="查询预处理"):
            engine.search("   ", top_k=1)
