"""Core 层 Reranker 回退测试：验证后端失败时保留 fusion 排名。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pytest

from core.query_engine.reranker import QueryRerankerError, RerankResult, Reranker
from core.settings import load_settings
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult
from libs.reranker.base_reranker import (
    BaseReranker,
    NoneReranker,
    RerankerError,
    RerankerFallbackSignal,
)


def _candidate(
    chunk_id: str,
    score: float = 0.5,
    text: str = "",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text or f"text-{chunk_id}",
        metadata={"source_path": f"{chunk_id}.md"},
    )


class ReverseBackend(BaseReranker):
    """测试后端：反转候选顺序。"""

    def rerank(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
        trace: Any | None = None,
    ) -> list[dict[str, Any]]:
        validated = self._validate_candidates(candidates)
        return list(reversed(validated))


class FailingBackend(BaseReranker):
    """测试后端：抛出可回退信号。"""

    def __init__(self, error: Exception) -> None:
        self._error = error

    def rerank(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
        trace: Any | None = None,
    ) -> list[dict[str, Any]]:
        self._validate_query(query)
        self._validate_candidates(candidates)
        raise self._error


@pytest.mark.unit
class TestRerankerFallback:
    """验证精排成功、禁用与失败回退场景。"""

    def test_successful_rerank_returns_reordered_results(self) -> None:
        """后端成功时应返回重排结果且 fallback=false。"""
        settings = load_settings()
        reranker = Reranker(settings, backend=ReverseBackend())
        candidates = [_candidate("a", 0.9), _candidate("b", 0.8)]

        result = reranker.rerank("Azure 配置", candidates, top_k=2)

        assert isinstance(result, RerankResult)
        assert result.fallback is False
        assert result.fallback_reason is None
        assert [item.chunk_id for item in result.results] == ["b", "a"]

    def test_fallback_signal_returns_fusion_order(self) -> None:
        """RerankerFallbackSignal 时应回退 fusion 原序并标记 fallback=true。"""
        settings = load_settings()
        backend = FailingBackend(RerankerFallbackSignal("timeout"))
        reranker = Reranker(settings, backend=backend)
        candidates = [_candidate("a"), _candidate("b"), _candidate("c")]

        result = reranker.rerank("query", candidates, top_k=2)

        assert result.fallback is True
        assert result.fallback_reason == "timeout"
        assert [item.chunk_id for item in result.results] == ["a", "b"]

    def test_backend_error_falls_back_to_fusion_order(self) -> None:
        """普通 RerankerError 也应回退 fusion 排名，保证可用性。"""
        settings = load_settings()
        backend = FailingBackend(RerankerError("backend exploded"))
        reranker = Reranker(settings, backend=backend)
        candidates = [_candidate("x"), _candidate("y")]

        result = reranker.rerank("query", candidates, top_k=2)

        assert result.fallback is True
        assert "backend exploded" in (result.fallback_reason or "")
        assert [item.chunk_id for item in result.results] == ["x", "y"]

    def test_none_backend_passthrough_without_fallback(self) -> None:
        """provider=none 时应保持原序且 fallback=false。"""
        settings = load_settings()
        reranker = Reranker(settings, backend=NoneReranker())
        candidates = [_candidate("a"), _candidate("b")]

        result = reranker.rerank("query", candidates, top_k=2)

        assert result.fallback is False
        assert [item.chunk_id for item in result.results] == ["a", "b"]

    def test_empty_candidates_returns_empty(self) -> None:
        settings = load_settings()
        reranker = Reranker(settings, backend=ReverseBackend())
        result = reranker.rerank("query", [], top_k=5)
        assert result.results == []
        assert result.fallback is False

    def test_records_trace_with_fallback_flag(self) -> None:
        """trace 中应记录 reranker 阶段及 fallback 标记。"""
        settings = load_settings()
        backend = FailingBackend(RerankerFallbackSignal("mock failure"))
        reranker = Reranker(settings, backend=backend)
        trace = TraceContext(trace_type="query")

        reranker.rerank("query", [_candidate("a")], top_k=1, trace=trace)
        trace.finish()
        payload = trace.to_dict()
        rerank_stage = next(stage for stage in payload["stages"] if stage["name"] == "reranker")

        assert rerank_stage["fallback"] is True
        assert rerank_stage["fallback_reason"] == "mock failure"

    def test_invalid_query_raises(self) -> None:
        settings = load_settings()
        reranker = Reranker(settings, backend=NoneReranker())
        with pytest.raises(QueryRerankerError, match="query"):
            reranker.rerank("  ", [_candidate("a")], top_k=1)

    def test_invalid_top_k_raises(self) -> None:
        settings = load_settings()
        reranker = Reranker(settings, backend=NoneReranker())
        with pytest.raises(QueryRerankerError, match="top_k"):
            reranker.rerank("query", [_candidate("a")], top_k=0)
