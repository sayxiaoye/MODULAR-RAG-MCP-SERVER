"""Core 层 Reranker 编排：接入 libs.reranker 后端，失败时回退 fusion 排名。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult
from libs.reranker.base_reranker import BaseReranker, RerankerError, RerankerFallbackSignal
from libs.reranker.reranker_factory import RerankerFactory


class QueryRerankerError(Exception):
    """Core 层精排编排参数或契约校验失败时抛出。"""


@dataclass(frozen=True)
class RerankResult:
    """
    精排阶段输出契约。

    Attributes:
        results: 精排后（或回退后）的 RetrievalResult 列表。
        fallback: 是否因后端失败而回退到 fusion 原序。
        fallback_reason: 回退原因（仅 fallback=True 时有值）。
    """

    results: list[RetrievalResult]
    fallback: bool
    fallback_reason: str | None = None


class Reranker:
    """
    查询精排编排器：将 RetrievalResult 转为 libs 契约，调用可插拔后端并重排。

    对应 spec D6：后端异常/超时时回退 fusion 排名，并在结果中标记 ``fallback=true``。
    """

    def __init__(
        self,
        settings: Settings,
        backend: BaseReranker | None = None,
    ) -> None:
        self._settings = settings
        self._backend = backend or RerankerFactory.create(settings)

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        top_k: int | None = None,
        trace: Any | None = None,
    ) -> RerankResult:
        """
        对融合候选执行精排；失败时回退到输入顺序。

        Args:
            query: 用户查询文本。
            candidates: HybridSearch 产出的 fusion 排名候选。
            top_k: 返回条数上限；默认 ``settings.rerank.top_k``。
            trace: 可选 TraceContext。

        Returns:
            RerankResult，含结果列表与 fallback 标记。
        """
        if not isinstance(query, str) or not query.strip():
            raise QueryRerankerError("query 必须是非空字符串")

        limit = top_k if top_k is not None else self._settings.rerank.top_k
        if limit <= 0:
            raise QueryRerankerError("top_k 必须大于 0")

        baseline = list(candidates)
        if not baseline:
            return RerankResult(results=[], fallback=False)

        start = time.perf_counter()
        payload = [_to_backend_candidate(item) for item in baseline]

        try:
            ranked = self._backend.rerank(query.strip(), payload, trace=trace)
            results = [_from_backend_candidate(item) for item in ranked[:limit]]
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._record_trace(
                trace,
                elapsed_ms=elapsed_ms,
                fallback=False,
                fallback_reason=None,
                result_count=len(results),
            )
            return RerankResult(results=results, fallback=False)
        except RerankerFallbackSignal as exc:
            return self._build_fallback(
                baseline,
                limit,
                reason=str(exc),
                trace=trace,
                start=start,
            )
        except RerankerError as exc:
            return self._build_fallback(
                baseline,
                limit,
                reason=str(exc),
                trace=trace,
                start=start,
            )

    def _build_fallback(
        self,
        baseline: list[RetrievalResult],
        limit: int,
        reason: str,
        trace: Any | None,
        start: float,
    ) -> RerankResult:
        """后端失败时保留 fusion 原序并标记 fallback。"""
        results = baseline[:limit]
        elapsed_ms = (time.perf_counter() - start) * 1000
        self._record_trace(
            trace,
            elapsed_ms=elapsed_ms,
            fallback=True,
            fallback_reason=reason,
            result_count=len(results),
        )
        return RerankResult(
            results=results,
            fallback=True,
            fallback_reason=reason,
        )

    @staticmethod
    def _record_trace(
        trace: Any | None,
        *,
        elapsed_ms: float,
        fallback: bool,
        fallback_reason: str | None,
        result_count: int,
    ) -> None:
        if isinstance(trace, TraceContext):
            trace.record_stage(
                "reranker",
                elapsed_ms=elapsed_ms,
                fallback=fallback,
                fallback_reason=fallback_reason,
                result_count=result_count,
            )


def _to_backend_candidate(item: RetrievalResult) -> dict[str, Any]:
    """RetrievalResult → libs.reranker 候选契约（id 字段）。"""
    return {
        "id": item.chunk_id,
        "score": float(item.score),
        "text": item.text,
        "metadata": dict(item.metadata),
    }


def _from_backend_candidate(item: Mapping[str, Any]) -> RetrievalResult:
    """libs.reranker 输出 → RetrievalResult。"""
    return RetrievalResult(
        chunk_id=str(item["id"]),
        score=float(item["score"]),
        text=str(item.get("text", "")),
        metadata=dict(item.get("metadata", {})),
    )
