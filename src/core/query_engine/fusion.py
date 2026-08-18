"""RRFFusion：将 Dense/Sparse 多路检索结果按 Reciprocal Rank Fusion 合并排序。"""

from __future__ import annotations

import time
from typing import Any, Sequence

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult


class FusionError(Exception):
    """结果融合失败时抛出。"""


class RRFFusion:
    """
    Reciprocal Rank Fusion 实现：按各路排名倒数加权，输出统一排序。

    公式：``score(d) = Σ 1 / (k + rank_i(d))``，rank 从 1 起算；未出现在某路则该项为 0。
    k 默认取自 ``settings.retrieval.rrf_k``。
    """

    def __init__(self, settings: Settings, rrf_k: int | None = None) -> None:
        self._settings = settings
        configured_k = rrf_k if rrf_k is not None else settings.retrieval.rrf_k
        if configured_k <= 0:
            raise FusionError("rrf_k 必须大于 0")
        self._rrf_k = configured_k

    def fuse(
        self,
        result_lists: Sequence[Sequence[RetrievalResult]],
        top_k: int | None = None,
        trace: Any | None = None,
    ) -> list[RetrievalResult]:
        """
        融合多路 RetrievalResult 排名并返回 Top-K。

        Args:
            result_lists: 各路检索结果（如 dense、sparse），每路已按相关性降序。
            top_k: 返回条数上限；默认使用 ``settings.retrieval.fusion_top_k``。
            trace: 可选 TraceContext。

        Returns:
            按 RRF 分数降序的 RetrievalResult 列表；score 为融合分。
        """
        limit = top_k if top_k is not None else self._settings.retrieval.fusion_top_k
        if limit <= 0:
            raise FusionError("top_k 必须大于 0")

        start = time.perf_counter()
        fused_scores: dict[str, float] = {}
        # 保留首次出现的正文与 metadata（通常各路 chunk_id 对齐）
        records: dict[str, RetrievalResult] = {}

        for result_list in result_lists:
            if not result_list:
                continue
            for rank, item in enumerate(result_list, start=1):
                chunk_id = item.chunk_id
                fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + (
                    1.0 / (self._rrf_k + rank)
                )
                records.setdefault(chunk_id, item)

        if not fused_scores:
            if isinstance(trace, TraceContext):
                trace.record_stage(
                    "fusion",
                    elapsed_ms=0.0,
                    method="rrf",
                    rrf_k=self._rrf_k,
                    input_lists=len(result_lists),
                    result_count=0,
                )
            return []

        # 分数降序；同分按 chunk_id 升序，保证 deterministic
        ranked_ids = sorted(
            fused_scores.keys(),
            key=lambda chunk_id: (-fused_scores[chunk_id], chunk_id),
        )

        results: list[RetrievalResult] = []
        for chunk_id in ranked_ids[:limit]:
            source = records[chunk_id]
            results.append(
                RetrievalResult(
                    chunk_id=chunk_id,
                    score=fused_scores[chunk_id],
                    text=source.text,
                    metadata=dict(source.metadata),
                )
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        if isinstance(trace, TraceContext):
            trace.record_stage(
                "fusion",
                elapsed_ms=elapsed_ms,
                method="rrf",
                rrf_k=self._rrf_k,
                input_lists=len(result_lists),
                result_count=len(results),
            )

        return results
