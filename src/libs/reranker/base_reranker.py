"""Reranker 抽象层：对检索候选进行精排，供 Core 层 HybridSearch 后处理复用。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

# 候选条目最小字段，与 VectorStore query / RetrievalResult 对齐
_REQUIRED_CANDIDATE_KEYS = frozenset({"id", "score", "text", "metadata"})


class RerankerError(Exception):
    """Reranker 调用或候选校验失败时抛出。"""


class RerankerFallbackSignal(RerankerError):
    """精排失败但可由 Core 层回退到 fusion 排名时抛出（供 D6 fallback 捕获）。"""


class BaseReranker(ABC):
    """Reranker 抽象基类：根据 query 对候选列表重新排序。"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
        trace: Any | None = None,
    ) -> list[dict[str, Any]]:
        """
        对检索候选按相关性重新排序。

        Args:
            query: 用户查询文本。
            candidates: 候选列表，每项含 id/score/text/metadata。
            trace: 可选追踪上下文（F 阶段注入）。

        Returns:
            重排后的候选列表（结构与输入一致，顺序可能变化）。
        """

    def _validate_query(self, query: str) -> str:
        """校验查询文本非空。"""
        if not isinstance(query, str) or not query.strip():
            raise RerankerError("query 必须是非空字符串")
        return query.strip()

    def _validate_candidates(
        self,
        candidates: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """校验候选 shape，并规范化为 dict 列表。"""
        if not candidates:
            return []
        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(candidates):
            if not isinstance(item, Mapping):
                raise RerankerError(f"candidates[{index}] 必须是 mapping")
            missing = _REQUIRED_CANDIDATE_KEYS - set(item.keys())
            if missing:
                raise RerankerError(
                    f"candidates[{index}] 缺少字段: {', '.join(sorted(missing))}"
                )
            if not isinstance(item["metadata"], Mapping):
                raise RerankerError(f"candidates[{index}].metadata 必须是 mapping")
            normalized.append(
                {
                    "id": str(item["id"]),
                    "score": float(item["score"]),
                    "text": str(item["text"]),
                    "metadata": dict(item["metadata"]),
                }
            )
        return normalized


class NoneReranker(BaseReranker):
    """空实现 Reranker：保持原顺序，用于 disabled 或 provider=none 时的默认回退。"""

    def rerank(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
        trace: Any | None = None,
    ) -> list[dict[str, Any]]:
        self._validate_query(query)
        return self._validate_candidates(candidates)
