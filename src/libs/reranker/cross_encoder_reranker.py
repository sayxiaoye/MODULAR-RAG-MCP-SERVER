"""Cross-Encoder Reranker：对 query-candidate 对打分并重排候选。"""

from __future__ import annotations

import concurrent.futures
from typing import Any, Mapping, Protocol, Sequence

from core.settings import RerankSettings
from libs.reranker.base_reranker import BaseReranker, RerankerError, RerankerFallbackSignal

DEFAULT_SCORE_TIMEOUT_SECONDS = 30.0


class CrossEncoderScorer(Protocol):
    """Cross-Encoder 打分器协议：输入 query 与文本列表，返回相关性分数。"""

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """对每条 text 相对 query 的相关性打分，顺序与 texts 一致。"""


class _LazySentenceTransformerScorer:
    """默认打分器：懒加载 sentence-transformers CrossEncoder（可选依赖）。"""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: Any = None

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RerankerError(
                    "CrossEncoder Reranker 需要安装 sentence-transformers："
                    "pip install sentence-transformers"
                ) from exc
            self._model = CrossEncoder(self.model_name)
        pairs = [(query, text) for text in texts]
        raw_scores = self._model.predict(pairs)
        return [float(value) for value in raw_scores]


class CrossEncoderReranker(BaseReranker):
    """使用 Cross-Encoder 对 Top-M 候选逐对打分并重排。"""

    def __init__(
        self,
        settings: RerankSettings,
        scorer: CrossEncoderScorer | None = None,
        timeout_seconds: float = DEFAULT_SCORE_TIMEOUT_SECONDS,
    ) -> None:
        self.settings = settings
        self._scorer = scorer or _LazySentenceTransformerScorer(settings.model)
        self._timeout_seconds = timeout_seconds

    def _score_candidates(self, query: str, texts: Sequence[str]) -> list[float]:
        """在超时限制内调用 scorer，失败时抛出回退信号。"""
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self._scorer.score, query, texts)
                return future.result(timeout=self._timeout_seconds)
        except concurrent.futures.TimeoutError as exc:
            raise RerankerFallbackSignal(
                f"[cross_encoder] 打分超时（>{self._timeout_seconds}s），建议回退 fusion 排名"
            ) from exc
        except RerankerError:
            raise
        except Exception as exc:
            raise RerankerFallbackSignal(
                f"[cross_encoder] 打分失败，建议回退 fusion 排名: {exc}"
            ) from exc

    def rerank(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
        trace: Any | None = None,
    ) -> list[dict[str, Any]]:
        validated_query = self._validate_query(query)
        validated_candidates = self._validate_candidates(candidates)
        if not validated_candidates:
            return []

        texts = [item["text"] for item in validated_candidates]
        scores = self._score_candidates(validated_query, texts)
        if len(scores) != len(validated_candidates):
            raise RerankerError(
                f"scorer 返回分数数量 {len(scores)} 与候选数量 {len(validated_candidates)} 不一致"
            )

        scored_pairs: list[tuple[float, dict[str, Any]]] = []
        for index, candidate in enumerate(validated_candidates):
            reranked_item = dict(candidate)
            reranked_item["score"] = float(scores[index])
            scored_pairs.append((float(scores[index]), reranked_item))

        scored_pairs.sort(key=lambda item: item[0], reverse=True)
        top_k = max(1, self.settings.top_k)
        return [item[1] for item in scored_pairs[:top_k]]
