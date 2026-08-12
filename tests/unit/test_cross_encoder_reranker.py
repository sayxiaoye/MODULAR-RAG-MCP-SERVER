"""Cross-Encoder Reranker 单元测试（mock scorer，保证确定性）。"""

from __future__ import annotations

import time

import pytest

from core.settings import RerankSettings, Settings, load_settings
from libs.reranker.base_reranker import RerankerError, RerankerFallbackSignal
from libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from libs.reranker.reranker_factory import RerankerFactory


def _sample_candidates() -> list[dict[str, object]]:
    return [
        {
            "id": "a",
            "score": 0.9,
            "text": "azure guide",
            "metadata": {"source_path": "doc1.pdf"},
        },
        {
            "id": "b",
            "score": 0.5,
            "text": "other topic with more words",
            "metadata": {"source_path": "doc2.pdf"},
        },
    ]


class FakeScorer:
    """按文本长度打分的 mock scorer，便于确定性测试。"""

    def __init__(self, should_fail: bool = False, delay_seconds: float = 0.0) -> None:
        self.should_fail = should_fail
        self.delay_seconds = delay_seconds

    def score(self, query: str, texts: list[str]) -> list[float]:
        if self.delay_seconds > 0:
            time.sleep(self.delay_seconds)
        if self.should_fail:
            raise RuntimeError("mock scorer failure")
        return [float(len(text)) for text in texts]


@pytest.mark.unit
class TestCrossEncoderReranker:
    """验证 Cross-Encoder 打分重排与回退信号。"""

    def test_rerank_by_scorer_scores(self) -> None:
        """较长文本得分更高时应排在前面。"""
        reranker = CrossEncoderReranker(
            RerankSettings(enabled=True, provider="cross_encoder", model="m", top_k=5),
            scorer=FakeScorer(),
        )
        result = reranker.rerank("azure", _sample_candidates())
        assert [item["id"] for item in result] == ["b", "a"]
        assert result[0]["score"] == float(len("other topic with more words"))

    def test_top_k_limits_output(self) -> None:
        """top_k 应限制返回条数。"""
        reranker = CrossEncoderReranker(
            RerankSettings(enabled=True, provider="cross_encoder", model="m", top_k=1),
            scorer=FakeScorer(),
        )
        result = reranker.rerank("q", _sample_candidates())
        assert len(result) == 1

    def test_scorer_failure_raises_fallback_signal(self) -> None:
        """scorer 异常时应抛出 RerankerFallbackSignal。"""
        reranker = CrossEncoderReranker(
            RerankSettings(enabled=True, provider="cross_encoder", model="m", top_k=5),
            scorer=FakeScorer(should_fail=True),
        )
        with pytest.raises(RerankerFallbackSignal, match="回退"):
            reranker.rerank("q", _sample_candidates())

    def test_timeout_raises_fallback_signal(self) -> None:
        """超时应抛出 RerankerFallbackSignal。"""
        reranker = CrossEncoderReranker(
            RerankSettings(enabled=True, provider="cross_encoder", model="m", top_k=5),
            scorer=FakeScorer(delay_seconds=0.2),
            timeout_seconds=0.05,
        )
        with pytest.raises(RerankerFallbackSignal, match="超时"):
            reranker.rerank("q", _sample_candidates())

    def test_invalid_score_length_raises(self) -> None:
        """scorer 返回长度不匹配时应报错。"""

        class BadScorer:
            def score(self, query: str, texts: list[str]) -> list[float]:
                return [1.0]

        reranker = CrossEncoderReranker(
            RerankSettings(enabled=True, provider="cross_encoder", model="m", top_k=5),
            scorer=BadScorer(),
        )
        with pytest.raises(RerankerError, match="不一致"):
            reranker.rerank("q", _sample_candidates())


@pytest.mark.unit
class TestCrossEncoderFactoryRouting:
    """验证 RerankerFactory 能创建 CrossEncoderReranker。"""

    def test_factory_creates_cross_encoder(self) -> None:
        """provider=cross_encoder 时应返回 CrossEncoderReranker。"""
        base = load_settings()
        settings = Settings(
            llm=base.llm,
            embedding=base.embedding,
            vector_store=base.vector_store,
            retrieval=base.retrieval,
            rerank=RerankSettings(
                enabled=True,
                provider="cross_encoder",
                model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                top_k=5,
            ),
            evaluation=base.evaluation,
            observability=base.observability,
            ingestion=base.ingestion,
            vision_llm=base.vision_llm,
        )
        reranker = RerankerFactory.create(settings)
        assert isinstance(reranker, CrossEncoderReranker)
