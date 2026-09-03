"""Reranker 工厂与 NoneReranker 回退行为的单元测试。"""

from __future__ import annotations

import pytest

from core.settings import RerankSettings, Settings, load_settings
from libs.reranker.base_reranker import BaseReranker, NoneReranker, RerankerError
from libs.reranker.reranker_factory import (
    RerankerFactory,
    RerankerFactoryError,
    register_reranker,
)


def _sample_candidates() -> list[dict[str, object]]:
    return [
        {
            "id": "a",
            "score": 0.9,
            "text": "first",
            "metadata": {"source_path": "doc1.pdf"},
        },
        {
            "id": "b",
            "score": 0.5,
            "text": "second",
            "metadata": {"source_path": "doc2.pdf"},
        },
    ]


class ReverseReranker(BaseReranker):
    """测试用 Reranker：反转候选顺序，验证工厂路由。"""

    def __init__(self, settings: RerankSettings) -> None:
        self.settings = settings

    def rerank(self, query, candidates, trace=None) -> list[dict[str, object]]:
        self._validate_query(query)
        validated = self._validate_candidates(candidates)
        return list(reversed(validated))


@pytest.fixture(autouse=True)
def _reset_reranker_factory() -> None:
    """每个用例前后恢复工厂默认构造器。"""
    RerankerFactory.reset_constructor()
    yield
    RerankerFactory.reset_constructor()


@pytest.mark.unit
class TestNoneReranker:
    """验证 NoneReranker 不改变候选顺序。"""

    def test_preserves_candidate_order(self) -> None:
        """输出顺序应与输入完全一致。"""
        reranker = NoneReranker()
        candidates = _sample_candidates()
        result = reranker.rerank("测试查询", candidates)
        assert [item["id"] for item in result] == ["a", "b"]

    def test_empty_candidates_returns_empty(self) -> None:
        """空候选列表应原样返回空列表。"""
        reranker = NoneReranker()
        assert reranker.rerank("query", []) == []


@pytest.mark.unit
class TestRerankerFactoryRouting:
    """验证工厂按配置路由或回退到 NoneReranker。"""

    def test_disabled_rerank_uses_none(self) -> None:
        """enabled=false 时应返回 NoneReranker。"""
        base = load_settings()
        settings = Settings(
            llm=base.llm,
            embedding=base.embedding,
            vector_store=base.vector_store,
            retrieval=base.retrieval,
            rerank=RerankSettings(
                enabled=False,
                provider="llm",
                model="m",
                top_k=5,
            ),
            evaluation=base.evaluation,
            observability=base.observability,
            ingestion=base.ingestion,
            vision_llm=base.vision_llm,
        )
        reranker = RerankerFactory.create(settings)
        assert isinstance(reranker, NoneReranker)

    def test_provider_none_uses_none_reranker(self) -> None:
        """provider=none 时应返回 NoneReranker（与默认 settings.yaml 一致）。"""
        reranker = RerankerFactory.create(load_settings())
        assert isinstance(reranker, NoneReranker)

    def test_fake_provider_routing(self) -> None:
        """注册自定义 provider 后应路由到对应实现。"""
        register_reranker("reverse", ReverseReranker)
        base = load_settings()
        settings = Settings(
            llm=base.llm,
            embedding=base.embedding,
            vector_store=base.vector_store,
            retrieval=base.retrieval,
            rerank=RerankSettings(
                enabled=True,
                provider="reverse",
                model="m",
                top_k=5,
            ),
            evaluation=base.evaluation,
            observability=base.observability,
            ingestion=base.ingestion,
            vision_llm=base.vision_llm,
        )
        reranker = RerankerFactory.create(settings)
        assert isinstance(reranker, ReverseReranker)
        result = reranker.rerank("q", _sample_candidates())
        assert [item["id"] for item in result] == ["b", "a"]

    def test_unknown_provider_raises(self) -> None:
        """启用但 provider 未注册时应抛出可读错误。"""
        base = load_settings()
        settings = Settings(
            llm=base.llm,
            embedding=base.embedding,
            vector_store=base.vector_store,
            retrieval=base.retrieval,
            rerank=RerankSettings(
                enabled=True,
                provider="unknown_rerank_xyz",
                model="m",
                top_k=5,
            ),
            evaluation=base.evaluation,
            observability=base.observability,
            ingestion=base.ingestion,
            vision_llm=base.vision_llm,
        )
        with pytest.raises(RerankerFactoryError, match="unknown_rerank_xyz"):
            RerankerFactory.create(settings)

    def test_invalid_candidate_raises(self) -> None:
        """缺少必填字段的候选应在校验阶段失败。"""
        reranker = NoneReranker()
        with pytest.raises(RerankerError, match="缺少字段"):
            reranker.rerank("q", [{"id": "only-id"}])

    def test_blank_query_raises(self) -> None:
        """空白 query 不是合法精排输入。"""
        reranker = NoneReranker()
        with pytest.raises(RerankerError, match="query"):
            reranker.rerank("   ", _sample_candidates())

    def test_candidate_not_mapping_raises(self) -> None:
        """候选必须是 mapping，否则无法校验 id/score/text/metadata。"""
        reranker = NoneReranker()
        with pytest.raises(RerankerError, match="必须是 mapping"):
            reranker.rerank("q", ["not-a-dict"])  # type: ignore[list-item]

    def test_candidate_metadata_not_mapping_raises(self) -> None:
        """metadata 必须是 mapping，与 RetrievalResult 契约对齐。"""
        reranker = NoneReranker()
        with pytest.raises(RerankerError, match="metadata"):
            reranker.rerank(
                "q",
                [
                    {
                        "id": "a",
                        "score": 0.1,
                        "text": "t",
                        "metadata": "bad",
                    }
                ],
            )

    def test_disabled_ignores_registered_provider(self) -> None:
        """enabled=false 时即使已注册自定义 provider 也必须回退 NoneReranker。"""
        register_reranker("reverse", ReverseReranker)
        base = load_settings()
        settings = Settings(
            llm=base.llm,
            embedding=base.embedding,
            vector_store=base.vector_store,
            retrieval=base.retrieval,
            rerank=RerankSettings(
                enabled=False,
                provider="reverse",
                model="m",
                top_k=5,
            ),
            evaluation=base.evaluation,
            observability=base.observability,
            ingestion=base.ingestion,
            vision_llm=base.vision_llm,
        )
        reranker = RerankerFactory.create(settings)
        assert isinstance(reranker, NoneReranker)

    def test_provider_none_is_case_insensitive(self) -> None:
        """provider 大小写不影响 none 回退。"""
        base = load_settings()
        settings = Settings(
            llm=base.llm,
            embedding=base.embedding,
            vector_store=base.vector_store,
            retrieval=base.retrieval,
            rerank=RerankSettings(
                enabled=True,
                provider="NONE",
                model="m",
                top_k=5,
            ),
            evaluation=base.evaluation,
            observability=base.observability,
            ingestion=base.ingestion,
            vision_llm=base.vision_llm,
        )
        assert isinstance(RerankerFactory.create(settings), NoneReranker)

    def test_register_blank_name_raises(self) -> None:
        """空 Provider 名不能写入注册表。"""
        with pytest.raises(RerankerFactoryError, match="不能为空"):
            register_reranker("  ", ReverseReranker)
