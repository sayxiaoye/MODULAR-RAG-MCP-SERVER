"""Embedding 工厂与 BaseEmbedding 契约的单元测试。"""

from __future__ import annotations

import pytest

from core.settings import EmbeddingSettings, Settings, load_settings
from libs.embedding.base_embedding import BaseEmbedding, EmbeddingError
from libs.embedding.embedding_factory import (
    EmbeddingFactory,
    EmbeddingFactoryError,
    register_embedding_provider,
)


class FakeEmbedding(BaseEmbedding):
    """测试用 Fake Embedding：基于文本生成确定性向量，不发起网络请求。"""

    def __init__(self, settings: EmbeddingSettings) -> None:
        self.settings = settings
        self.dimensions = settings.dimensions

    def embed(self, texts, trace=None) -> list[list[float]]:
        validated = self._validate_texts(texts)
        vectors: list[list[float]] = []
        for text in validated:
            # 用字符码生成稳定、可复现的伪向量（维度与配置一致）
            seed = sum(ord(c) for c in text)
            vectors.append([(seed + i) % 97 / 97.0 for i in range(self.dimensions)])
        return vectors


@pytest.fixture(autouse=True)
def _reset_embedding_factory() -> None:
    """每个用例前后恢复工厂默认构造器，避免测试间污染。"""
    EmbeddingFactory.reset_constructor()
    yield
    EmbeddingFactory.reset_constructor()


@pytest.mark.unit
class TestFakeEmbedding:
    """验证 Fake Embedding 输出稳定且维度正确。"""

    def test_stable_vectors_for_same_text(self) -> None:
        """相同输入应产生相同向量（确定性）。"""
        base = load_settings()
        emb = FakeEmbedding(base.embedding)
        v1 = emb.embed(["hello"])
        v2 = emb.embed(["hello"])
        assert v1 == v2
        assert len(v1[0]) == base.embedding.dimensions

    def test_empty_texts_raises(self) -> None:
        """空 texts 应抛出可读错误。"""
        base = load_settings()
        emb = FakeEmbedding(base.embedding)
        with pytest.raises(EmbeddingError, match="texts 不能为空"):
            emb.embed([])


@pytest.mark.unit
class TestEmbeddingFactoryRouting:
    """验证工厂按 provider 路由到已注册实现。"""

    def test_fake_provider_routing(self) -> None:
        """注册 fake provider 后应返回 FakeEmbedding 并能批量编码。"""
        register_embedding_provider("fake", FakeEmbedding)
        base = load_settings()
        fake_settings = Settings(
            llm=base.llm,
            embedding=EmbeddingSettings(
                provider="fake",
                model="fake-embed",
                dimensions=8,
            ),
            vector_store=base.vector_store,
            retrieval=base.retrieval,
            rerank=base.rerank,
            evaluation=base.evaluation,
            observability=base.observability,
            ingestion=base.ingestion,
            vision_llm=base.vision_llm,
        )
        embedding = EmbeddingFactory.create(fake_settings)
        assert isinstance(embedding, FakeEmbedding)
        vectors = embedding.embed(["a", "b"])
        assert len(vectors) == 2
        assert len(vectors[0]) == 8
        assert vectors[0] != vectors[1]

    def test_unknown_provider_raises(self) -> None:
        """未注册的 provider 应抛出包含名称的错误。"""
        base = load_settings()
        unknown = Settings(
            llm=base.llm,
            embedding=EmbeddingSettings(
                provider="unknown_embed_xyz",
                model="m",
                dimensions=4,
            ),
            vector_store=base.vector_store,
            retrieval=base.retrieval,
            rerank=base.rerank,
            evaluation=base.evaluation,
            observability=base.observability,
            ingestion=base.ingestion,
            vision_llm=base.vision_llm,
        )
        with pytest.raises(EmbeddingFactoryError, match="unknown_embed_xyz"):
            EmbeddingFactory.create(unknown)
