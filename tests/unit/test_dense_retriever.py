"""DenseRetriever 单元测试：mock Embedding 与 VectorStore 编排。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pytest

from core.settings import load_settings
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult
from core.query_engine.dense_retriever import DenseRetriever, DenseRetrieverError
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.vector_store_factory import VectorStoreFactory


class FakeEmbedding(BaseEmbedding):
    """记录 embed 调用并返回固定维度向量。"""

    def __init__(self, settings) -> None:
        self.settings = settings
        self.last_texts: list[str] | None = None

    def embed(self, texts, trace=None) -> list[list[float]]:
        self.last_texts = list(texts)
        dim = self.settings.dimensions
        return [[0.2] * dim for _ in texts]


class RecordingVectorStore(BaseVectorStore):
    """记录 query 参数并返回预设结果。"""

    def __init__(self) -> None:
        self.last_vector: list[float] | None = None
        self.last_top_k: int | None = None
        self.last_filters: Mapping[str, Any] | None = None
        self.last_collection: str | None = None
        self._results: list[dict[str, Any]] = []

    def set_results(self, results: list[dict[str, Any]]) -> None:
        self._results = results

    def upsert(
        self,
        records: Sequence[Mapping[str, Any]],
        trace: Any | None = None,
        *,
        collection: str | None = None,
    ) -> None:
        raise NotImplementedError

    def query(
        self,
        vector: Sequence[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace: Any | None = None,
        *,
        collection: str | None = None,
    ) -> list[dict[str, Any]]:
        self.last_vector = [float(v) for v in vector]
        self.last_top_k = top_k
        self.last_filters = dict(filters) if filters else None
        self.last_collection = collection
        return list(self._results[:top_k])

    def get_by_ids(
        self,
        ids: Sequence[str],
        trace: Any | None = None,
        *,
        collection: str | None = None,
    ) -> list[dict[str, Any]]:
        return []


@pytest.fixture(autouse=True)
def _reset_factories() -> None:
    EmbeddingFactory.reset_constructor()
    VectorStoreFactory.reset_constructor()
    yield
    EmbeddingFactory.reset_constructor()
    VectorStoreFactory.reset_constructor()


@pytest.mark.unit
class TestDenseRetriever:
    """验证 query→embed→vector_store.query 编排与结果契约。"""

    def test_retrieve_orchestrates_embedding_and_vector_store(self) -> None:
        settings = load_settings()
        embedding = FakeEmbedding(settings.embedding)
        vector_store = RecordingVectorStore()
        vector_store.set_results(
            [
                {
                    "id": "chunk-001",
                    "score": 0.92,
                    "text": "Azure 配置指南",
                    "metadata": {"source_path": "guide.pdf", "collection": "docs"},
                }
            ]
        )

        retriever = DenseRetriever(settings, embedding=embedding, vector_store=vector_store)
        results = retriever.retrieve("Azure 配置", top_k=3, filters={"collection": "docs"})

        assert embedding.last_texts == ["Azure 配置"]
        assert vector_store.last_top_k == 3
        assert vector_store.last_filters is None
        assert vector_store.last_collection == "docs"
        assert len(vector_store.last_vector) == settings.embedding.dimensions

        assert len(results) == 1
        assert results[0].chunk_id == "chunk-001"
        assert results[0].score == 0.92
        assert results[0].text == "Azure 配置指南"
        assert results[0].metadata["collection"] == "docs"

    def test_retrieval_result_serializable(self) -> None:
        settings = load_settings()
        embedding = FakeEmbedding(settings.embedding)
        vector_store = RecordingVectorStore()
        vector_store.set_results(
            [
                {
                    "id": "c1",
                    "score": 0.5,
                    "text": "hello",
                    "metadata": {"source_path": "a.pdf"},
                }
            ]
        )
        retriever = DenseRetriever(settings, embedding=embedding, vector_store=vector_store)
        payload = retriever.retrieve("hello", top_k=1)[0].to_dict()

        assert payload["chunk_id"] == "c1"
        assert payload["score"] == 0.5
        assert payload["text"] == "hello"
        assert isinstance(payload["metadata"], dict)

    def test_empty_query_raises(self) -> None:
        settings = load_settings()
        retriever = DenseRetriever(
            settings,
            embedding=FakeEmbedding(settings.embedding),
            vector_store=RecordingVectorStore(),
        )
        with pytest.raises(DenseRetrieverError, match="query"):
            retriever.retrieve("  ", top_k=5)

    def test_invalid_top_k_raises(self) -> None:
        settings = load_settings()
        retriever = DenseRetriever(
            settings,
            embedding=FakeEmbedding(settings.embedding),
            vector_store=RecordingVectorStore(),
        )
        with pytest.raises(DenseRetrieverError, match="top_k"):
            retriever.retrieve("test", top_k=0)

    def test_records_trace_stage(self) -> None:
        settings = load_settings()
        vector_store = RecordingVectorStore()
        vector_store.set_results(
            [
                {
                    "id": "c1",
                    "score": 0.8,
                    "text": "t",
                    "metadata": {},
                }
            ]
        )
        trace = TraceContext(trace_type="query")
        retriever = DenseRetriever(
            settings,
            embedding=FakeEmbedding(settings.embedding),
            vector_store=vector_store,
        )
        retriever.retrieve("Azure", top_k=1, trace=trace)

        trace.finish()
        stages = [stage["name"] for stage in trace.to_dict()["stages"]]
        assert "dense_retriever" in stages

    def test_from_dict_accepts_id_alias(self) -> None:
        result = RetrievalResult.from_dict(
            {
                "id": "alias-id",
                "score": 0.1,
                "text": "body",
                "metadata": {"k": "v"},
            }
        )
        assert result.chunk_id == "alias-id"
