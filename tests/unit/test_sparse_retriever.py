"""SparseRetriever 单元测试：mock BM25Indexer 与 VectorStore 编排。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pytest

from core.settings import load_settings
from core.trace.trace_context import TraceContext
from core.types import Chunk, RetrievalResult
from core.query_engine.sparse_retriever import SparseRetriever, SparseRetrieverError
from ingestion.embedding.sparse_encoder import SparseEncoder
from ingestion.storage.bm25_indexer import BM25Indexer
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.vector_store_factory import VectorStoreFactory


class RecordingBM25Indexer:
    """记录 query 调用并返回预设 BM25 排名。"""

    def __init__(self) -> None:
        self.last_query_text: str | None = None
        self.last_top_k: int | None = None
        self._results: list[tuple[str, float]] = []

    def set_results(self, results: list[tuple[str, float]]) -> None:
        self._results = results

    def query(self, query_text: str, top_k: int = 10) -> list[tuple[str, float]]:
        self.last_query_text = query_text
        self.last_top_k = top_k
        return list(self._results[:top_k])


class RecordingVectorStore(BaseVectorStore):
    """记录 get_by_ids 参数并返回预设记录。"""

    def __init__(self) -> None:
        self.last_ids: list[str] | None = None
        self._records: dict[str, dict[str, Any]] = {}

    def set_records(self, records: list[dict[str, Any]]) -> None:
        self._records = {str(item["id"]): item for item in records}

    def upsert(
        self,
        records: Sequence[Mapping[str, Any]],
        trace: Any | None = None,
    ) -> None:
        raise NotImplementedError

    def query(
        self,
        vector: Sequence[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError

    def get_by_ids(
        self,
        ids: Sequence[str],
        trace: Any | None = None,
    ) -> list[dict[str, Any]]:
        self.last_ids = [str(record_id) for record_id in ids]
        results: list[dict[str, Any]] = []
        for record_id in self.last_ids:
            record = self._records.get(record_id)
            if record is not None:
                results.append(
                    {
                        "id": record["id"],
                        "text": record["text"],
                        "metadata": dict(record["metadata"]),
                    }
                )
        return self._validate_get_by_ids_results(results)


def _bm25_stats(chunk_id: str, text: str):
    """用 SparseEncoder 生成 BM25 索引所需的统计结构。"""
    encoder = SparseEncoder()
    return encoder.encode(
        [
            Chunk(
                id=chunk_id,
                text=text,
                metadata={"source_path": "test.md"},
                start_offset=0,
                end_offset=len(text),
            )
        ]
    )[0]


@pytest.fixture(autouse=True)
def _reset_vector_store_factory() -> None:
    VectorStoreFactory.reset_constructor()
    yield
    VectorStoreFactory.reset_constructor()


@pytest.mark.unit
class TestSparseRetriever:
    """验证 keywords→BM25→get_by_ids 编排与结果契约。"""

    def test_retrieve_orchestrates_bm25_and_vector_store(self) -> None:
        settings = load_settings()
        bm25 = RecordingBM25Indexer()
        bm25.set_results([("chunk-001", 3.5), ("chunk-002", 2.1)])
        vector_store = RecordingVectorStore()
        vector_store.set_records(
            [
                {
                    "id": "chunk-001",
                    "text": "Azure 配置指南",
                    "metadata": {"source_path": "guide.pdf", "collection": "docs"},
                },
                {
                    "id": "chunk-002",
                    "text": "BM25 检索说明",
                    "metadata": {"source_path": "sparse.md", "collection": "docs"},
                },
            ]
        )

        retriever = SparseRetriever(settings, bm25_indexer=bm25, vector_store=vector_store)
        results = retriever.retrieve(["azure", "配置"], top_k=2)

        assert bm25.last_query_text == "azure 配置"
        assert bm25.last_top_k == 2
        assert vector_store.last_ids == ["chunk-001", "chunk-002"]

        assert len(results) == 2
        assert results[0].chunk_id == "chunk-001"
        assert results[0].score == 3.5
        assert results[0].text == "Azure 配置指南"
        assert results[0].metadata["collection"] == "docs"
        assert results[1].chunk_id == "chunk-002"

    def test_retrieval_result_serializable(self) -> None:
        settings = load_settings()
        bm25 = RecordingBM25Indexer()
        bm25.set_results([("c1", 1.0)])
        vector_store = RecordingVectorStore()
        vector_store.set_records(
            [
                {
                    "id": "c1",
                    "text": "hello",
                    "metadata": {"source_path": "a.pdf"},
                }
            ]
        )
        retriever = SparseRetriever(settings, bm25_indexer=bm25, vector_store=vector_store)
        payload = retriever.retrieve(["hello"], top_k=1)[0].to_dict()

        assert payload["chunk_id"] == "c1"
        assert payload["score"] == 1.0
        assert payload["text"] == "hello"
        assert isinstance(payload["metadata"], dict)

    def test_fixture_corpus_hits_expected_chunk_id(self, tmp_path) -> None:
        """对已构建 BM25 索引的 fixtures 语料，关键词应命中预期 chunk。"""
        stats = [
            _bm25_stats("chunk_a", "Azure OpenAI 配置指南"),
            _bm25_stats("chunk_b", "BM25 检索与 RRF 融合"),
            _bm25_stats("chunk_c", "Azure 向量数据库部署"),
        ]
        indexer = BM25Indexer(collection="fixture", index_root=tmp_path / "bm25")
        indexer.build(stats, rebuild=True)

        vector_store = RecordingVectorStore()
        vector_store.set_records(
            [
                {
                    "id": "chunk_a",
                    "text": "Azure OpenAI 配置指南",
                    "metadata": {"source_path": "a.md"},
                },
                {
                    "id": "chunk_c",
                    "text": "Azure 向量数据库部署",
                    "metadata": {"source_path": "c.md"},
                },
            ]
        )

        settings = load_settings()
        retriever = SparseRetriever(
            settings,
            bm25_indexer=indexer,
            vector_store=vector_store,
        )
        results = retriever.retrieve(["azure", "配置"], top_k=2)

        assert results
        assert results[0].chunk_id in {"chunk_a", "chunk_c"}
        assert results[0].text
        assert results[0].metadata["source_path"]

    def test_empty_keywords_raises(self) -> None:
        settings = load_settings()
        retriever = SparseRetriever(
            settings,
            bm25_indexer=RecordingBM25Indexer(),
            vector_store=RecordingVectorStore(),
        )
        with pytest.raises(SparseRetrieverError, match="keywords"):
            retriever.retrieve([], top_k=5)

    def test_invalid_top_k_raises(self) -> None:
        settings = load_settings()
        retriever = SparseRetriever(
            settings,
            bm25_indexer=RecordingBM25Indexer(),
            vector_store=RecordingVectorStore(),
        )
        with pytest.raises(SparseRetrieverError, match="top_k"):
            retriever.retrieve(["azure"], top_k=0)

    def test_records_trace_stage(self) -> None:
        settings = load_settings()
        bm25 = RecordingBM25Indexer()
        bm25.set_results([("c1", 0.8)])
        vector_store = RecordingVectorStore()
        vector_store.set_records(
            [
                {
                    "id": "c1",
                    "text": "t",
                    "metadata": {},
                }
            ]
        )
        trace = TraceContext(trace_type="query")
        retriever = SparseRetriever(settings, bm25_indexer=bm25, vector_store=vector_store)
        retriever.retrieve(["azure"], top_k=1, trace=trace)

        stages = [stage["name"] for stage in trace.finish()["stages"]]
        assert "sparse_retriever" in stages

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

    def test_bm25_no_hits_returns_empty(self) -> None:
        settings = load_settings()
        bm25 = RecordingBM25Indexer()
        vector_store = RecordingVectorStore()
        retriever = SparseRetriever(settings, bm25_indexer=bm25, vector_store=vector_store)

        assert retriever.retrieve(["missing"], top_k=5) == []
        assert vector_store.last_ids is None
