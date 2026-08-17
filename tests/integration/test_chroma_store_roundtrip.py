"""ChromaStore 集成测试：真实 upsert→query roundtrip 与持久化验证。"""

from __future__ import annotations

import shutil
import tempfile

import pytest

from core.settings import Settings, VectorStoreSettings, load_settings
from libs.vector_store.chroma_store import ChromaStore
from libs.vector_store.vector_store_factory import VectorStoreFactory


def _vector_store_settings(
    persist_directory: str,
    collection_name: str = "test_collection",
) -> VectorStoreSettings:
    return VectorStoreSettings(
        provider="chroma",
        persist_directory=persist_directory,
        collection_name=collection_name,
    )


@pytest.fixture
def chroma_temp_dir() -> str:
    """为每个用例创建独立临时目录，结束后自动清理。"""
    path = tempfile.mkdtemp(prefix="chroma_test_")
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.mark.integration
class TestChromaStoreRoundtrip:
    """验证 ChromaStore 写入、检索与持久化行为。"""

    def test_upsert_and_query_roundtrip(self, chroma_temp_dir: str) -> None:
        """upsert 后应能检索到最相似记录并返回完整字段。"""
        store = ChromaStore(_vector_store_settings(chroma_temp_dir))
        records = [
            {
                "id": "chunk-001",
                "text": "Azure 配置指南",
                "metadata": {"source_path": "guide.pdf", "collection": "docs"},
                "dense_vector": [1.0, 0.0, 0.0],
            },
            {
                "id": "chunk-002",
                "text": "其他主题",
                "metadata": {"source_path": "other.pdf", "collection": "docs"},
                "dense_vector": [0.0, 1.0, 0.0],
            },
        ]
        store.upsert(records)
        results = store.query([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == "chunk-001"
        assert results[0]["text"] == "Azure 配置指南"
        assert results[0]["metadata"]["source_path"] == "guide.pdf"
        assert isinstance(results[0]["score"], float)

    def test_top_k_limits_results(self, chroma_temp_dir: str) -> None:
        """top_k 应限制返回条数。"""
        store = ChromaStore(_vector_store_settings(chroma_temp_dir, "topk_test"))
        store.upsert(
            [
                {
                    "id": f"id-{i}",
                    "text": f"text-{i}",
                    "metadata": {"collection": "batch"},
                    "dense_vector": [float(i), 1.0, 0.0],
                }
                for i in range(5)
            ]
        )
        results = store.query([4.0, 1.0, 0.0], top_k=2)
        assert len(results) == 2

    def test_query_with_metadata_filters(self, chroma_temp_dir: str) -> None:
        """metadata filters 应只返回匹配记录。"""
        store = ChromaStore(_vector_store_settings(chroma_temp_dir, "filter_test"))
        store.upsert(
            [
                {
                    "id": "a",
                    "text": "keep me",
                    "metadata": {"collection": "keep"},
                    "dense_vector": [1.0, 0.0],
                },
                {
                    "id": "b",
                    "text": "skip me",
                    "metadata": {"collection": "skip"},
                    "dense_vector": [1.0, 0.0],
                },
            ]
        )
        results = store.query([1.0, 0.0], top_k=10, filters={"collection": "keep"})
        assert len(results) == 1
        assert results[0]["id"] == "a"

    def test_persistence_roundtrip(self, chroma_temp_dir: str) -> None:
        """重启客户端后应能从持久化目录读到已写入数据。"""
        settings = _vector_store_settings(chroma_temp_dir, "persist_test")
        writer = ChromaStore(settings)
        writer.upsert(
            [
                {
                    "id": "persist-1",
                    "text": "持久化文本",
                    "metadata": {"collection": "persist"},
                    "dense_vector": [0.9, 0.1, 0.0],
                }
            ]
        )
        reader = ChromaStore(settings)
        results = reader.query([0.9, 0.1, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == "persist-1"
        assert results[0]["text"] == "持久化文本"

    def test_get_by_ids_returns_stored_records(self, chroma_temp_dir: str) -> None:
        """get_by_ids 应批量返回已写入的 text 与 metadata。"""
        store = ChromaStore(_vector_store_settings(chroma_temp_dir, "get_by_ids_test"))
        store.upsert(
            [
                {
                    "id": "chunk-001",
                    "text": "Azure 配置指南",
                    "metadata": {"source_path": "guide.pdf", "collection": "docs"},
                    "dense_vector": [1.0, 0.0, 0.0],
                },
                {
                    "id": "chunk-002",
                    "text": "其他主题",
                    "metadata": {"source_path": "other.pdf", "collection": "docs"},
                    "dense_vector": [0.0, 1.0, 0.0],
                },
            ]
        )
        results = store.get_by_ids(["chunk-001", "missing-id"])
        assert len(results) == 1
        assert results[0]["id"] == "chunk-001"
        assert results[0]["text"] == "Azure 配置指南"
        assert results[0]["metadata"]["source_path"] == "guide.pdf"


@pytest.mark.integration
class TestChromaStoreFactoryRouting:
    """验证 VectorStoreFactory 能创建 ChromaStore。"""

    def test_factory_creates_chroma(self, chroma_temp_dir: str) -> None:
        """provider=chroma 时应返回 ChromaStore 实例。"""
        base = load_settings()
        settings = Settings(
            llm=base.llm,
            embedding=base.embedding,
            vector_store=VectorStoreSettings(
                provider="chroma",
                persist_directory=chroma_temp_dir,
                collection_name="factory_test",
            ),
            retrieval=base.retrieval,
            rerank=base.rerank,
            evaluation=base.evaluation,
            observability=base.observability,
            ingestion=base.ingestion,
            vision_llm=base.vision_llm,
        )
        store = VectorStoreFactory.create(settings)
        assert isinstance(store, ChromaStore)
