"""ChromaStore.get_collection_stats 单元测试：统计 chunk 与去重文档数。"""

from __future__ import annotations

import pytest

from core.settings import VectorStoreSettings
from libs.vector_store.chroma_store import ChromaStore


def _settings(tmp_path, name: str = "stats_col") -> VectorStoreSettings:
    return VectorStoreSettings(
        provider="chroma",
        persist_directory=str(tmp_path / "chroma"),
        collection_name=name,
    )


@pytest.mark.unit
class TestChromaCollectionStats:
    """验证集合统计在空库与写入后的行为。"""

    def test_empty_collection_returns_zeros(self, tmp_path) -> None:
        """空集合的文档数与 chunk 数应为 0。"""
        store = ChromaStore(_settings(tmp_path))
        stats = store.get_collection_stats()
        assert stats.collection == "stats_col"
        assert stats.chunk_count == 0
        assert stats.document_count == 0

    def test_counts_unique_source_paths(self, tmp_path) -> None:
        """同一文档的多个 chunk 应计为 1 个文档。"""
        store = ChromaStore(_settings(tmp_path, "docs"))
        store.upsert(
            [
                {
                    "id": "c1",
                    "text": "一段",
                    "metadata": {"source_path": "a.pdf"},
                    "dense_vector": [1.0, 0.0],
                },
                {
                    "id": "c2",
                    "text": "二段",
                    "metadata": {"source_path": "a.pdf"},
                    "dense_vector": [0.9, 0.1],
                },
                {
                    "id": "c3",
                    "text": "另一篇",
                    "metadata": {"source_path": "b.pdf"},
                    "dense_vector": [0.0, 1.0],
                },
            ]
        )
        stats = store.get_collection_stats("docs")
        assert stats.chunk_count == 3
        assert stats.document_count == 2
