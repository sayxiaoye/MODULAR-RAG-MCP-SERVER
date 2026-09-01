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

    def test_upsert_and_query_are_scoped_to_collection(self, tmp_path) -> None:
        """同一 PersistentClient 下，写入 col_a 不应出现在 col_b 的检索结果中。"""
        store = ChromaStore(_settings(tmp_path, "col_a"))
        store.upsert(
            [
                {
                    "id": "only-a",
                    "text": "alpha",
                    "metadata": {"source_path": "a.pdf"},
                    "dense_vector": [1.0, 0.0],
                }
            ],
            collection="col_a",
        )
        assert store.query([1.0, 0.0], top_k=3, collection="col_b") == []
        hits = store.query([1.0, 0.0], top_k=3, collection="col_a")
        assert [item["id"] for item in hits] == ["only-a"]
        assert store.list_collection_names() == ["col_a"]

    def test_delete_collection_removes_empty_shell(self, tmp_path) -> None:
        """delete_collection 后集合名不应再出现在 list 中。"""
        store = ChromaStore(_settings(tmp_path, "empty_col"))
        store.upsert(
            [
                {
                    "id": "c1",
                    "text": "一段",
                    "metadata": {"source_path": "a.pdf", "collection": "empty_col"},
                    "dense_vector": [1.0, 0.0],
                }
            ],
            collection="empty_col",
        )
        deleted = store.delete_by_metadata(
            {"source_path": "a.pdf"},
            collection="empty_col",
        )
        assert deleted == 1
        assert store.list_collection_names() == ["empty_col"]
        store.delete_collection("empty_col")
        assert store.list_collection_names() == []
        store.delete_collection("empty_col")
        assert store.list_collection_names() == []

    def test_cleanup_removes_unreferenced_uuid_dir(self, tmp_path) -> None:
        """sqlite 未引用的 UUID 目录应被 cleanup 删除，不影响仍在用的集合。"""
        store = ChromaStore(_settings(tmp_path, "keep_col"))
        store.upsert(
            [
                {
                    "id": "keep-1",
                    "text": "保留",
                    "metadata": {"source_path": "keep.pdf"},
                    "dense_vector": [1.0, 0.0],
                }
            ],
            collection="keep_col",
        )
        root = tmp_path / "chroma"
        orphan = root / "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        orphan.mkdir()
        (orphan / "data_level0.bin").write_bytes(b"orphan")
        removed = store.cleanup_orphan_segment_dirs()
        assert "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee" in removed
        assert not orphan.exists()
        assert store.list_collection_names() == ["keep_col"]
        hits = store.query([1.0, 0.0], top_k=1, collection="keep_col")
        assert [item["id"] for item in hits] == ["keep-1"]

    def test_delete_collection_sweeps_orphans_when_unlocked(self, tmp_path) -> None:
        """delete_collection 会触发孤儿清理（未锁定的目录应被删掉）。"""
        store = ChromaStore(_settings(tmp_path, "keep_col"))
        store.upsert(
            [
                {
                    "id": "keep-1",
                    "text": "保留",
                    "metadata": {"source_path": "keep.pdf"},
                    "dense_vector": [1.0, 0.0],
                }
            ],
            collection="keep_col",
        )
        orphan = tmp_path / "chroma" / "11111111-2222-3333-4444-555555555555"
        orphan.mkdir()
        (orphan / "header.bin").write_bytes(b"x")
        store.delete_collection("missing_col")
        assert not orphan.exists()
        assert store.list_collection_names() == ["keep_col"]
