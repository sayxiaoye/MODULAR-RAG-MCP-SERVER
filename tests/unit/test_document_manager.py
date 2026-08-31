"""DocumentManager 单元测试：列表、详情、跨存储删除与集合统计。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pytest

from ingestion.document_manager import DocumentManager, DocumentManagerError
from ingestion.embedding.sparse_encoder import SparseChunkStats
from ingestion.storage.bm25_indexer import BM25Indexer


def _record(
    chunk_id: str,
    source_path: str,
    collection: str = "docs",
    *,
    doc_hash: str | None = "hash-a",
    text: str = "chunk text",
    file_hash: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source_path": source_path,
        "collection": collection,
    }
    if doc_hash:
        metadata["doc_hash"] = doc_hash
    if file_hash:
        metadata["file_hash"] = file_hash
    return {"id": chunk_id, "text": text, "metadata": metadata}


class FakeChroma:
    """内存向量库替身，支持按 metadata 查询与删除。"""

    def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
        self.records = list(records or [])

    def get_by_metadata(
        self,
        filters: Mapping[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[dict[str, Any]]:
        if not filters:
            return list(self.records)
        return [
            item
            for item in self.records
            if all(item.get("metadata", {}).get(key) == value for key, value in filters.items())
        ]

    def delete_by_metadata(
        self,
        filters: Mapping[str, Any],
        trace: Any | None = None,
    ) -> int:
        remaining: list[dict[str, Any]] = []
        deleted = 0
        for item in self.records:
            if all(item.get("metadata", {}).get(key) == value for key, value in filters.items()):
                deleted += 1
            else:
                remaining.append(item)
        self.records = remaining
        return deleted


class FakeBM25:
    """记录 remove_document 调用，便于断言协调删除。"""

    def __init__(self) -> None:
        self.removed: list[tuple[str, list[str] | None]] = []
        self.saved = False

    def remove_document(self, source: str, chunk_ids: Sequence[str] | None = None) -> None:
        self.removed.append((source, list(chunk_ids) if chunk_ids is not None else None))

    def save(self) -> None:
        self.saved = True


class FakeImageStorage:
    def __init__(self) -> None:
        self.images: dict[tuple[str | None, str | None], list[str]] = {}
        self.deleted: list[tuple[str, str | None]] = []

    def list_images(self, collection: str | None = None, doc_hash: str | None = None) -> list[str]:
        return list(self.images.get((collection, doc_hash), []))

    def delete_images(self, collection: str, doc_hash: str | None = None) -> int:
        self.deleted.append((collection, doc_hash))
        key = (collection, doc_hash)
        count = len(self.images.pop(key, []))
        return count


class FakeIntegrity:
    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = list(rows or [])
        self.removed: list[str] = []

    def remove_record(self, file_hash: str) -> None:
        self.removed.append(file_hash)
        self.rows = [row for row in self.rows if row.get("file_hash") != file_hash]

    def list_processed(self) -> list[dict[str, Any]]:
        return list(self.rows)


@pytest.mark.unit
class TestDocumentManager:
    """验证 DocumentManager 的列表、删除与统计契约。"""

    def test_list_documents_returns_source_chunk_and_image_counts(self) -> None:
        """同一 source 的多个 chunk 应合并，并统计关联图片。"""
        chroma = FakeChroma(
            [
                _record("c1", "a.pdf", doc_hash="h1"),
                _record("c2", "a.pdf", doc_hash="h1"),
                _record("c3", "b.pdf", collection="other", doc_hash="h2"),
            ]
        )
        images = FakeImageStorage()
        images.images[("docs", "h1")] = ["img-1", "img-2"]
        manager = DocumentManager(chroma, FakeBM25(), images, FakeIntegrity())

        listed = manager.list_documents("docs")
        assert len(listed) == 1
        assert listed[0].source_path == "a.pdf"
        assert listed[0].chunk_count == 2
        assert listed[0].image_count == 2
        assert listed[0].doc_id == "h1"

    def test_delete_document_coordinates_four_stores(self) -> None:
        """删除应同时清理 Chroma、BM25、图片与摄取历史。"""
        chroma = FakeChroma(
            [
                _record("c1", "a.pdf", doc_hash="h1", file_hash="sha-a"),
                _record("c2", "a.pdf", doc_hash="h1", file_hash="sha-a"),
                _record("keep", "b.pdf", doc_hash="h2", file_hash="sha-b"),
            ]
        )
        bm25 = FakeBM25()
        images = FakeImageStorage()
        images.images[("docs", "h1")] = ["img-1"]
        integrity = FakeIntegrity([{"file_hash": "sha-a", "file_path": "a.pdf"}])
        manager = DocumentManager(chroma, bm25, images, integrity)

        result = manager.delete_document("a.pdf", "docs")

        assert result.chroma_deleted == 2
        assert result.images_deleted == 1
        assert result.bm25_removed is True
        assert result.integrity_removed is True
        assert bm25.removed == [("a.pdf", ["c1", "c2"])]
        assert bm25.saved is True
        assert images.deleted == [("docs", "h1")]
        assert integrity.removed == ["sha-a"]
        remaining = manager.list_documents("docs")
        assert [item.source_path for item in remaining] == ["b.pdf"]

    def test_delete_falls_back_to_integrity_file_path(self) -> None:
        """chunk 无 file_hash 时，应按规范化路径匹配摄取历史。"""
        chroma = FakeChroma([_record("c1", r"D:\docs\a.pdf", doc_hash=None)])
        integrity = FakeIntegrity(
            [{"file_hash": "sha-path", "file_path": r"D:\docs\a.pdf"}]
        )
        manager = DocumentManager(chroma, FakeBM25(), FakeImageStorage(), integrity)

        result = manager.delete_document(r"D:\docs\a.pdf", "docs")
        assert result.integrity_removed is True
        assert integrity.removed == ["sha-path"]

    def test_get_document_detail_by_doc_hash(self) -> None:
        """doc_id 为 doc_hash 时应返回该文档全部 chunk。"""
        chroma = FakeChroma(
            [
                _record("c1", "a.pdf", doc_hash="h1", text="one"),
                _record("c2", "a.pdf", doc_hash="h1", text="two"),
            ]
        )
        manager = DocumentManager(chroma, FakeBM25(), FakeImageStorage(), FakeIntegrity())
        detail = manager.get_document_detail("h1")
        assert detail.info.source_path == "a.pdf"
        assert [item["id"] for item in detail.chunks] == ["c1", "c2"]

    def test_get_document_detail_unknown_raises(self) -> None:
        """未知 doc_id 应抛出 DocumentManagerError。"""
        manager = DocumentManager(FakeChroma(), FakeBM25(), FakeImageStorage(), FakeIntegrity())
        with pytest.raises(DocumentManagerError, match="未找到"):
            manager.get_document_detail("missing")

    def test_collection_stats_sums_documents(self) -> None:
        """集合统计应汇总文档数、chunk 数与图片数。"""
        chroma = FakeChroma(
            [
                _record("c1", "a.pdf", doc_hash="h1"),
                _record("c2", "a.pdf", doc_hash="h1"),
                _record("c3", "b.pdf", doc_hash="h2"),
            ]
        )
        images = FakeImageStorage()
        images.images[("docs", "h1")] = ["i1"]
        images.images[("docs", "h2")] = ["i2", "i3"]
        manager = DocumentManager(chroma, FakeBM25(), images, FakeIntegrity())

        stats = manager.get_collection_stats("docs")
        assert stats.document_count == 2
        assert stats.chunk_count == 3
        assert stats.image_count == 3

    def test_bm25_remove_document_drops_chunk_ids(self, tmp_path) -> None:
        """真实 BM25Indexer.remove_document 应按 chunk_ids 移除条目。"""
        indexer = BM25Indexer(collection="docs", index_root=tmp_path)
        indexer.add(
            [
                SparseChunkStats("c1", {"azure": 1}, 1),
                SparseChunkStats("c2", {"guide": 1}, 1),
            ]
        )
        indexer.remove_document("a.pdf", chunk_ids=["c1"])
        assert "c1" not in indexer._doc_lengths
        assert "c2" in indexer._doc_lengths
