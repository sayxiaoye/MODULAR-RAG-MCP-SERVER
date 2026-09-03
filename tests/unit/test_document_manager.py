"""DocumentManager 单元测试：列表、详情、跨存储删除与集合统计。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pytest

from ingestion.document_manager import DocumentManager, DocumentManagerError
from ingestion.embedding.sparse_encoder import SparseChunkStats
from ingestion.storage.bm25_indexer import BM25Indexer


@pytest.fixture(autouse=True)
def _mute_delete_trace_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """默认删除 trace 不写入仓库 traces.jsonl；需要断言时传入 trace_writer。"""
    monkeypatch.setattr("observability.logger.write_trace", lambda *args, **kwargs: None)


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
        self.deleted_collections: list[str] = []

    def get_by_metadata(
        self,
        filters: Mapping[str, Any] | None = None,
        trace: Any | None = None,
        *,
        collection: str | None = None,
    ) -> list[dict[str, Any]]:
        records = list(self.records)
        if collection:
            records = [
                item
                for item in records
                if str(item.get("metadata", {}).get("collection") or "") == collection
            ]
        if not filters:
            return records
        return [
            item
            for item in records
            if all(item.get("metadata", {}).get(key) == value for key, value in filters.items())
        ]

    def delete_by_metadata(
        self,
        filters: Mapping[str, Any],
        trace: Any | None = None,
        *,
        collection: str | None = None,
    ) -> int:
        remaining: list[dict[str, Any]] = []
        deleted = 0
        scoped = list(self.records)
        if collection:
            scoped = [
                item
                for item in scoped
                if str(item.get("metadata", {}).get("collection") or "") == collection
            ]
        scoped_ids = {item.get("id") for item in scoped}
        for item in self.records:
            if item.get("id") not in scoped_ids:
                remaining.append(item)
                continue
            if all(item.get("metadata", {}).get(key) == value for key, value in filters.items()):
                deleted += 1
            else:
                remaining.append(item)
        self.records = remaining
        return deleted

    def delete_collection(self, collection: str, trace: Any | None = None) -> None:
        self.deleted_collections.append(collection)
        name = (collection or "").strip()
        self.records = [
            item
            for item in self.records
            if str(item.get("metadata", {}).get("collection") or "") != name
        ]


class FakeBM25:
    """记录 remove_document 调用，便于断言协调删除。"""

    def __init__(self) -> None:
        self.removed: list[tuple[str, list[str] | None]] = []
        self.saved = False
        self.last_doc_hash: str | None = None

    def remove_document(
        self,
        source: str,
        chunk_ids: Sequence[str] | None = None,
        *,
        doc_hash: str | None = None,
    ) -> None:
        self.removed.append((source, list(chunk_ids) if chunk_ids is not None else None))
        self.last_doc_hash = doc_hash

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

    def remove_record(self, file_hash: str, collection: str | None = None) -> None:
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
        assert bm25.last_doc_hash == "h1"
        assert bm25.saved is True
        assert images.deleted == [("docs", "h1")]
        assert integrity.removed == ["sha-a"]
        remaining = manager.list_documents("docs")
        assert [item.source_path for item in remaining] == ["b.pdf"]
        assert chroma.deleted_collections == []

    def test_delete_document_writes_deleted_trace(self) -> None:
        """删除文档应写入 ingestion trace，阶段名为 deleted。"""
        captured: list[dict[str, Any]] = []
        chroma = FakeChroma(
            [_record("c1", "a.pdf", doc_hash="h1", file_hash="sha-a")]
        )
        manager = DocumentManager(
            chroma,
            FakeBM25(),
            FakeImageStorage(),
            FakeIntegrity(),
            trace_writer=captured.append,
        )
        manager.delete_document("a.pdf", "docs")
        assert len(captured) == 1
        payload = captured[0]
        assert payload["trace_type"] == "ingestion"
        assert payload["finished_at"]
        stages = payload["stages"]
        assert stages[0]["name"] == "deleted"
        assert stages[0]["source_path"] == "a.pdf"
        assert stages[0]["collection"] == "docs"
        assert stages[0]["chunk_count"] == 1

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

    def test_bm25_remove_document_by_doc_hash_prefix(self, tmp_path) -> None:
        """Chroma 稳定 id 对不上时，应按 SparseEncoder 的 doc_hash 前缀清 posting。"""
        doc_hash = "95058c505d940402e473ee4cc785ea12d0bdba952a976e119b3587e75ec2351f"
        bm25_id = f"{doc_hash}_0000_78883563"
        keep_id = "otherdoc_0000_aaaaaaaa"
        indexer = BM25Indexer(collection="docs", index_root=tmp_path)
        indexer.add(
            [
                SparseChunkStats(bm25_id, {"深": 1}, 1),
                SparseChunkStats(keep_id, {"guide": 1}, 1),
            ]
        )
        indexer.remove_document("a.pdf", chunk_ids=["chroma-stable-id"], doc_hash=doc_hash)
        assert bm25_id not in indexer._doc_lengths
        assert keep_id in indexer._doc_lengths

    def test_bm25_save_deletes_empty_index_file(self, tmp_path) -> None:
        """集合清空后 save 应删除 bm25 json，而不是留下空文件。"""
        indexer = BM25Indexer(collection="col_x", index_root=tmp_path)
        indexer.add([SparseChunkStats("c1", {"深": 1}, 1)])
        indexer.save()
        path = tmp_path / "col_x.json"
        assert path.is_file()
        indexer.remove_document("a.pdf", chunk_ids=["c1"])
        indexer.save()
        assert not path.exists()

    def test_delete_drops_orphan_bm25_when_collection_empty(self, tmp_path) -> None:
        """Chroma 已删光时，即使 id 不一致也应删除孤儿 BM25 文件。"""
        doc_hash = "95058c505d940402e473ee4cc785ea12d0bdba952a976e119b3587e75ec2351f"
        bm25_id = f"{doc_hash}_0000_78883563"
        chroma_id = "06d8e83bc14709759edda613280c9c272d114b38e8e3be89c1c96b176112d708"
        indexer = BM25Indexer(collection="col_x", index_root=tmp_path)
        indexer.add([SparseChunkStats(bm25_id, {"深": 1}, 1)])
        indexer.save()

        chroma = FakeChroma(
            [_record(chroma_id, "a.pdf", collection="col_x", doc_hash=doc_hash)]
        )
        manager = DocumentManager(chroma, indexer, FakeImageStorage(), FakeIntegrity())
        manager.delete_document("a.pdf", "col_x")
        assert not (tmp_path / "col_x.json").exists()
        assert chroma.deleted_collections == ["col_x"]

    def test_delete_last_document_drops_chroma_collection(self) -> None:
        """删光集合内最后一篇文档时应 drop 空的 Chroma collection。"""
        chroma = FakeChroma(
            [_record("c1", "a.pdf", collection="col_x", doc_hash="h1")]
        )
        manager = DocumentManager(chroma, FakeBM25(), FakeImageStorage(), FakeIntegrity())
        manager.delete_document("a.pdf", "col_x")
        assert chroma.deleted_collections == ["col_x"]
        assert chroma.records == []

    def test_delete_uses_integrity_hash_when_chroma_empty(self) -> None:
        """Chroma 已空时，应从摄取历史回退 file_hash 给 BM25。"""
        chroma = FakeChroma()
        bm25 = FakeBM25()
        integrity = FakeIntegrity([{"file_hash": "sha-a", "file_path": "a.pdf"}])
        manager = DocumentManager(chroma, bm25, FakeImageStorage(), integrity)

        manager.delete_document("a.pdf", "docs")
        assert bm25.last_doc_hash == "sha-a"
        assert bm25.removed == [("a.pdf", None)]

    def test_delete_blank_source_path_raises(self) -> None:
        """空 source_path 禁止删除，避免误清整库。"""
        manager = DocumentManager(FakeChroma(), FakeBM25(), FakeImageStorage(), FakeIntegrity())
        with pytest.raises(DocumentManagerError, match="source_path"):
            manager.delete_document("  ", "docs")

    def test_delete_blank_collection_raises(self) -> None:
        """空 collection 禁止删除，避免跨集合误删。"""
        manager = DocumentManager(FakeChroma(), FakeBM25(), FakeImageStorage(), FakeIntegrity())
        with pytest.raises(DocumentManagerError, match="collection"):
            manager.delete_document("a.pdf", "")
