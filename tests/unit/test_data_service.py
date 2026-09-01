"""DataService 单元测试：文档列表、集合筛选、chunk 详情与图片解析。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import pytest

from observability.dashboard.services.data_service import DataService, DataServiceError


def _record(
    chunk_id: str,
    source_path: str,
    collection: str = "docs",
    *,
    text: str = "chunk text",
    doc_hash: str | None = "hash-a",
    chunk_index: int | None = None,
    image_refs: list[str] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source_path": source_path,
        "collection": collection,
    }
    if doc_hash:
        metadata["doc_hash"] = doc_hash
    if chunk_index is not None:
        metadata["chunk_index"] = chunk_index
    if image_refs is not None:
        metadata["image_refs"] = image_refs
    return {"id": chunk_id, "text": text, "metadata": metadata}


class FakeChroma:
    """内存向量库替身，支持按 metadata 等值过滤。"""

    def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
        self.records = list(records or [])

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


class FakeImageStorage:
    """记录 list_images / get_path 调用，返回预设索引。"""

    def __init__(self, images: list[dict[str, Any]] | None = None) -> None:
        self.images = list(images or [])
        self.paths: dict[str, Path] = {}

    def list_images(
        self,
        collection: str | None = None,
        doc_hash: str | None = None,
    ) -> Sequence[dict[str, Any]]:
        result = self.images
        if collection is not None:
            result = [item for item in result if item.get("collection") == collection]
        if doc_hash is not None:
            result = [item for item in result if item.get("doc_hash") == doc_hash]
        return result

    def get_path(self, image_id: str) -> Path | None:
        return self.paths.get(image_id)


class FakeIntegrity:
    """返回预设摄取历史，供 list_documents 填 processed_at。"""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = list(rows or [])

    def list_processed(self) -> list[dict[str, Any]]:
        return list(self.rows)


@pytest.mark.unit
class TestDataService:
    """验证 DataService 对 Chroma / ImageStorage 的封装行为。"""

    def test_list_documents_groups_chunks_and_attach_ingest_time(self, tmp_path: Path) -> None:
        """同一 source_path 的 chunk 应合并为一行，并带上摄取时间与图片数。"""
        source = str(tmp_path / "a.pdf")
        chroma = FakeChroma(
            [
                _record("c1", source, chunk_index=0),
                _record("c2", source, chunk_index=1),
                _record("c3", str(tmp_path / "b.pdf"), doc_hash="hash-b"),
            ]
        )
        images = FakeImageStorage(
            [
                {"image_id": "img-1", "file_path": "/x.png", "collection": "docs", "doc_hash": "hash-a"},
            ]
        )
        integrity = FakeIntegrity(
            [{"file_path": source, "processed_at": "2026-08-31 10:00:00"}]
        )
        service = DataService(chroma, images, integrity)

        rows = service.list_documents()
        assert len(rows) == 2
        first = next(item for item in rows if item.source_path == source)
        assert first.chunk_count == 2
        assert first.image_count == 1
        assert first.processed_at == "2026-08-31 10:00:00"
        assert first.doc_id == "hash-a"

    def test_list_documents_filters_by_collection_and_keyword(self, tmp_path: Path) -> None:
        """集合筛选与 source_path 关键词应同时生效。"""
        chroma = FakeChroma(
            [
                _record("c1", str(tmp_path / "alpha.pdf"), "kb"),
                _record("c2", str(tmp_path / "beta.pdf"), "kb"),
                _record("c3", str(tmp_path / "alpha.pdf"), "other"),
            ]
        )
        service = DataService(chroma, FakeImageStorage())
        rows = service.list_documents(collection="kb", keyword="alpha")
        assert len(rows) == 1
        assert rows[0].collection == "kb"
        assert "alpha.pdf" in rows[0].source_path

    def test_list_collections_includes_default_and_metadata(self) -> None:
        """集合下拉应包含 chunk metadata 中的名称与默认集合。"""
        chroma = FakeChroma([_record("c1", "/a.pdf", "kb")])
        service = DataService(chroma, FakeImageStorage(), default_collection="knowledge_hub")
        assert service.list_collections() == ["kb", "knowledge_hub"]

    def test_get_chunks_sorts_and_resolves_image_refs(self, tmp_path: Path) -> None:
        """chunk 应按 chunk_index 排序，image_refs 通过 ImageStorage.get_path 解析。"""
        source = str(tmp_path / "doc.pdf")
        image_file = tmp_path / "fig.png"
        image_file.write_bytes(b"png")
        chroma = FakeChroma(
            [
                _record("c2", source, text="second", chunk_index=1),
                _record("c1", source, text="first", chunk_index=0, image_refs=["img-1"]),
            ]
        )
        images = FakeImageStorage()
        images.paths["img-1"] = image_file
        service = DataService(chroma, images)

        chunks = service.get_chunks(source, "docs")
        assert [item.chunk_id for item in chunks] == ["c1", "c2"]
        assert chunks[0].text == "first"
        assert chunks[0].images[0].image_id == "img-1"
        assert chunks[0].images[0].exists is True
        assert chunks[1].images == []

    def test_get_chunks_parses_json_image_refs(self, tmp_path: Path) -> None:
        """Chroma 把 list 序列化成 JSON 字符串后仍应解析出 image_id。"""
        source = str(tmp_path / "doc.pdf")
        chroma = FakeChroma(
            [
                {
                    "id": "c1",
                    "text": "body",
                    "metadata": {
                        "source_path": source,
                        "collection": "docs",
                        "image_refs": '["img-json"]',
                    },
                }
            ]
        )
        service = DataService(chroma, FakeImageStorage())
        chunks = service.get_chunks(source, "docs")
        assert chunks[0].images[0].image_id == "img-json"
        assert chunks[0].images[0].exists is False

    def test_get_chunks_rejects_empty_identity(self) -> None:
        """缺少 source_path 或 collection 时应抛出 DataServiceError。"""
        service = DataService(FakeChroma(), FakeImageStorage())
        with pytest.raises(DataServiceError):
            service.get_chunks("", "docs")
        with pytest.raises(DataServiceError):
            service.get_chunks("/a.pdf", "")

    def test_list_images_marks_missing_files(self, tmp_path: Path) -> None:
        """索引有路径但文件不存在时 exists 应为 False。"""
        existing = tmp_path / "ok.png"
        existing.write_bytes(b"png")
        images = FakeImageStorage(
            [
                {"image_id": "ok", "file_path": str(existing), "collection": "docs", "doc_hash": "h"},
                {"image_id": "gone", "file_path": str(tmp_path / "missing.png"), "collection": "docs", "doc_hash": "h"},
            ]
        )
        service = DataService(FakeChroma(), images)
        previews = service.list_images(collection="docs", doc_hash="h")
        by_id = {item.image_id: item for item in previews}
        assert by_id["ok"].exists is True
        assert by_id["gone"].exists is False
