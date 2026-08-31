"""DocumentManager：跨 Chroma / BM25 / ImageStorage / FileIntegrity 的文档生命周期管理。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


class DocumentManagerError(Exception):
    """文档管理协调失败时抛出。"""


class _VectorStoreLike(Protocol):
    def get_by_metadata(
        self,
        filters: Mapping[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[dict[str, Any]]: ...

    def delete_by_metadata(
        self,
        filters: Mapping[str, Any],
        trace: Any | None = None,
    ) -> int: ...


class _BM25Like(Protocol):
    def remove_document(self, source: str, chunk_ids: Sequence[str] | None = None) -> None: ...

    def save(self) -> None: ...


class _ImageStorageLike(Protocol):
    def list_images(
        self,
        collection: str | None = None,
        doc_hash: str | None = None,
    ) -> Sequence[Any]: ...

    def delete_images(self, collection: str, doc_hash: str | None = None) -> int: ...


class _IntegrityLike(Protocol):
    def remove_record(self, file_hash: str) -> None: ...

    def list_processed(self) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class DocumentInfo:
    """已摄入文档的列表项。"""

    source_path: str
    collection: str
    chunk_count: int
    image_count: int
    doc_id: str | None = None
    processed_at: str | None = None


@dataclass(frozen=True)
class DocumentDetail:
    """单个文档的 chunk 与图片详情。"""

    info: DocumentInfo
    chunks: list[dict[str, Any]] = field(default_factory=list)
    images: list[Any] = field(default_factory=list)


@dataclass(frozen=True)
class DeleteResult:
    """跨存储删除结果摘要。"""

    source_path: str
    collection: str
    chroma_deleted: int
    bm25_removed: bool
    images_deleted: int
    integrity_removed: bool


@dataclass(frozen=True)
class CollectionStats:
    """集合级资产统计。"""

    collection: str | None
    document_count: int
    chunk_count: int
    image_count: int


class DocumentManager:
    """
    文档生命周期管理器，对应 G2。

    协调向量库、BM25、图片存储与摄取历史，供 Dashboard 浏览/删除。
    """

    def __init__(
        self,
        chroma_store: _VectorStoreLike,
        bm25_indexer: _BM25Like,
        image_storage: _ImageStorageLike,
        file_integrity: _IntegrityLike,
    ) -> None:
        self._chroma = chroma_store
        self._bm25 = bm25_indexer
        self._images = image_storage
        self._integrity = file_integrity

    def list_documents(self, collection: str | None = None) -> list[DocumentInfo]:
        """列出已摄入文档（source、chunk 数、图片数）。"""
        records = self._fetch_records(collection)
        grouped = self._group_by_source(records, collection)
        processed = self._processed_by_path()
        infos: list[DocumentInfo] = []
        for key, chunks in grouped.items():
            source_path, coll = key
            doc_hash = _first_doc_hash(chunks)
            image_count = len(self._images.list_images(collection=coll, doc_hash=doc_hash))
            history = processed.get(_normalize_path(source_path))
            infos.append(
                DocumentInfo(
                    source_path=source_path,
                    collection=coll,
                    chunk_count=len(chunks),
                    image_count=image_count,
                    doc_id=doc_hash,
                    processed_at=None if history is None else history.get("processed_at"),
                )
            )
        infos.sort(key=lambda item: (item.collection, item.source_path))
        return infos

    def get_document_detail(self, doc_id: str) -> DocumentDetail:
        """
        按 doc_id 取详情。

        doc_id 可以是 source_path 或 metadata.doc_hash / document_id。
        """
        if not doc_id or not doc_id.strip():
            raise DocumentManagerError("doc_id 不能为空")
        needle = doc_id.strip()
        records = self._chroma.get_by_metadata(None)
        matched = [
            item
            for item in records
            if _record_matches_doc_id(item, needle)
        ]
        if not matched:
            raise DocumentManagerError(f"未找到文档: {needle}")

        collection = str(matched[0].get("metadata", {}).get("collection") or "")
        infos = self._group_by_source(matched, collection or None)
        (source_path, coll), chunks = next(iter(infos.items()))
        doc_hash = _first_doc_hash(chunks)
        images = list(self._images.list_images(collection=coll or None, doc_hash=doc_hash))
        processed = self._processed_by_path()
        history = processed.get(_normalize_path(source_path))
        info = DocumentInfo(
            source_path=source_path,
            collection=coll,
            chunk_count=len(chunks),
            image_count=len(images),
            doc_id=doc_hash or needle,
            processed_at=None if history is None else history.get("processed_at"),
        )
        return DocumentDetail(info=info, chunks=chunks, images=images)

    def delete_document(self, source_path: str, collection: str) -> DeleteResult:
        """协调删除 Chroma、BM25、ImageStorage 与 FileIntegrity 中的同一文档。"""
        if not source_path or not source_path.strip():
            raise DocumentManagerError("source_path 不能为空")
        if not collection or not collection.strip():
            raise DocumentManagerError("collection 不能为空")

        source = source_path.strip()
        coll = collection.strip()
        filters = {"source_path": source, "collection": coll}
        records = self._chroma.get_by_metadata(filters)
        chunk_ids = [str(item.get("id", "")) for item in records if item.get("id")]
        doc_hash = _first_doc_hash(records)

        chroma_deleted = self._chroma.delete_by_metadata(filters)
        self._bm25.remove_document(source, chunk_ids=chunk_ids or None)
        if hasattr(self._bm25, "save"):
            self._bm25.save()
        images_deleted = self._images.delete_images(coll, doc_hash=doc_hash)

        integrity_removed = False
        file_hash = _first_file_hash(records, doc_hash)
        if file_hash:
            self._integrity.remove_record(file_hash)
            integrity_removed = True
        else:
            # 回退：按规范化路径匹配摄取历史
            target = _normalize_path(source)
            for row in self._integrity.list_processed():
                if _normalize_path(str(row.get("file_path", ""))) == target:
                    self._integrity.remove_record(str(row["file_hash"]))
                    integrity_removed = True
                    break

        return DeleteResult(
            source_path=source,
            collection=coll,
            chroma_deleted=chroma_deleted,
            bm25_removed=True,
            images_deleted=images_deleted,
            integrity_removed=integrity_removed,
        )

    def get_collection_stats(self, collection: str | None = None) -> CollectionStats:
        """汇总文档数、chunk 数与图片数。"""
        documents = self.list_documents(collection)
        chunk_count = sum(item.chunk_count for item in documents)
        image_count = sum(item.image_count for item in documents)
        return CollectionStats(
            collection=collection,
            document_count=len(documents),
            chunk_count=chunk_count,
            image_count=image_count,
        )

    def _fetch_records(self, collection: str | None) -> list[dict[str, Any]]:
        filters: dict[str, Any] | None = None
        if collection and collection.strip():
            filters = {"collection": collection.strip()}
        return self._chroma.get_by_metadata(filters)

    def _group_by_source(
        self,
        records: Sequence[Mapping[str, Any]],
        fallback_collection: str | None,
    ) -> dict[tuple[str, str], list[dict[str, Any]]]:
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for item in records:
            metadata = dict(item.get("metadata") or {})
            source = str(metadata.get("source_path") or metadata.get("source") or "").strip()
            if not source:
                continue
            coll = str(metadata.get("collection") or fallback_collection or "").strip()
            grouped.setdefault((source, coll), []).append(dict(item))
        return grouped

    def _processed_by_path(self) -> dict[str, dict[str, Any]]:
        mapping: dict[str, dict[str, Any]] = {}
        for row in self._integrity.list_processed():
            path = _normalize_path(str(row.get("file_path", "")))
            if path:
                mapping[path] = row
        return mapping


def _first_doc_hash(records: Sequence[Mapping[str, Any]]) -> str | None:
    for item in records:
        metadata = item.get("metadata") or {}
        value = metadata.get("doc_hash") or metadata.get("document_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _first_file_hash(records: Sequence[Mapping[str, Any]], doc_hash: str | None) -> str | None:
    for item in records:
        metadata = item.get("metadata") or {}
        value = metadata.get("file_hash")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return doc_hash


def _record_matches_doc_id(item: Mapping[str, Any], needle: str) -> bool:
    metadata = item.get("metadata") or {}
    candidates = [
        metadata.get("source_path"),
        metadata.get("source"),
        metadata.get("doc_hash"),
        metadata.get("document_id"),
        item.get("id"),
    ]
    normalized = _normalize_path(needle)
    for value in candidates:
        if not isinstance(value, str) or not value.strip():
            continue
        if value.strip() == needle or _normalize_path(value) == normalized:
            return True
    return False


def _normalize_path(path: str) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).resolve()).replace("\\", "/").lower()
    except OSError:
        return path.replace("\\", "/").lower()
