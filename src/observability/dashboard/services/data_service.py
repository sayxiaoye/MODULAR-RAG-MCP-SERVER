"""Dashboard 数据浏览服务：封装 ChromaStore / ImageStorage 读取（G3）。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from core.settings import Settings, load_settings
from ingestion.storage.image_storage import ImageStorage
from libs.loader.file_integrity import SQLiteIntegrityChecker
from libs.vector_store.vector_store_factory import VectorStoreFactory


class DataServiceError(Exception):
    """数据浏览服务读取存储失败时抛出。"""


class _ChromaLike(Protocol):
    def get_by_metadata(
        self,
        filters: Mapping[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[dict[str, Any]]: ...


class _ImageStorageLike(Protocol):
    def list_images(
        self,
        collection: str | None = None,
        doc_hash: str | None = None,
    ) -> Sequence[Any]: ...

    def get_path(self, image_id: str) -> Path | None: ...


class _IntegrityLike(Protocol):
    def list_processed(self) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class DocumentRow:
    """文档列表行：来源路径、集合、chunk 数与摄入时间。"""

    source_path: str
    collection: str
    chunk_count: int
    processed_at: str | None = None
    doc_id: str | None = None
    image_count: int = 0


@dataclass(frozen=True)
class ImagePreview:
    """关联图片预览：路径可能缺失（索引有记录但文件已删）。"""

    image_id: str
    file_path: str
    page_num: int | None = None
    exists: bool = False


@dataclass(frozen=True)
class ChunkView:
    """单个 chunk 的展示结构：正文、metadata 与关联图片。"""

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    images: list[ImagePreview] = field(default_factory=list)


class DataService:
    """
    数据浏览器的读取层，对应 G3 DataService。

    通过 ``ChromaStore.get_by_metadata()`` 拉取 chunk，再用
    ``ImageStorage.list_images()`` / ``get_path()`` 解析关联图片。
    不依赖 Trace；摄入时间来自 FileIntegrity 历史（可选）。
    """

    def __init__(
        self,
        chroma_store: _ChromaLike,
        image_storage: _ImageStorageLike,
        file_integrity: _IntegrityLike | None = None,
        default_collection: str | None = None,
    ) -> None:
        self._chroma = chroma_store
        self._images = image_storage
        self._integrity = file_integrity
        self._default_collection = (default_collection or "").strip() or None

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> DataService:
        """按全局 Settings 构造默认 Chroma / ImageStorage / 摄取历史读取器。"""
        resolved = settings or load_settings()
        chroma = VectorStoreFactory.create(resolved)
        return cls(
            chroma_store=chroma,
            image_storage=ImageStorage(),
            file_integrity=SQLiteIntegrityChecker(),
            default_collection=resolved.vector_store.collection_name,
        )

    def list_collections(self) -> list[str]:
        """从 chunk metadata 收集去重后的集合名，供筛选下拉框使用。"""
        records = self._chroma.get_by_metadata(None)
        names: set[str] = set()
        for item in records:
            metadata = item.get("metadata") or {}
            name = str(metadata.get("collection") or "").strip()
            if name:
                names.add(name)
        if self._default_collection:
            names.add(self._default_collection)
        return sorted(names)

    def list_documents(
        self,
        collection: str | None = None,
        keyword: str | None = None,
    ) -> list[DocumentRow]:
        """
        列出已摄入文档。

        Args:
            collection: 按 metadata.collection 筛选；空则不过滤。
            keyword: 对 source_path 做不区分大小写的子串匹配。

        Returns:
            按集合名、路径排序的文档行。
        """
        filters: dict[str, Any] | None = None
        if collection and collection.strip():
            filters = {"collection": collection.strip()}
        records = self._chroma.get_by_metadata(filters)
        grouped = _group_by_source(records, collection)
        processed = self._processed_by_path()
        needle = (keyword or "").strip().lower()

        rows: list[DocumentRow] = []
        for (source_path, coll), chunks in grouped.items():
            if needle and needle not in source_path.lower():
                continue
            doc_hash = _first_doc_hash(chunks)
            image_count = len(self._images.list_images(collection=coll or None, doc_hash=doc_hash))
            history = processed.get(_normalize_path(source_path))
            rows.append(
                DocumentRow(
                    source_path=source_path,
                    collection=coll,
                    chunk_count=len(chunks),
                    processed_at=None if history is None else history.get("processed_at"),
                    doc_id=doc_hash,
                    image_count=image_count,
                )
            )
        rows.sort(key=lambda item: (item.collection, item.source_path))
        return rows

    def get_chunks(self, source_path: str, collection: str) -> list[ChunkView]:
        """
        读取某文档的全部 chunk，并解析 metadata.image_refs 对应图片。

        Args:
            source_path: 文档来源路径（与 upsert 写入的 metadata.source_path 一致）。
            collection: 所属集合。

        Returns:
            按 chunk_index / id 排序的 ChunkView 列表。
        """
        if not source_path or not source_path.strip():
            raise DataServiceError("source_path 不能为空")
        if not collection or not collection.strip():
            raise DataServiceError("collection 不能为空")

        source = source_path.strip()
        coll = collection.strip()
        records = self._chroma.get_by_metadata({"source_path": source, "collection": coll})
        records.sort(key=_chunk_sort_key)

        views: list[ChunkView] = []
        for item in records:
            metadata = dict(item.get("metadata") or {})
            refs = _extract_image_refs(metadata)
            images = [self._preview_for(image_id, metadata) for image_id in refs]
            views.append(
                ChunkView(
                    chunk_id=str(item.get("id") or ""),
                    text=str(item.get("text") or ""),
                    metadata=metadata,
                    images=images,
                )
            )
        return views

    def list_images(
        self,
        collection: str | None = None,
        doc_hash: str | None = None,
    ) -> list[ImagePreview]:
        """封装 ImageStorage.list_images，并检查本地文件是否仍存在。"""
        records = self._images.list_images(collection=collection, doc_hash=doc_hash)
        previews: list[ImagePreview] = []
        for record in records:
            image_id, file_path, page_num = _unpack_image_record(record)
            if not image_id:
                continue
            path = Path(file_path) if file_path else None
            previews.append(
                ImagePreview(
                    image_id=image_id,
                    file_path=file_path,
                    page_num=page_num,
                    exists=bool(path is not None and path.is_file()),
                )
            )
        return previews

    def _preview_for(self, image_id: str, metadata: Mapping[str, Any]) -> ImagePreview:
        """优先用 metadata.images 中的 path，否则查 ImageStorage 索引。"""
        from_meta = _path_from_metadata_images(image_id, metadata)
        if from_meta is not None:
            return ImagePreview(
                image_id=image_id,
                file_path=str(from_meta),
                exists=from_meta.is_file(),
            )
        try:
            resolved = self._images.get_path(image_id)
        except Exception:
            resolved = None
        if resolved is not None:
            return ImagePreview(
                image_id=image_id,
                file_path=str(resolved),
                exists=resolved.is_file(),
            )
        return ImagePreview(image_id=image_id, file_path="", exists=False)

    def _processed_by_path(self) -> dict[str, dict[str, Any]]:
        if self._integrity is None:
            return {}
        mapping: dict[str, dict[str, Any]] = {}
        for row in self._integrity.list_processed():
            path = _normalize_path(str(row.get("file_path", "")))
            if path:
                mapping[path] = row
        return mapping


def _group_by_source(
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


def _first_doc_hash(records: Sequence[Mapping[str, Any]]) -> str | None:
    for item in records:
        metadata = item.get("metadata") or {}
        value = metadata.get("doc_hash") or metadata.get("document_id")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _chunk_sort_key(item: Mapping[str, Any]) -> tuple[int, str]:
    metadata = item.get("metadata") or {}
    index = metadata.get("chunk_index")
    try:
        numeric = int(index)
    except (TypeError, ValueError):
        numeric = 10**9
    return (numeric, str(item.get("id") or ""))


def _extract_image_refs(metadata: Mapping[str, Any]) -> list[str]:
    """解析 image_refs，兼容 list 与 Chroma 序列化后的 JSON 字符串。"""
    raw = metadata.get("image_refs")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return [raw.strip()] if raw.strip() else []
    if not isinstance(raw, list):
        return []
    refs: list[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            refs.append(item.strip())
    return refs


def _path_from_metadata_images(image_id: str, metadata: Mapping[str, Any]) -> Path | None:
    images = metadata.get("images")
    if isinstance(images, str):
        try:
            images = json.loads(images)
        except json.JSONDecodeError:
            return None
    if not isinstance(images, list):
        return None
    for item in images:
        if not isinstance(item, Mapping):
            continue
        if str(item.get("id") or "").strip() != image_id:
            continue
        path = str(item.get("path") or "").strip()
        return Path(path) if path else None
    return None


def _unpack_image_record(record: Any) -> tuple[str, str, int | None]:
    if isinstance(record, Mapping):
        page = record.get("page_num")
        return (
            str(record.get("image_id") or "").strip(),
            str(record.get("file_path") or ""),
            int(page) if isinstance(page, int) else None,
        )
    image_id = str(getattr(record, "image_id", "") or "").strip()
    file_path = str(getattr(record, "file_path", "") or "")
    page = getattr(record, "page_num", None)
    return (image_id, file_path, int(page) if isinstance(page, int) else None)


def _normalize_path(path: str) -> str:
    if not path:
        return ""
    try:
        return str(Path(path).resolve()).replace("\\", "/").lower()
    except OSError:
        return path.replace("\\", "/").lower()
