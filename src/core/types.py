"""全链路核心数据契约：Document / Chunk / ChunkRecord 及多模态图片元数据。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


class TypesError(ValueError):
    """核心类型字段校验或序列化失败时抛出。"""


IMAGE_PLACEHOLDER_TEMPLATE = "[IMAGE: {image_id}]"


def format_image_placeholder(image_id: str) -> str:
    """生成 Document.text 中的图片占位符，格式为 ``[IMAGE: {image_id}]``。"""
    if not isinstance(image_id, str) or not image_id.strip():
        raise TypesError("image_id 必须是非空字符串")
    return IMAGE_PLACEHOLDER_TEMPLATE.format(image_id=image_id.strip())


def _require_mapping(data: Mapping[str, Any], label: str) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise TypesError(f"{label} 必须是 mapping")
    return dict(data)


def _require_str_field(data: Mapping[str, Any], key: str, label: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TypesError(f"{label} 缺少或非法字段 {key}")
    return value.strip()


def _require_int_field(data: Mapping[str, Any], key: str, label: str) -> int:
    value = data.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypesError(f"{label} 缺少或非法字段 {key}")
    return value


@dataclass(frozen=True)
class ImageMetadata:
    """文档内单张图片的元数据，对应 ``metadata.images`` 列表元素。"""

    id: str
    path: str
    text_offset: int
    text_length: int
    page: int | None = None
    position: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "path": self.path,
            "text_offset": self.text_offset,
            "text_length": self.text_length,
        }
        if self.page is not None:
            payload["page"] = self.page
        if self.position is not None:
            payload["position"] = dict(self.position)
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ImageMetadata:
        label = "ImageMetadata"
        image_id = _require_str_field(data, "id", label)
        path = _require_str_field(data, "path", label)
        text_offset = _require_int_field(data, "text_offset", label)
        text_length = _require_int_field(data, "text_length", label)
        page = data.get("page")
        if page is not None and (not isinstance(page, int) or isinstance(page, bool)):
            raise TypesError(f"{label} 字段 page 必须是整数或 None")
        position = data.get("position")
        if position is not None and not isinstance(position, Mapping):
            raise TypesError(f"{label} 字段 position 必须是 mapping 或 None")
        return cls(
            id=image_id,
            path=path,
            text_offset=text_offset,
            text_length=text_length,
            page=page,
            position=dict(position) if position is not None else None,
        )


def _validate_source_path(metadata: Mapping[str, Any], label: str) -> None:
    """metadata 至少包含 source_path，供 ingestion/retrieval 溯源。"""
    if "source_path" not in metadata:
        raise TypesError(f"{label}.metadata 必须包含 source_path")
    source_path = metadata["source_path"]
    if not isinstance(source_path, str) or not source_path.strip():
        raise TypesError(f"{label}.metadata.source_path 必须是非空字符串")


def _normalize_images(metadata: dict[str, Any], label: str) -> dict[str, Any]:
    """将 metadata.images 规范化为可 JSON 序列化的 dict 列表。"""
    images = metadata.get("images")
    if images is None:
        return metadata
    if not isinstance(images, list):
        raise TypesError(f"{label}.metadata.images 必须是列表")
    normalized_images: list[dict[str, Any]] = []
    for index, item in enumerate(images):
        if isinstance(item, ImageMetadata):
            normalized_images.append(item.to_dict())
        elif isinstance(item, Mapping):
            normalized_images.append(ImageMetadata.from_dict(item).to_dict())
        else:
            raise TypesError(f"{label}.metadata.images[{index}] 必须是 ImageMetadata 或 mapping")
    metadata["images"] = normalized_images
    return metadata


@dataclass(frozen=True)
class Document:
    """Loader 产出的标准文档对象：规范化 Markdown 文本 + 元数据。"""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """校验 Document 契约字段。"""
        if not isinstance(self.id, str) or not self.id.strip():
            raise TypesError("Document.id 必须是非空字符串")
        if not isinstance(self.text, str):
            raise TypesError("Document.text 必须是字符串")
        _validate_source_path(self.metadata, "Document")
        _normalize_images(dict(self.metadata), "Document")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        metadata = _normalize_images(dict(self.metadata), "Document")
        return {
            "id": self.id,
            "text": self.text,
            "metadata": metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Document:
        payload = _require_mapping(data, "Document")
        doc = cls(
            id=_require_str_field(payload, "id", "Document"),
            text=str(payload.get("text", "")),
            metadata=_require_mapping(payload.get("metadata", {}), "Document.metadata"),
        )
        doc.validate()
        return doc


@dataclass(frozen=True)
class Chunk:
    """Splitter 产出的文本片段，携带定位与溯源信息。"""

    id: str
    text: str
    metadata: dict[str, Any]
    start_offset: int
    end_offset: int
    source_ref: str | None = None

    def validate(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise TypesError("Chunk.id 必须是非空字符串")
        if not isinstance(self.text, str):
            raise TypesError("Chunk.text 必须是字符串")
        _validate_source_path(self.metadata, "Chunk")
        if not isinstance(self.start_offset, int) or isinstance(self.start_offset, bool):
            raise TypesError("Chunk.start_offset 必须是整数")
        if not isinstance(self.end_offset, int) or isinstance(self.end_offset, bool):
            raise TypesError("Chunk.end_offset 必须是整数")
        if self.end_offset < self.start_offset:
            raise TypesError("Chunk.end_offset 不能小于 start_offset")
        if self.source_ref is not None and (
            not isinstance(self.source_ref, str) or not self.source_ref.strip()
        ):
            raise TypesError("Chunk.source_ref 必须是非空字符串或 None")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload: dict[str, Any] = {
            "id": self.id,
            "text": self.text,
            "metadata": dict(self.metadata),
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
        }
        if self.source_ref is not None:
            payload["source_ref"] = self.source_ref
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Chunk:
        payload = _require_mapping(data, "Chunk")
        chunk = cls(
            id=_require_str_field(payload, "id", "Chunk"),
            text=str(payload.get("text", "")),
            metadata=_require_mapping(payload.get("metadata", {}), "Chunk.metadata"),
            start_offset=_require_int_field(payload, "start_offset", "Chunk"),
            end_offset=_require_int_field(payload, "end_offset", "Chunk"),
            source_ref=payload.get("source_ref"),
        )
        chunk.validate()
        return chunk


@dataclass(frozen=True)
class ChunkRecord:
    """存储与检索载体：文本 + metadata + 可选稠密/稀疏向量。"""

    id: str
    text: str
    metadata: dict[str, Any]
    dense_vector: list[float] | None = None
    sparse_vector: dict[str, float] | None = None

    def validate(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise TypesError("ChunkRecord.id 必须是非空字符串")
        if not isinstance(self.text, str):
            raise TypesError("ChunkRecord.text 必须是字符串")
        _validate_source_path(self.metadata, "ChunkRecord")
        if self.dense_vector is not None:
            if not isinstance(self.dense_vector, list):
                raise TypesError("ChunkRecord.dense_vector 必须是 list 或 None")
            for index, value in enumerate(self.dense_vector):
                if not isinstance(value, (int, float)):
                    raise TypesError(f"ChunkRecord.dense_vector[{index}] 必须是数值")
        if self.sparse_vector is not None:
            if not isinstance(self.sparse_vector, dict):
                raise TypesError("ChunkRecord.sparse_vector 必须是 dict 或 None")
            for key, value in self.sparse_vector.items():
                if not isinstance(key, str):
                    raise TypesError("ChunkRecord.sparse_vector 的键必须是字符串")
                if not isinstance(value, (int, float)):
                    raise TypesError(f"ChunkRecord.sparse_vector[{key}] 必须是数值")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload: dict[str, Any] = {
            "id": self.id,
            "text": self.text,
            "metadata": dict(self.metadata),
        }
        if self.dense_vector is not None:
            payload["dense_vector"] = [float(v) for v in self.dense_vector]
        if self.sparse_vector is not None:
            payload["sparse_vector"] = {
                key: float(value) for key, value in self.sparse_vector.items()
            }
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ChunkRecord:
        payload = _require_mapping(data, "ChunkRecord")
        dense_vector = payload.get("dense_vector")
        sparse_vector = payload.get("sparse_vector")
        record = cls(
            id=_require_str_field(payload, "id", "ChunkRecord"),
            text=str(payload.get("text", "")),
            metadata=_require_mapping(payload.get("metadata", {}), "ChunkRecord.metadata"),
            dense_vector=list(dense_vector) if dense_vector is not None else None,
            sparse_vector=dict(sparse_vector) if sparse_vector is not None else None,
        )
        record.validate()
        return record


@dataclass(frozen=True)
class RetrievalResult:
    """检索链路统一结果载体，供 Dense/Sparse/Hybrid 与 MCP 返回组装复用。"""

    chunk_id: str
    score: float
    text: str
    metadata: dict[str, Any]

    def validate(self) -> None:
        if not isinstance(self.chunk_id, str) or not self.chunk_id.strip():
            raise TypesError("RetrievalResult.chunk_id 必须是非空字符串")
        if not isinstance(self.score, (int, float)):
            raise TypesError("RetrievalResult.score 必须是数值")
        if not isinstance(self.text, str):
            raise TypesError("RetrievalResult.text 必须是字符串")
        if not isinstance(self.metadata, dict):
            raise TypesError("RetrievalResult.metadata 必须是 dict")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "chunk_id": self.chunk_id,
            "score": float(self.score),
            "text": self.text,
            "metadata": dict(self.metadata),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RetrievalResult:
        payload = _require_mapping(data, "RetrievalResult")
        chunk_id = payload.get("chunk_id", payload.get("id"))
        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise TypesError("RetrievalResult 缺少合法 chunk_id/id")
        score = payload.get("score")
        if not isinstance(score, (int, float)):
            raise TypesError("RetrievalResult 缺少合法 score")
        text = payload.get("text")
        if not isinstance(text, str):
            raise TypesError("RetrievalResult 缺少合法 text")
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise TypesError("RetrievalResult.metadata 必须是 mapping")
        result = cls(
            chunk_id=chunk_id.strip(),
            score=float(score),
            text=text,
            metadata=dict(metadata),
        )
        result.validate()
        return result


def documents_to_json(documents: Sequence[Document]) -> str:
    """批量 Document 序列化为 JSON 数组字符串。"""
    return json.dumps([doc.to_dict() for doc in documents], ensure_ascii=False)
