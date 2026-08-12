"""core.types 数据契约单元测试。"""

from __future__ import annotations

import json

import pytest

from core.types import (
    Chunk,
    ChunkRecord,
    Document,
    ImageMetadata,
    TypesError,
    format_image_placeholder,
)


@pytest.mark.unit
class TestImagePlaceholder:
    """验证图片占位符格式。"""

    def test_format_image_placeholder(self) -> None:
        assert format_image_placeholder("doc_1_0") == "[IMAGE: doc_1_0]"


@pytest.mark.unit
class TestDocumentSerialization:
    """验证 Document 序列化与 metadata 契约。"""

    def test_document_roundtrip_json(self) -> None:
        image = ImageMetadata(
            id="hash_1_0",
            path="data/images/default/hash_1_0.png",
            page=1,
            text_offset=10,
            text_length=len("[IMAGE: hash_1_0]"),
            position={"x": 0, "y": 0},
        )
        placeholder = format_image_placeholder("hash_1_0")
        doc = Document(
            id="doc-001",
            text=f"前言{placeholder}结束",
            metadata={
                "source_path": "tests/fixtures/sample.pdf",
                "doc_type": "pdf",
                "images": [image],
            },
        )
        restored = Document.from_dict(json.loads(doc.to_json()))
        assert restored.id == doc.id
        assert restored.text == doc.text
        assert restored.metadata["source_path"] == "tests/fixtures/sample.pdf"
        assert restored.metadata["images"][0]["id"] == "hash_1_0"

    def test_document_missing_source_path_raises(self) -> None:
        doc = Document(id="d1", text="hello", metadata={})
        with pytest.raises(TypesError, match="source_path"):
            doc.validate()


@pytest.mark.unit
class TestChunkSerialization:
    """验证 Chunk 定位字段与序列化稳定性。"""

    def test_chunk_roundtrip(self) -> None:
        chunk = Chunk(
            id="chunk-001",
            text="段落内容",
            metadata={"source_path": "guide.pdf", "chunk_index": 0},
            start_offset=0,
            end_offset=4,
            source_ref="doc-001",
        )
        payload = json.loads(chunk.to_json())
        restored = Chunk.from_dict(payload)
        assert restored.id == chunk.id
        assert restored.start_offset == 0
        assert restored.end_offset == 4
        assert restored.source_ref == "doc-001"


@pytest.mark.unit
class TestChunkRecordSerialization:
    """验证 ChunkRecord 向量字段可选与 JSON 稳定。"""

    def test_chunk_record_with_vectors(self) -> None:
        record = ChunkRecord(
            id="rec-1",
            text="Azure 配置",
            metadata={"source_path": "guide.pdf", "collection": "docs"},
            dense_vector=[0.1, 0.2, 0.3],
            sparse_vector={"azure": 1.5, "配置": 0.8},
        )
        restored = ChunkRecord.from_dict(json.loads(record.to_json()))
        assert restored.dense_vector == [0.1, 0.2, 0.3]
        assert restored.sparse_vector == {"azure": 1.5, "配置": 0.8}

    def test_chunk_record_without_vectors(self) -> None:
        record = ChunkRecord(
            id="rec-2",
            text="纯文本",
            metadata={"source_path": "a.pdf"},
        )
        payload = record.to_dict()
        assert "dense_vector" not in payload
        assert "sparse_vector" not in payload
