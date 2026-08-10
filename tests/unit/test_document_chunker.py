"""DocumentChunker 单元测试：使用 FakeSplitter 隔离 libs.splitter。"""

from __future__ import annotations

from typing import Any

import pytest

from core.settings import IngestionSettings, Settings, load_settings
from core.types import Document, ImageMetadata, format_image_placeholder
from ingestion.chunking.document_chunker import DocumentChunker
from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.recursive_splitter import RecursiveSplitter


class FakeSplitter(BaseSplitter):
    """可控切分器：按固定片段列表返回，便于断言业务逻辑。"""

    def __init__(self, chunks: list[str]) -> None:
        self._chunks = chunks

    def split_text(self, text: str, trace: Any | None = None) -> list[str]:
        return list(self._chunks)


def _ingestion(**overrides: object) -> IngestionSettings:
    base = load_settings()
    data = {
        "chunk_size": base.ingestion.chunk_size,
        "chunk_overlap": base.ingestion.chunk_overlap,
        "splitter": base.ingestion.splitter,
        "batch_size": base.ingestion.batch_size,
    }
    data.update(overrides)
    return IngestionSettings(**data)


def _settings_with_ingestion(**overrides: object) -> Settings:
    base = load_settings()
    ingestion = _ingestion(**overrides)
    return Settings(
        llm=base.llm,
        embedding=base.embedding,
        vector_store=base.vector_store,
        retrieval=base.retrieval,
        rerank=base.rerank,
        evaluation=base.evaluation,
        observability=base.observability,
        ingestion=ingestion,
        vision_llm=base.vision_llm,
    )


def _sample_document(text: str, doc_id: str = "doc-hash-001") -> Document:
    return Document(
        id=doc_id,
        text=text,
        metadata={
            "source_path": "tests/fixtures/sample.pdf",
            "doc_type": "pdf",
            "title": "Sample",
        },
    )


@pytest.mark.unit
class TestDocumentChunkerWithFakeSplitter:
    """验证 Chunk ID、溯源、元数据继承与图片分发。"""

    def test_chunk_ids_are_unique_and_deterministic(self) -> None:
        """同一 Document 重复切分应产生相同 ID 序列，且 ID 在文档内唯一。"""
        doc = _sample_document("段落一\n\n段落二")
        splitter = FakeSplitter(["段落一", "段落二"])
        chunker = DocumentChunker(_settings_with_ingestion(), splitter=splitter)

        first = chunker.split_document(doc)
        second = chunker.split_document(doc)

        assert len(first) == 2
        assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
        assert len({chunk.id for chunk in first}) == 2
        assert all(chunk.id.startswith("doc-hash-001_") for chunk in first)

    def test_source_ref_and_metadata_inheritance(self) -> None:
        """Chunk 应指向父 Document，并继承文档级 metadata + chunk_index。"""
        doc = _sample_document("唯一段落")
        chunker = DocumentChunker(
            _settings_with_ingestion(),
            splitter=FakeSplitter(["唯一段落"]),
        )
        chunks = chunker.split_document(doc)

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.source_ref == doc.id
        assert chunk.metadata["source_path"] == "tests/fixtures/sample.pdf"
        assert chunk.metadata["doc_type"] == "pdf"
        assert chunk.metadata["title"] == "Sample"
        assert chunk.metadata["chunk_index"] == 0

    def test_image_refs_distributed_per_chunk(self) -> None:
        """含占位符的 chunk 仅携带自身引用的 images 子集。"""
        img_a = ImageMetadata(
            id="img_a",
            path="data/images/hash/img_a.png",
            text_offset=0,
            text_length=10,
        )
        img_b = ImageMetadata(
            id="img_b",
            path="data/images/hash/img_b.png",
            text_offset=20,
            text_length=10,
        )
        placeholder_a = format_image_placeholder("img_a")
        placeholder_b = format_image_placeholder("img_b")
        doc = Document(
            id="doc-images",
            text=f"块一 {placeholder_a}\n\n块二 {placeholder_b}",
            metadata={
                "source_path": "with_images.pdf",
                "doc_type": "pdf",
                "images": [img_a, img_b],
            },
        )
        chunker = DocumentChunker(
            _settings_with_ingestion(),
            splitter=FakeSplitter(
                [f"块一 {placeholder_a}", f"块二 {placeholder_b}"],
            ),
        )
        chunks = chunker.split_document(doc)

        assert chunks[0].metadata["image_refs"] == ["img_a"]
        assert len(chunks[0].metadata["images"]) == 1
        assert chunks[0].metadata["images"][0]["id"] == "img_a"

        assert chunks[1].metadata["image_refs"] == ["img_b"]
        assert len(chunks[1].metadata["images"]) == 1
        assert chunks[1].metadata["images"][0]["id"] == "img_b"

    def test_chunk_without_placeholder_has_no_images_field(self) -> None:
        """无占位符的 chunk 不应包含 images 字段。"""
        img = ImageMetadata(
            id="img_only",
            path="data/images/hash/img_only.png",
            text_offset=0,
            text_length=10,
        )
        placeholder = format_image_placeholder("img_only")
        doc = Document(
            id="doc-mixed",
            text=f"有图 {placeholder}\n\n纯文本块",
            metadata={
                "source_path": "mixed.pdf",
                "images": [img],
            },
        )
        chunker = DocumentChunker(
            _settings_with_ingestion(),
            splitter=FakeSplitter([f"有图 {placeholder}", "纯文本块"]),
        )
        chunks = chunker.split_document(doc)

        assert "images" in chunks[0].metadata
        assert "images" not in chunks[1].metadata
        assert "image_refs" not in chunks[1].metadata

    def test_offsets_cover_chunk_text_in_document(self) -> None:
        """start/end_offset 应能在原文中还原 chunk 文本。"""
        body = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        doc = _sample_document(body)
        chunker = DocumentChunker(
            _settings_with_ingestion(),
            splitter=FakeSplitter(["ABCDEF", "GHIJKLMNOPQRSTUVWXYZ"]),
        )
        chunks = chunker.split_document(doc)

        for chunk in chunks:
            slice_text = doc.text[chunk.start_offset:chunk.end_offset]
            assert slice_text == chunk.text


@pytest.mark.unit
class TestDocumentChunkerConfigDriven:
    """验证修改 chunk_size 会影响产出 chunk 数量。"""

    def test_smaller_chunk_size_produces_more_chunks(self) -> None:
        long_text = "\n\n".join(f"Section {index}: " + ("word " * 30) for index in range(6))
        doc = _sample_document(long_text)

        small_chunker = DocumentChunker(
            _settings_with_ingestion(chunk_size=80, chunk_overlap=0),
            splitter=RecursiveSplitter(_ingestion(chunk_size=80, chunk_overlap=0)),
        )
        large_chunker = DocumentChunker(
            _settings_with_ingestion(chunk_size=400, chunk_overlap=0),
            splitter=RecursiveSplitter(_ingestion(chunk_size=400, chunk_overlap=0)),
        )

        small_chunks = small_chunker.split_document(doc)
        large_chunks = large_chunker.split_document(doc)

        assert len(small_chunks) > len(large_chunks)
