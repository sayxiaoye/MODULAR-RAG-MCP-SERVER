"""Pipeline on_progress 单元测试：验证 F5 进度回调阶段名与 current/total 契约。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import pytest

from core.settings import load_settings
from core.types import Chunk, Document
from ingestion.embedding.batch_processor import BatchEncodingResult
from ingestion.pipeline import IngestionPipeline
from ingestion.transform.base_transform import BaseTransform


class _FakeIntegrity:
    """可控完整性检查：默认继续摄取，可切换为跳过。"""

    def __init__(self, *, skip: bool = False) -> None:
        self._skip = skip

    def compute_sha256(self, path: str) -> str:
        return "fake-hash"

    def should_skip(self, file_hash: str) -> bool:
        return self._skip

    def mark_success(self, file_hash: str, source_path: str, chunk_count: int = 0) -> None:
        return None


class _FakeLoader:
    """返回固定 Document，避免真实 PDF 解析。"""

    def load(self, path: str) -> Document:
        return Document(
            id="doc-progress",
            text="进度回调测试文本",
            metadata={"source_path": path, "doc_type": "pdf", "doc_hash": "fake-hash"},
        )


class _FakeChunker:
    """产出单个 Chunk，隔离 split 之后的编排。"""

    def split_document(self, document: Document) -> list[Chunk]:
        text = document.text
        return [
            Chunk(
                id="chunk-1",
                text=text,
                metadata={"source_path": document.metadata["source_path"], "chunk_index": 0},
                start_offset=0,
                end_offset=len(text),
            )
        ]


class _IdentityTransform(BaseTransform):
    """透传 Transform，仅用于触发 transform 进度回调。"""

    def transform(self, chunks: Sequence[Chunk], trace: Any | None = None) -> list[Chunk]:
        return list(chunks)


class _FakeBatchProcessor:
    """返回与输入等长的假向量，不访问外部 Embedding。"""

    def process(self, chunks: Sequence[Chunk], trace: Any | None = None) -> BatchEncodingResult:
        return BatchEncodingResult(
            dense_vectors=[[0.1, 0.2] for _ in chunks],
            sparse_stats=[],
            batch_count=1,
        )


class _FakeUpserter:
    def upsert(
        self,
        chunks: Sequence[Chunk],
        dense_vectors: Sequence[Any],
        trace: Any | None = None,
    ) -> list[str]:
        return [chunk.id for chunk in chunks]


class _FakeBM25:
    def add(self, stats: Sequence[Any]) -> None:
        return None

    def save(self) -> None:
        return None


def _build_pipeline(*, skip: bool = False) -> IngestionPipeline:
    """组装全 Fake 依赖的 Pipeline，避免 Chroma / Embedding 副作用。"""
    return IngestionPipeline(
        settings=load_settings(),
        integrity_checker=_FakeIntegrity(skip=skip),
        loader=_FakeLoader(),
        chunker=_FakeChunker(),
        transforms=[_IdentityTransform()],
        batch_processor=_FakeBatchProcessor(),
        vector_upserter=_FakeUpserter(),
        bm25_indexer=_FakeBM25(),
    )


@pytest.mark.unit
class TestPipelineProgress:
    """验证 on_progress 在各阶段被调用，且 current/total 合法。"""

    def test_on_progress_invoked_for_canonical_stages(self, tmp_path: Path) -> None:
        """完整摄取应按顺序回调 integrity/load/split/transform/embed/upsert。"""
        events: list[tuple[str, int, int]] = []
        source = tmp_path / "progress.pdf"
        source.write_bytes(b"%PDF-1.4 progress")

        result = _build_pipeline().run(
            str(source),
            collection="progress",
            force=True,
            on_progress=lambda stage, current, total: events.append((stage, current, total)),
        )

        assert result.skipped is False
        ordered = []
        for stage, current, total in events:
            assert total >= 1
            assert 1 <= current <= total
            if stage not in ordered:
                ordered.append(stage)

        assert ordered == [
            "integrity",
            "load",
            "split",
            "transform",
            "embed",
            "upsert",
        ]

    def test_on_progress_none_does_not_affect_run(self, tmp_path: Path) -> None:
        """on_progress 为 None 时应正常完成摄取。"""
        source = tmp_path / "progress.pdf"
        source.write_bytes(b"%PDF-1.4 progress")

        result = _build_pipeline().run(
            str(source),
            collection="progress",
            force=True,
            on_progress=None,
        )

        assert result.skipped is False
        assert result.chunk_count == 1

    def test_skipped_run_only_reports_integrity(self, tmp_path: Path) -> None:
        """integrity 跳过时只应收到 integrity 进度，不应进入后续阶段。"""
        events: list[tuple[str, int, int]] = []
        source = tmp_path / "progress.pdf"
        source.write_bytes(b"%PDF-1.4 progress")

        result = _build_pipeline(skip=True).run(
            str(source),
            collection="progress",
            force=False,
            on_progress=lambda stage, current, total: events.append((stage, current, total)),
        )

        assert result.skipped is True
        assert events == [("integrity", 1, 1)]
