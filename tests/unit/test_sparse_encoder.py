"""SparseEncoder 单元测试：验证 BM25 词频统计契约。"""

from __future__ import annotations

import pytest

from core.types import Chunk
from ingestion.embedding.sparse_encoder import SparseChunkStats, SparseEncoder


def _chunk(text: str, chunk_id: str = "chunk_sparse_001") -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={"source_path": "sample.pdf", "chunk_index": 0},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-001",
    )


@pytest.mark.unit
class TestSparseEncoder:
    """验证 SparseEncoder 输出可用于 BM25Indexer。"""

    def test_encode_returns_stats_for_each_chunk(self) -> None:
        encoder = SparseEncoder()
        chunks = [_chunk("Azure 配置指南", "c0"), _chunk("BM25 检索原理", "c1")]
        stats = encoder.encode(chunks)

        assert len(stats) == 2
        assert stats[0].chunk_id == "c0"
        assert stats[1].chunk_id == "c1"

    def test_term_frequencies_and_doc_length(self) -> None:
        encoder = SparseEncoder()
        stats = encoder.encode([_chunk("RAG RAG pipeline")])[0]

        assert stats.term_frequencies["rag"] == 2
        assert stats.term_frequencies["pipeline"] == 1
        assert stats.doc_length == 3

    def test_empty_text_has_explicit_empty_stats(self) -> None:
        encoder = SparseEncoder()
        stats = encoder.encode([_chunk("")])[0]

        assert stats.term_frequencies == {}
        assert stats.doc_length == 0

    def test_whitespace_only_text_is_empty(self) -> None:
        stats = SparseEncoder().encode([_chunk("   \n\t  ")])[0]
        assert stats.term_frequencies == {}
        assert stats.doc_length == 0

    def test_to_sparse_vector_matches_term_frequencies(self) -> None:
        stats = SparseChunkStats(
            chunk_id="c",
            term_frequencies={"azure": 2, "配置": 1},
            doc_length=3,
        )
        vector = stats.to_sparse_vector()
        assert vector == {"azure": 2.0, "配置": 1.0}

    def test_chinese_and_english_mixed_tokenization(self) -> None:
        stats = SparseEncoder().encode([_chunk("混合 Mixed 检索 search")])[0]
        assert "混合" in stats.term_frequencies
        assert "mixed" in stats.term_frequencies
        assert stats.doc_length == 4

    def test_encode_empty_chunk_list(self) -> None:
        assert SparseEncoder().encode([]) == []
