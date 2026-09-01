"""BM25Indexer 往返测试：构建、持久化、加载与查询稳定性。"""

from __future__ import annotations

import math

import pytest

from core.types import Chunk
from ingestion.embedding.sparse_encoder import SparseEncoder
from ingestion.storage.bm25_indexer import BM25Indexer, _compute_idf


def _stats(chunk_id: str, text: str):
    encoder = SparseEncoder()
    return encoder.encode(
        [
            Chunk(
                id=chunk_id,
                text=text,
                metadata={"source_path": "test.md"},
                start_offset=0,
                end_offset=len(text),
            )
        ]
    )[0]


@pytest.mark.unit
class TestBM25IndexerRoundtrip:
    """验证索引 build/load/query 与 IDF 计算。"""

    def test_build_save_load_query_returns_stable_top_ids(self, tmp_path) -> None:
        stats = [
            _stats("chunk_a", "Azure OpenAI 配置指南"),
            _stats("chunk_b", "BM25 检索与 RRF 融合"),
            _stats("chunk_c", "Azure 向量数据库部署"),
        ]
        indexer = BM25Indexer(collection="test", index_root=tmp_path / "bm25")
        indexer.build(stats, rebuild=True)
        indexer.save()

        reloaded = BM25Indexer(collection="test", index_root=tmp_path / "bm25")
        reloaded.load()
        first = reloaded.query("Azure 配置", top_k=2)
        second = reloaded.query("Azure 配置", top_k=2)

        assert first == second
        assert first[0][0] in {"chunk_a", "chunk_c"}
        assert len(first) == 2

    def test_idf_formula_matches_spec(self) -> None:
        document_count = 10
        document_frequency = 3
        expected = math.log((document_count - document_frequency + 0.5) / (document_frequency + 0.5))
        assert _compute_idf(document_count, document_frequency) == expected

    def test_indexer_idf_after_build(self, tmp_path) -> None:
        stats = [
            _stats("c1", "azure azure cloud"),
            _stats("c2", "bm25 sparse retrieval"),
        ]
        indexer = BM25Indexer(collection="idf", index_root=tmp_path)
        indexer.build(stats, rebuild=True)

        azure_idf = indexer.get_idf("azure")
        bm25_idf = indexer.get_idf("bm25")
        assert azure_idf is not None
        assert bm25_idf is not None
        # azure 出现在 1/2 文档
        assert azure_idf == _compute_idf(2, 1)
        assert bm25_idf == _compute_idf(2, 1)

    def test_incremental_add_updates_index(self, tmp_path) -> None:
        indexer = BM25Indexer(collection="incr", index_root=tmp_path)
        indexer.build([_stats("c1", "first doc")], rebuild=True)
        indexer.add([_stats("c2", "second doc bm25")])
        results = indexer.query("bm25", top_k=3)
        assert results[0][0] == "c2"

    def test_rebuild_replaces_old_documents(self, tmp_path) -> None:
        indexer = BM25Indexer(collection="rebuild", index_root=tmp_path)
        indexer.build([_stats("old", "legacy content")], rebuild=True)
        indexer.build([_stats("new", "fresh bm25 content")], rebuild=True)
        results = indexer.query("legacy", top_k=5)
        assert results == []

    def test_query_empty_index_returns_empty(self, tmp_path) -> None:
        indexer = BM25Indexer(collection="empty", index_root=tmp_path)
        indexer.build([], rebuild=True)
        assert indexer.query("anything") == []

    def test_new_indexer_load_then_add_preserves_previous_chunks(self, tmp_path) -> None:
        """第二次打开同一 json 应先 load 再 add，不能只留下最新文档。"""
        root = tmp_path / "bm25"
        first = BM25Indexer(collection="col_a", index_root=root)
        first.add([_stats("doc1_0000", "first document azure")])
        first.save()

        second = BM25Indexer(collection="col_a", index_root=root)
        second.load()
        second.add([_stats("doc2_0000", "second document bm25")])
        second.save()

        reloaded = BM25Indexer(collection="col_a", index_root=root)
        reloaded.load()
        assert "doc1_0000" in reloaded._doc_lengths
        assert "doc2_0000" in reloaded._doc_lengths
        assert reloaded._document_count == 2
