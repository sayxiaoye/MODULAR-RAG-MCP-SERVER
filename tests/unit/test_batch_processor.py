"""BatchProcessor 单元测试：验证分批逻辑与顺序稳定性。"""

from __future__ import annotations

import pytest

from core.settings import IngestionSettings, Settings, load_settings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.embedding.batch_processor import BatchProcessor, BatchProcessorError
from ingestion.embedding.dense_encoder import DenseEncoder
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory


class CountingEmbedding(BaseEmbedding):
    """记录 embed 调用次数与批次大小的 Fake Embedding。"""

    def __init__(self, settings) -> None:
        self.settings = settings
        self.call_sizes: list[int] = []

    def embed(self, texts, trace=None) -> list[list[float]]:
        self.call_sizes.append(len(texts))
        dim = self.settings.dimensions
        return [[0.1] * dim for _ in texts]


def _settings(batch_size: int) -> Settings:
    base = load_settings()
    ingestion = IngestionSettings(
        chunk_size=base.ingestion.chunk_size,
        chunk_overlap=base.ingestion.chunk_overlap,
        splitter=base.ingestion.splitter,
        batch_size=batch_size,
        chunk_refiner=base.ingestion.chunk_refiner,
        metadata_enricher=base.ingestion.metadata_enricher,
    )
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


def _chunk(index: int) -> Chunk:
    text = f"chunk text {index}"
    return Chunk(
        id=f"chunk_{index:04d}",
        text=text,
        metadata={"source_path": "sample.pdf", "chunk_index": index},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-batch",
    )


@pytest.fixture(autouse=True)
def _reset_embedding_factory() -> None:
    EmbeddingFactory.reset_constructor()
    yield
    EmbeddingFactory.reset_constructor()


@pytest.mark.unit
class TestBatchProcessor:
    """验证 batch_size=2 时 5 chunks 分成 3 批且顺序稳定。"""

    def test_batch_sizes_for_five_chunks(self) -> None:
        processor = BatchProcessor(_settings(batch_size=2))
        assert processor.batch_sizes_for_count(5) == [2, 2, 1]

    def test_process_runs_three_batches_for_five_chunks(self) -> None:
        settings = _settings(batch_size=2)
        counting = CountingEmbedding(settings.embedding)
        dense = DenseEncoder(settings, embedding=counting)
        processor = BatchProcessor(settings, dense_encoder=dense, batch_size=2)
        chunks = [_chunk(i) for i in range(5)]

        result = processor.process(chunks)

        assert result.batch_count == 3
        assert counting.call_sizes == [2, 2, 1]
        assert len(result.dense_vectors) == 5
        assert len(result.sparse_stats) == 5

    def test_output_order_matches_input_chunks(self) -> None:
        settings = _settings(batch_size=2)
        counting = CountingEmbedding(settings.embedding)
        dense = DenseEncoder(settings, embedding=counting)
        processor = BatchProcessor(settings, dense_encoder=dense, batch_size=2)
        chunks = [_chunk(i) for i in range(5)]
        result = processor.process(chunks)

        assert [stat.chunk_id for stat in result.sparse_stats] == [chunk.id for chunk in chunks]

    def test_empty_chunks_returns_empty_result(self) -> None:
        processor = BatchProcessor(_settings(batch_size=2))
        result = processor.process([])
        assert result.dense_vectors == []
        assert result.sparse_stats == []
        assert result.batch_count == 0

    def test_trace_records_batch_stages(self) -> None:
        settings = _settings(batch_size=2)
        trace = TraceContext(trace_type="ingestion")
        counting = CountingEmbedding(settings.embedding)
        dense = DenseEncoder(settings, embedding=counting)
        processor = BatchProcessor(settings, dense_encoder=dense, batch_size=2)
        processor.process([_chunk(i) for i in range(3)], trace=trace)
        summary = trace.finish()
        batch_stages = [s for s in summary["stages"] if s["name"] == "embedding_batch"]
        assert len(batch_stages) == 2

    def test_invalid_batch_size_raises(self) -> None:
        settings = _settings(batch_size=2)
        with pytest.raises(BatchProcessorError, match="batch_size"):
            BatchProcessor(settings, batch_size=0)
