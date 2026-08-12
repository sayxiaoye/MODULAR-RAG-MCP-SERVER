"""DenseEncoder 单元测试：验证向量数量与维度契约。"""

from __future__ import annotations

import pytest

from core.settings import EmbeddingSettings, Settings, load_settings
from core.types import Chunk
from ingestion.embedding.dense_encoder import DenseEncoder, DenseEncoderError
from libs.embedding.base_embedding import BaseEmbedding, EmbeddingError
from libs.embedding.embedding_factory import EmbeddingFactory


class FakeEmbedding(BaseEmbedding):
    """测试用 Embedding：生成确定性向量，不访问外部 API。"""

    def __init__(self, settings: EmbeddingSettings, dimensions: int | None = None) -> None:
        self.settings = settings
        self.dimensions = dimensions or settings.dimensions

    def embed(self, texts, trace=None) -> list[list[float]]:
        validated = self._validate_texts(texts)
        vectors: list[list[float]] = []
        for text in validated:
            seed = sum(ord(char) for char in text)
            vectors.append([(seed + index) % 97 / 97.0 for index in range(self.dimensions)])
        return vectors


class BrokenCountEmbedding(BaseEmbedding):
    """故意返回错误数量的向量，用于契约测试。"""

    def __init__(self, settings: EmbeddingSettings) -> None:
        self.settings = settings

    def embed(self, texts, trace=None) -> list[list[float]]:
        self._validate_texts(texts)
        return [[0.1, 0.2]]  # 仅 1 条，与输入数量不一致


class BrokenDimEmbedding(BaseEmbedding):
    """故意返回错误维度，用于契约测试。"""

    def __init__(self, settings: EmbeddingSettings) -> None:
        self.settings = settings

    def embed(self, texts, trace=None) -> list[list[float]]:
        validated = self._validate_texts(texts)
        return [[0.1, 0.2] for _ in validated]  # 维度 2，与配置 768 不一致


def _chunk(text: str, index: int) -> Chunk:
    return Chunk(
        id=f"chunk_{index:04d}",
        text=text,
        metadata={"source_path": "sample.pdf", "chunk_index": index},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-001",
    )


@pytest.fixture(autouse=True)
def _reset_embedding_factory() -> None:
    EmbeddingFactory.reset_constructor()
    yield
    EmbeddingFactory.reset_constructor()


@pytest.mark.unit
class TestDenseEncoder:
    """验证 DenseEncoder 输出与 Chunk 列表对齐。"""

    def test_encode_returns_vectors_matching_chunk_count(self) -> None:
        settings = load_settings()
        encoder = DenseEncoder(settings, embedding=FakeEmbedding(settings.embedding))
        chunks = [_chunk("第一段", 0), _chunk("第二段", 1)]

        vectors = encoder.encode(chunks)

        assert len(vectors) == len(chunks)
        assert all(len(vector) == settings.embedding.dimensions for vector in vectors)

    def test_encode_empty_chunks_returns_empty_list(self) -> None:
        encoder = DenseEncoder(load_settings(), embedding=FakeEmbedding(load_settings().embedding))
        assert encoder.encode([]) == []

    def test_encode_vectors_have_consistent_dimensions(self) -> None:
        settings = load_settings()
        encoder = DenseEncoder(settings, embedding=FakeEmbedding(settings.embedding))
        chunks = [_chunk("A", 0), _chunk("BBBB", 1), _chunk("CCCCCC", 2)]
        vectors = encoder.encode(chunks)
        dims = {len(vector) for vector in vectors}
        assert len(dims) == 1
        assert dims.pop() == settings.embedding.dimensions

    def test_mismatched_vector_count_raises(self) -> None:
        settings = load_settings()
        encoder = DenseEncoder(settings, embedding=BrokenCountEmbedding(settings.embedding))
        with pytest.raises(DenseEncoderError, match="不一致"):
            encoder.encode([_chunk("only one", 0), _chunk("two", 1)])

    def test_mismatched_vector_dimension_raises(self) -> None:
        settings = load_settings()
        encoder = DenseEncoder(settings, embedding=BrokenDimEmbedding(settings.embedding))
        with pytest.raises(DenseEncoderError, match="维度"):
            encoder.encode([_chunk("text", 0)])

    def test_embedding_failure_wraps_as_dense_encoder_error(self) -> None:
        class FailingEmbedding(BaseEmbedding):
            def embed(self, texts, trace=None):
                raise EmbeddingError("mock embed failure")

        settings = load_settings()
        encoder = DenseEncoder(settings, embedding=FailingEmbedding())
        with pytest.raises(DenseEncoderError, match="Dense 编码失败"):
            encoder.encode([_chunk("x", 0)])

    def test_factory_injected_embedding_via_set_constructor(self) -> None:
        """未显式注入时，应能通过工厂构造 Embedding。"""
        settings = load_settings()
        EmbeddingFactory.set_constructor(lambda cfg: FakeEmbedding(cfg))
        encoder = DenseEncoder(settings)
        vectors = encoder.encode([_chunk("factory path", 0)])
        assert len(vectors) == 1
        assert len(vectors[0]) == settings.embedding.dimensions
