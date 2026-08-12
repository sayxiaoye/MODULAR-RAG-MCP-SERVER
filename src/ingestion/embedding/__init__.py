"""摄取链路 Embedding 模块：稠密与稀疏向量编码。"""

from ingestion.embedding.batch_processor import BatchProcessor, BatchProcessorError, BatchEncodingResult
from ingestion.embedding.dense_encoder import DenseEncoder, DenseEncoderError
from ingestion.embedding.sparse_encoder import SparseChunkStats, SparseEncoder, SparseEncoderError

__all__ = [
    "DenseEncoder",
    "DenseEncoderError",
    "SparseEncoder",
    "SparseEncoderError",
    "SparseChunkStats",
    "BatchProcessor",
    "BatchProcessorError",
    "BatchEncodingResult",
]
