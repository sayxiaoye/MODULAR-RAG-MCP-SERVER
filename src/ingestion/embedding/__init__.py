"""摄取链路 Embedding 模块：稠密与稀疏向量编码。"""

from ingestion.embedding.dense_encoder import DenseEncoder, DenseEncoderError

__all__ = ["DenseEncoder", "DenseEncoderError"]
