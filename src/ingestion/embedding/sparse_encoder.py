"""SparseEncoder：为 Chunk 提取 BM25 所需的词项统计（TF 与文档长度）。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from core.types import Chunk

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9\u4e00-\u9fff]+")


@dataclass(frozen=True)
class SparseChunkStats:
    """
    单 Chunk 的稀疏统计结果，供 C11 BM25Indexer 构建倒排索引。

    Attributes:
        chunk_id: 与 Chunk.id 对齐。
        term_frequencies: 词项 -> 词频 TF。
        doc_length: 文档内 token 总数（BM25 长度归一化用）。
    """

    chunk_id: str
    term_frequencies: dict[str, int]
    doc_length: int

    def to_sparse_vector(self) -> dict[str, float]:
        """转换为 ChunkRecord.sparse_vector 兼容的 float 权重字典。"""
        return {term: float(count) for term, count in self.term_frequencies.items()}


class SparseEncoderError(Exception):
    """稀疏编码失败或输入非法时抛出。"""


class SparseEncoder:
    """摄取链路 Sparse 编码器：输出 BM25 索引构建所需的 per-chunk 词频统计。"""

    def encode(self, chunks: Sequence[Chunk]) -> list[SparseChunkStats]:
        """
        对 Chunk 文本做分词并统计 TF。

        Args:
            chunks: 待编码 Chunk 列表。

        Returns:
            与 chunks 等长的 SparseChunkStats 列表。
        """
        if not chunks:
            return []

        results: list[SparseChunkStats] = []
        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            term_frequencies = self._count_term_frequencies(tokens)
            doc_length = len(tokens)
            results.append(
                SparseChunkStats(
                    chunk_id=chunk.id,
                    term_frequencies=term_frequencies,
                    doc_length=doc_length,
                )
            )
        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """将文本切分为 BM25 用词项；空文本返回空列表。"""
        if not isinstance(text, str) or not text.strip():
            return []
        return _TOKEN_PATTERN.findall(text.lower())

    @staticmethod
    def _count_term_frequencies(tokens: Sequence[str]) -> dict[str, int]:
        """统计词频 TF。"""
        frequencies: dict[str, int] = {}
        for token in tokens:
            frequencies[token] = frequencies.get(token, 0) + 1
        return frequencies
