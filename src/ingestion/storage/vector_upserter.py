"""VectorUpserter：生成稳定 chunk_id 并幂等写入向量库。"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from core.trace.trace_context import TraceContext
from core.types import Chunk
from libs.vector_store.base_vector_store import BaseVectorStore, VectorStoreError


class VectorUpserterError(Exception):
    """向量写入编排失败或输入契约不满足时抛出。"""


def compute_content_hash_prefix(text: str, length: int = 8) -> str:
    """计算文本 SHA256 哈希前缀，供稳定 ID 组合使用。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def generate_stable_chunk_id(source_path: str, chunk_index: int, text: str) -> str:
    """
    生成确定性 chunk_id：hash(source_path + chunk_index + content_hash[:8])。

    与 C12 规范一致：内容变更会导致 content_hash 变化，从而得到新 ID。
    """
    if not source_path or not source_path.strip():
        raise VectorUpserterError("source_path 不能为空")
    content_hash = compute_content_hash_prefix(text)
    raw = f"{source_path}{chunk_index}{content_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class VectorUpserter:
    """将 DenseEncoder 向量与 Chunk 元数据组装为 ChunkRecord 并 upsert 到 VectorStore。"""

    def __init__(self, vector_store: BaseVectorStore) -> None:
        self._vector_store = vector_store

    def upsert(
        self,
        chunks: Sequence[Chunk],
        dense_vectors: Sequence[Sequence[float]],
        trace: Any | None = None,
    ) -> list[str]:
        """
        批量幂等写入向量记录，返回与输入等长的稳定 chunk_id 列表。

        Args:
            chunks: 待写入 Chunk，metadata 需含 source_path 与 chunk_index。
            dense_vectors: DenseEncoder 产出的向量，顺序与 chunks 对齐。
            trace: 可选 TraceContext，记录写入耗时。

        Returns:
            每条记录对应的稳定 chunk_id（顺序与 chunks 一致）。

        Raises:
            VectorUpserterError: 输入数量不一致或缺少必要 metadata。
        """
        if len(chunks) != len(dense_vectors):
            raise VectorUpserterError(
                f"chunks 数量({len(chunks)})与 dense_vectors 数量({len(dense_vectors)})不一致"
            )
        if not chunks:
            return []

        records: list[dict[str, Any]] = []
        chunk_ids: list[str] = []

        for chunk, vector in zip(chunks, dense_vectors, strict=True):
            source_path, chunk_index = self._extract_identity_fields(chunk)
            stable_id = generate_stable_chunk_id(source_path, chunk_index, chunk.text)
            chunk_ids.append(stable_id)
            records.append(
                {
                    "id": stable_id,
                    "text": chunk.text,
                    "metadata": dict(chunk.metadata),
                    "dense_vector": [float(v) for v in vector],
                }
            )

        try:
            self._vector_store.upsert(records, trace=trace)
        except VectorStoreError as exc:
            raise VectorUpserterError(f"向量库 upsert 失败: {exc}") from exc

        if isinstance(trace, TraceContext):
            trace.record_stage(
                "vector_upsert",
                elapsed_ms=0.0,
                record_count=len(records),
            )

        return chunk_ids

    @staticmethod
    def _extract_identity_fields(chunk: Chunk) -> tuple[str, int]:
        """从 Chunk metadata 提取 source_path 与 chunk_index。"""
        metadata = chunk.metadata
        if not isinstance(metadata, Mapping):
            raise VectorUpserterError("Chunk.metadata 必须是 mapping")

        source_path = str(metadata.get("source_path", "")).strip()
        if not source_path:
            raise VectorUpserterError(f"Chunk {chunk.id!r} 缺少 metadata.source_path")

        chunk_index_raw = metadata.get("chunk_index")
        if isinstance(chunk_index_raw, bool) or not isinstance(chunk_index_raw, int):
            raise VectorUpserterError(f"Chunk {chunk.id!r} 缺少合法 metadata.chunk_index")

        return source_path, chunk_index_raw
