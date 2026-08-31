"""Chroma 向量库实现：本地持久化 upsert 与 Dense 检索。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import chromadb

from core.settings import VectorStoreSettings
from libs.vector_store.base_vector_store import BaseVectorStore, VectorStoreError


def _sanitize_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """将 metadata 转为 Chroma 支持的标量类型（str/int/float/bool）。"""
    sanitized: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[str(key)] = value
        else:
            sanitized[str(key)] = json.dumps(value, ensure_ascii=False)
    return sanitized


def _build_where_clause(filters: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """将 filters 转为 Chroma where 表达式（等值匹配）。"""
    if not filters:
        return None
    if len(filters) == 1:
        key, value = next(iter(filters.items()))
        return {str(key): value}
    return {
        "$and": [{str(key): value} for key, value in filters.items()],
    }


@dataclass(frozen=True)
class CollectionStats:
    """单个 Chroma 集合的资产统计，供 G1 总览页展示。"""

    collection: str
    chunk_count: int
    document_count: int


class ChromaStore(BaseVectorStore):
    """基于 ChromaDB 的 VectorStore 实现，支持本地目录持久化。"""

    def __init__(self, settings: VectorStoreSettings) -> None:
        self.settings = settings
        self._client = chromadb.PersistentClient(path=settings.persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(
        self,
        records: Sequence[Mapping[str, Any]],
        trace: Any | None = None,
    ) -> None:
        validated = self._validate_upsert_records(records)
        ids = [str(record["id"]) for record in validated]
        embeddings = [[float(v) for v in record["dense_vector"]] for record in validated]
        documents = [str(record["text"]) for record in validated]
        metadatas = [_sanitize_metadata(record["metadata"]) for record in validated]
        try:
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as exc:
            raise VectorStoreError(f"[chroma] upsert 失败: {exc}") from exc

    def query(
        self,
        vector: Sequence[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[dict[str, Any]]:
        query_vector = self._validate_query_vector(vector, top_k)
        where = _build_where_clause(filters)
        try:
            raw = self._collection.query(
                query_embeddings=[query_vector],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"[chroma] query 失败: {exc}") from exc

        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        results: list[dict[str, Any]] = []
        for index, record_id in enumerate(ids):
            distance = float(distances[index]) if distances else 0.0
            # cosine 空间下 distance 越小越相似，转为 score 便于与检索链路对齐
            score = 1.0 - distance
            doc_text = documents[index] if documents else ""
            metadata = metadatas[index] if metadatas else {}
            results.append(
                {
                    "id": str(record_id),
                    "score": score,
                    "text": str(doc_text or ""),
                    "metadata": dict(metadata or {}),
                }
            )
        return self._validate_query_results(results)

    def get_by_ids(
        self,
        ids: Sequence[str],
        trace: Any | None = None,
    ) -> list[dict[str, Any]]:
        """按 ID 批量读取 documents 与 metadatas，供稀疏检索回填正文。"""
        if not ids:
            return []
        id_list = [str(record_id) for record_id in ids]
        try:
            raw = self._collection.get(
                ids=id_list,
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            raise VectorStoreError(f"[chroma] get_by_ids 失败: {exc}") from exc

        raw_ids = raw.get("ids", [])
        documents = raw.get("documents", [])
        metadatas = raw.get("metadatas", [])

        results: list[dict[str, Any]] = []
        for index, record_id in enumerate(raw_ids):
            doc_text = documents[index] if documents else ""
            metadata = metadatas[index] if metadatas else {}
            results.append(
                {
                    "id": str(record_id),
                    "text": str(doc_text or ""),
                    "metadata": dict(metadata or {}),
                }
            )
        return self._validate_get_by_ids_results(results)

    def get_collection_stats(self, collection: str | None = None) -> CollectionStats:
        """
        汇总集合内 chunk 数与去重文档数，供 Dashboard 总览页展示。

        Args:
            collection: 集合名；默认使用当前 ``settings.collection_name``。
        """
        name = (collection or self.settings.collection_name).strip()
        if not name:
            raise VectorStoreError("collection 不能为空")
        try:
            target = (
                self._collection
                if name == self._collection.name
                else self._client.get_collection(name)
            )
        except Exception as exc:
            raise VectorStoreError(f"[chroma] 读取集合失败: {exc}") from exc

        try:
            chunk_count = int(target.count())
        except Exception as exc:
            raise VectorStoreError(f"[chroma] count 失败: {exc}") from exc

        if chunk_count <= 0:
            return CollectionStats(
                collection=name,
                chunk_count=0,
                document_count=0,
            )

        try:
            raw = target.get(include=["metadatas"])
        except Exception as exc:
            raise VectorStoreError(f"[chroma] 读取 metadata 失败: {exc}") from exc

        sources: set[str] = set()
        for metadata in raw.get("metadatas") or []:
            if not isinstance(metadata, Mapping):
                continue
            source = metadata.get("source_path")
            if isinstance(source, str) and source.strip():
                sources.add(source.strip())
        return CollectionStats(
            collection=name,
            chunk_count=chunk_count,
            document_count=len(sources),
        )
