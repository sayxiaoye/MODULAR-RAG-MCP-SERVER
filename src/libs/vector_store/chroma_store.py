"""Chroma 向量库实现：本地持久化 upsert 与 Dense 检索。"""

from __future__ import annotations

import gc
import json
import logging
import re
import shutil
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import chromadb

from core.settings import VectorStoreSettings, resolve_path
from libs.vector_store.base_vector_store import BaseVectorStore, VectorStoreError

_HNSW_METADATA = {"hnsw:space": "cosine"}
_UUID_DIR = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
logger = logging.getLogger(__name__)


def _rmtree_with_retry(path: Path, attempts: int = 6) -> None:
    """删除目录；Windows 上 HNSW 文件可能短暂占用，做有限次重试。"""
    last_error: OSError | None = None
    for attempt in range(attempts):
        gc.collect()
        try:
            shutil.rmtree(path)
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.05 * (attempt + 1))
    if last_error is not None:
        raise last_error


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
    """
    基于 ChromaDB 的 VectorStore。

    PersistentClient 在构造时创建并长期复用；具体 collection 按方法参数解析，
    对应「一个逻辑集合 = 一个 Chroma collection」。
    """

    def __init__(self, settings: VectorStoreSettings) -> None:
        self.settings = settings
        self._client = chromadb.PersistentClient(path=settings.persist_directory)
        # 上次进程删集合后可能留下锁着的 UUID 目录，启动时再扫一遍
        self.cleanup_orphan_segment_dirs()

    def _collection_name(self, collection: str | None) -> str:
        name = (collection or self.settings.collection_name or "").strip()
        if not name:
            raise VectorStoreError("collection 不能为空")
        return name

    def _get_collection(self, collection: str | None, *, create: bool):
        """按逻辑名解析 Chroma collection；读路径不存在则返回 None。"""
        name = self._collection_name(collection)
        try:
            if create:
                return self._client.get_or_create_collection(
                    name=name,
                    metadata=_HNSW_METADATA,
                )
            return self._client.get_collection(name)
        except Exception as exc:
            if create:
                raise VectorStoreError(f"[chroma] 打开集合失败: {exc}") from exc
            return None

    def list_collection_names(self) -> list[str]:
        """列出持久化目录中已存在的 Chroma 集合名。"""
        try:
            return sorted(item.name for item in self._client.list_collections())
        except Exception as exc:
            raise VectorStoreError(f"[chroma] 列举集合失败: {exc}") from exc

    def delete_collection(self, collection: str, trace: Any | None = None) -> None:
        """删除 Chroma collection，并清掉磁盘上不再被引用的 UUID 索引目录。"""
        name = self._collection_name(collection)
        if self._get_collection(name, create=False) is not None:
            try:
                self._client.delete_collection(name)
            except Exception as exc:
                raise VectorStoreError(f"[chroma] 删除集合失败: {exc}") from exc
        self.cleanup_orphan_segment_dirs()

    def cleanup_orphan_segment_dirs(self) -> list[str]:
        """删除 persist 目录中 sqlite 已不再引用的 UUID 段目录。"""
        root = self._persist_root()
        if not root.is_dir():
            return []
        live = self._live_segment_ids()
        if live is None:
            return []
        removed: list[str] = []
        for child in root.iterdir():
            if not child.is_dir() or not _UUID_DIR.match(child.name):
                continue
            if child.name in live:
                continue
            try:
                _rmtree_with_retry(child)
                removed.append(child.name)
            except OSError:
                logger.warning("无法删除 Chroma 孤儿目录: %s", child, exc_info=True)
        return removed

    def _persist_root(self) -> Path:
        raw = Path(self.settings.persist_directory)
        if not raw.is_absolute():
            return resolve_path(raw)
        return raw

    def _live_segment_ids(self) -> set[str] | None:
        """读取仍被 chroma.sqlite3 引用的 segment id；读失败时返回 None，避免误删。"""
        db_path = self._persist_root() / "chroma.sqlite3"
        if not db_path.is_file():
            return set()
        try:
            conn = sqlite3.connect(str(db_path), timeout=5.0)
            try:
                rows = conn.execute("SELECT id FROM segments").fetchall()
            finally:
                conn.close()
        except sqlite3.Error:
            logger.warning("读取 chroma.sqlite3 segments 失败，跳过孤儿目录清理", exc_info=True)
            return None
        return {str(row[0]) for row in rows}

    def upsert(
        self,
        records: Sequence[Mapping[str, Any]],
        trace: Any | None = None,
        *,
        collection: str | None = None,
    ) -> None:
        validated = self._validate_upsert_records(records)
        target = self._get_collection(collection, create=True)
        ids = [str(record["id"]) for record in validated]
        embeddings = [[float(v) for v in record["dense_vector"]] for record in validated]
        documents = [str(record["text"]) for record in validated]
        metadatas = [_sanitize_metadata(record["metadata"]) for record in validated]
        try:
            target.upsert(
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
        *,
        collection: str | None = None,
    ) -> list[dict[str, Any]]:
        query_vector = self._validate_query_vector(vector, top_k)
        target = self._get_collection(collection, create=False)
        if target is None:
            return []
        where = _build_where_clause(filters)
        try:
            raw = target.query(
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
        *,
        collection: str | None = None,
    ) -> list[dict[str, Any]]:
        """按 ID 批量读取 documents 与 metadatas，供稀疏检索回填正文。"""
        if not ids:
            return []
        target = self._get_collection(collection, create=False)
        if target is None:
            return []
        id_list = [str(record_id) for record_id in ids]
        try:
            raw = target.get(
                ids=id_list,
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            raise VectorStoreError(f"[chroma] get_by_ids 失败: {exc}") from exc
        return self._records_from_get(raw)

    def get_collection_stats(self, collection: str | None = None) -> CollectionStats:
        """
        汇总集合内 chunk 数与去重文档数，供 Dashboard 总览页展示。

        Args:
            collection: 集合名；默认使用 ``settings.collection_name``。
        """
        name = self._collection_name(collection)
        target = self._get_collection(name, create=False)
        if target is None:
            return CollectionStats(collection=name, chunk_count=0, document_count=0)

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

    def get_by_metadata(
        self,
        filters: Mapping[str, Any] | None = None,
        trace: Any | None = None,
        *,
        collection: str | None = None,
    ) -> list[dict[str, Any]]:
        """按 metadata 等值条件读取 documents 与 metadatas。"""
        target = self._get_collection(collection, create=False)
        if target is None:
            return []
        where = _build_where_clause(filters)
        try:
            raw = target.get(
                where=where,
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            raise VectorStoreError(f"[chroma] get_by_metadata 失败: {exc}") from exc
        return self._records_from_get(raw)

    def delete_by_metadata(
        self,
        filters: Mapping[str, Any],
        trace: Any | None = None,
        *,
        collection: str | None = None,
    ) -> int:
        """按 metadata 条件删除，返回删除条数。"""
        if not filters:
            raise VectorStoreError("delete_by_metadata 的 filter 不能为空")
        records = self.get_by_metadata(filters, trace=trace, collection=collection)
        ids = [item["id"] for item in records]
        if not ids:
            return 0
        target = self._get_collection(collection, create=False)
        if target is None:
            return 0
        try:
            target.delete(ids=ids)
        except Exception as exc:
            raise VectorStoreError(f"[chroma] delete_by_metadata 失败: {exc}") from exc
        return len(ids)

    def _records_from_get(self, raw: Mapping[str, Any]) -> list[dict[str, Any]]:
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
