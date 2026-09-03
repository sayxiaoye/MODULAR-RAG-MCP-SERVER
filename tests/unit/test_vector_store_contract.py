"""VectorStore 契约测试：约束 upsert/query 输入输出 shape 与工厂路由。"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import pytest

from core.settings import Settings, VectorStoreSettings, load_settings
from libs.vector_store.base_vector_store import BaseVectorStore, VectorStoreError
from libs.vector_store.vector_store_factory import (
    VectorStoreFactory,
    VectorStoreFactoryError,
    register_vector_store,
)


class InMemoryVectorStore(BaseVectorStore):
    """测试用内存向量库：用余弦相似度检索，验证契约而非真实 ANN 性能。"""

    def __init__(self, settings: VectorStoreSettings) -> None:
        self.settings = settings
        self._records: dict[str, dict[str, Any]] = {}

    def upsert(
        self,
        records: Sequence[Mapping[str, Any]],
        trace: Any | None = None,
        *,
        collection: str | None = None,
    ) -> None:
        validated = self._validate_upsert_records(records)
        for record in validated:
            self._records[str(record["id"])] = {
                "id": str(record["id"]),
                "text": str(record["text"]),
                "metadata": dict(record["metadata"]),
                "dense_vector": [float(v) for v in record["dense_vector"]],
            }

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
        scored: list[tuple[float, dict[str, Any]]] = []

        for record in self._records.values():
            if filters:
                metadata = record["metadata"]
                if not all(metadata.get(k) == v for k, v in filters.items()):
                    continue
            score = _cosine_similarity(query_vector, record["dense_vector"])
            scored.append(
                (
                    score,
                    {
                        "id": record["id"],
                        "score": score,
                        "text": record["text"],
                        "metadata": dict(record["metadata"]),
                    },
                )
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        results = [item[1] for item in scored[:top_k]]
        return self._validate_query_results(results)

    def get_by_ids(
        self,
        ids: Sequence[str],
        trace: Any | None = None,
        *,
        collection: str | None = None,
    ) -> list[dict[str, Any]]:
        """按 ID 从内存字典批量读取记录。"""
        if not ids:
            return []
        results: list[dict[str, Any]] = []
        for record_id in ids:
            record = self._records.get(str(record_id))
            if record is None:
                continue
            results.append(
                {
                    "id": record["id"],
                    "text": record["text"],
                    "metadata": dict(record["metadata"]),
                }
            )
        return self._validate_get_by_ids_results(results)

    def get_by_metadata(
        self,
        filters: Mapping[str, Any] | None = None,
        trace: Any | None = None,
        *,
        collection: str | None = None,
    ) -> list[dict[str, Any]]:
        """按 metadata 等值条件读取；collection 限定逻辑集合。"""
        results: list[dict[str, Any]] = []
        for record in self._iter_scoped(collection):
            if filters and not all(record["metadata"].get(k) == v for k, v in filters.items()):
                continue
            results.append(
                {
                    "id": record["id"],
                    "text": record["text"],
                    "metadata": dict(record["metadata"]),
                }
            )
        return self._validate_get_by_ids_results(results)

    def delete_by_metadata(
        self,
        filters: Mapping[str, Any],
        trace: Any | None = None,
        *,
        collection: str | None = None,
    ) -> int:
        """按 metadata 批量删除；空 filter 拒绝，避免误删全库。"""
        if not filters:
            raise VectorStoreError("delete_by_metadata 的 filter 不能为空")
        remaining: dict[str, dict[str, Any]] = {}
        deleted = 0
        scoped_ids = {record["id"] for record in self._iter_scoped(collection)}
        for record_id, record in self._records.items():
            if record_id not in scoped_ids:
                remaining[record_id] = record
                continue
            if all(record["metadata"].get(k) == v for k, v in filters.items()):
                deleted += 1
                continue
            remaining[record_id] = record
        self._records = remaining
        return deleted

    def _iter_scoped(self, collection: str | None) -> list[dict[str, Any]]:
        """collection 非空时只遍历该逻辑集合内的记录。"""
        records = list(self._records.values())
        if not collection:
            return records
        name = collection.strip()
        return [
            record
            for record in records
            if str(record["metadata"].get("collection") or "") == name
        ]


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """计算两向量余弦相似度，供 Fake 检索排序。"""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@pytest.fixture(autouse=True)
def _reset_vector_store_factory() -> None:
    """每个用例前后恢复工厂默认构造器。"""
    VectorStoreFactory.reset_constructor()
    yield
    VectorStoreFactory.reset_constructor()


@pytest.mark.unit
class TestVectorStoreContract:
    """验证 upsert/query 契约与内存实现 roundtrip。"""

    def test_upsert_and_query_roundtrip(self) -> None:
        """写入后应能按向量检索并返回完整字段。"""
        store = InMemoryVectorStore(load_settings().vector_store)
        records = [
            {
                "id": "chunk-001",
                "text": "Azure 配置指南",
                "metadata": {"source_path": "guide.pdf", "collection": "docs"},
                "dense_vector": [1.0, 0.0, 0.0],
            },
            {
                "id": "chunk-002",
                "text": "其他主题",
                "metadata": {"source_path": "other.pdf", "collection": "docs"},
                "dense_vector": [0.0, 1.0, 0.0],
            },
        ]
        store.upsert(records)
        results = store.query([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0]["id"] == "chunk-001"
        assert results[0]["text"] == "Azure 配置指南"
        assert "score" in results[0]
        assert results[0]["metadata"]["source_path"] == "guide.pdf"

    def test_query_with_metadata_filters(self) -> None:
        """filters 应只返回 metadata 匹配的记录。"""
        store = InMemoryVectorStore(load_settings().vector_store)
        store.upsert(
            [
                {
                    "id": "a",
                    "text": "t1",
                    "metadata": {"collection": "keep"},
                    "dense_vector": [1.0, 0.0],
                },
                {
                    "id": "b",
                    "text": "t2",
                    "metadata": {"collection": "skip"},
                    "dense_vector": [1.0, 0.0],
                },
            ]
        )
        results = store.query([1.0, 0.0], top_k=10, filters={"collection": "keep"})
        assert len(results) == 1
        assert results[0]["id"] == "a"

    def test_invalid_upsert_record_raises(self) -> None:
        """缺少 dense_vector 的记录应被拒绝。"""
        store = InMemoryVectorStore(load_settings().vector_store)
        with pytest.raises(VectorStoreError, match="dense_vector"):
            store.upsert([{"id": "x", "text": "t", "metadata": {}}])

    def test_get_by_ids_returns_matching_records(self) -> None:
        """get_by_ids 应返回指定 ID 的 text 与 metadata。"""
        store = InMemoryVectorStore(load_settings().vector_store)
        store.upsert(
            [
                {
                    "id": "chunk-001",
                    "text": "Azure 配置指南",
                    "metadata": {"source_path": "guide.pdf", "collection": "docs"},
                    "dense_vector": [1.0, 0.0, 0.0],
                },
                {
                    "id": "chunk-002",
                    "text": "其他主题",
                    "metadata": {"source_path": "other.pdf", "collection": "docs"},
                    "dense_vector": [0.0, 1.0, 0.0],
                },
            ]
        )
        results = store.get_by_ids(["chunk-001", "missing"])
        assert len(results) == 1
        assert results[0]["id"] == "chunk-001"
        assert results[0]["text"] == "Azure 配置指南"
        assert results[0]["metadata"]["source_path"] == "guide.pdf"

    def test_empty_upsert_raises(self) -> None:
        """空 records 违反 upsert 契约，应拒绝写入。"""
        store = InMemoryVectorStore(load_settings().vector_store)
        with pytest.raises(VectorStoreError, match="records 不能为空"):
            store.upsert([])

    def test_invalid_top_k_raises(self) -> None:
        """top_k <= 0 不是合法检索形状。"""
        store = InMemoryVectorStore(load_settings().vector_store)
        store.upsert(
            [
                {
                    "id": "c1",
                    "text": "t",
                    "metadata": {},
                    "dense_vector": [1.0],
                }
            ]
        )
        with pytest.raises(VectorStoreError, match="top_k"):
            store.query([1.0], top_k=0)


@pytest.mark.unit
class TestVectorStoreDeleteByMetadata:
    """I4：delete_by_metadata 空 filter、零命中、集合隔离与删除计数。"""

    def _store_with_two_collections(self) -> InMemoryVectorStore:
        store = InMemoryVectorStore(load_settings().vector_store)
        store.upsert(
            [
                {
                    "id": "keep-a",
                    "text": "保留 A",
                    "metadata": {"source_path": "keep.pdf", "collection": "docs"},
                    "dense_vector": [1.0, 0.0],
                },
                {
                    "id": "drop-a",
                    "text": "删除 A",
                    "metadata": {"source_path": "drop.pdf", "collection": "docs"},
                    "dense_vector": [0.0, 1.0],
                },
                {
                    "id": "other-drop",
                    "text": "另一集合同名源",
                    "metadata": {"source_path": "drop.pdf", "collection": "other"},
                    "dense_vector": [0.0, 1.0],
                },
            ]
        )
        return store

    def test_empty_filters_raise(self) -> None:
        """空 filter 禁止全量删除，与 ChromaStore 契约一致。"""
        store = self._store_with_two_collections()
        with pytest.raises(VectorStoreError, match="filter 不能为空"):
            store.delete_by_metadata({})

    def test_no_match_returns_zero(self) -> None:
        """无匹配记录时应返回 0，且不改动已有数据。"""
        store = self._store_with_two_collections()
        deleted = store.delete_by_metadata({"source_path": "missing.pdf"})
        assert deleted == 0
        assert len(store.get_by_ids(["keep-a", "drop-a", "other-drop"])) == 3

    def test_deletes_matching_records_and_returns_count(self) -> None:
        """命中记录应删除，返回条数，query 不再命中。"""
        store = self._store_with_two_collections()
        deleted = store.delete_by_metadata({"source_path": "drop.pdf"})
        assert deleted == 2
        remaining = store.get_by_ids(["keep-a", "drop-a", "other-drop"])
        assert [item["id"] for item in remaining] == ["keep-a"]
        hits = store.query([0.0, 1.0], top_k=5)
        assert [item["id"] for item in hits] == ["keep-a"]

    def test_collection_scopes_delete(self) -> None:
        """collection 参数只删除该逻辑集合内的匹配记录。"""
        store = self._store_with_two_collections()
        deleted = store.delete_by_metadata(
            {"source_path": "drop.pdf"},
            collection="docs",
        )
        assert deleted == 1
        assert store.get_by_ids(["drop-a"]) == []
        leftover = store.get_by_ids(["other-drop"])
        assert len(leftover) == 1
        assert leftover[0]["metadata"]["collection"] == "other"


@pytest.mark.unit
class TestVectorStoreFactoryRouting:
    """验证工厂按 vector_store.provider 路由。"""

    def test_fake_provider_routing(self) -> None:
        """注册 fake provider 后应返回 InMemoryVectorStore。"""
        register_vector_store("fake", InMemoryVectorStore)
        base = load_settings()
        fake_settings = Settings(
            llm=base.llm,
            embedding=base.embedding,
            vector_store=VectorStoreSettings(
                provider="fake",
                persist_directory="./data/db/chroma",
                collection_name="test",
            ),
            retrieval=base.retrieval,
            rerank=base.rerank,
            evaluation=base.evaluation,
            observability=base.observability,
            ingestion=base.ingestion,
            vision_llm=base.vision_llm,
        )
        store = VectorStoreFactory.create(fake_settings)
        assert isinstance(store, InMemoryVectorStore)

    def test_unknown_provider_raises(self) -> None:
        """未注册的 provider 应抛出可读错误。"""
        base = load_settings()
        unknown = Settings(
            llm=base.llm,
            embedding=base.embedding,
            vector_store=VectorStoreSettings(
                provider="unknown_vs_xyz",
                persist_directory="./data",
                collection_name="test",
            ),
            retrieval=base.retrieval,
            rerank=base.rerank,
            evaluation=base.evaluation,
            observability=base.observability,
            ingestion=base.ingestion,
            vision_llm=base.vision_llm,
        )
        with pytest.raises(VectorStoreFactoryError, match="unknown_vs_xyz"):
            VectorStoreFactory.create(unknown)
