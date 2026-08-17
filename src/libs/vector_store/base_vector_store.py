"""VectorStore 抽象层：定义向量写入与检索契约，供摄取与 Dense 检索复用。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Sequence

# upsert 记录与 query 结果必须包含的字段（C1 ChunkRecord / D2 RetrievalResult 对齐）
_REQUIRED_UPSERT_KEYS = frozenset({"id", "text", "metadata", "dense_vector"})
_REQUIRED_QUERY_KEYS = frozenset({"id", "score", "text", "metadata"})
_REQUIRED_GET_BY_IDS_KEYS = frozenset({"id", "text", "metadata"})


class VectorStoreError(Exception):
    """向量库调用或记录契约校验失败时抛出。"""


class BaseVectorStore(ABC):
    """VectorStore 抽象基类：屏蔽 Chroma 等后端的存储与检索差异。"""

    @abstractmethod
    def upsert(
        self,
        records: Sequence[Mapping[str, Any]],
        trace: Any | None = None,
    ) -> None:
        """
        批量写入或更新向量记录（幂等由具体实现保证）。

        Args:
            records: 记录列表，每条至少含 id/text/metadata/dense_vector。
            trace: 可选追踪上下文（F 阶段注入）。
        """

    @abstractmethod
    def query(
        self,
        vector: Sequence[float],
        top_k: int,
        filters: Mapping[str, Any] | None = None,
        trace: Any | None = None,
    ) -> list[dict[str, Any]]:
        """
        按稠密向量检索 Top-K 相似记录。

        Args:
            vector: 查询向量。
            top_k: 返回条数上限。
            filters: 可选 metadata 过滤条件。
            trace: 可选追踪上下文。

        Returns:
            结果列表，每项含 id、score、text、metadata 字段。
        """

    @abstractmethod
    def get_by_ids(
        self,
        ids: Sequence[str],
        trace: Any | None = None,
    ) -> list[dict[str, Any]]:
        """
        按 chunk_id 批量获取文本与 metadata（供 SparseRetriever 回填正文）。

        Args:
            ids: chunk_id 列表。
            trace: 可选追踪上下文。

        Returns:
            记录列表，每项含 id、text、metadata 字段（顺序不保证与输入一致）。
        """

    def _validate_upsert_records(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> list[Mapping[str, Any]]:
        """校验 upsert 输入 shape，确保后续存储链路字段完整。"""
        if not records:
            raise VectorStoreError("records 不能为空")
        validated: list[Mapping[str, Any]] = []
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise VectorStoreError(f"records[{index}] 必须是 mapping")
            missing = _REQUIRED_UPSERT_KEYS - set(record.keys())
            if missing:
                raise VectorStoreError(
                    f"records[{index}] 缺少字段: {', '.join(sorted(missing))}"
                )
            dense_vector = record["dense_vector"]
            if not isinstance(dense_vector, Sequence) or isinstance(dense_vector, str):
                raise VectorStoreError(f"records[{index}].dense_vector 必须是数值序列")
            if not all(isinstance(v, (int, float)) for v in dense_vector):
                raise VectorStoreError(f"records[{index}].dense_vector 元素必须是数值")
            if not isinstance(record["metadata"], Mapping):
                raise VectorStoreError(f"records[{index}].metadata 必须是 mapping")
            validated.append(record)
        return validated

    def _validate_query_vector(self, vector: Sequence[float], top_k: int) -> list[float]:
        """校验 query 向量与 top_k 参数。"""
        if top_k <= 0:
            raise VectorStoreError("top_k 必须大于 0")
        if not isinstance(vector, Sequence) or isinstance(vector, str):
            raise VectorStoreError("vector 必须是数值序列")
        if not vector:
            raise VectorStoreError("vector 不能为空")
        floats: list[float] = []
        for index, value in enumerate(vector):
            if not isinstance(value, (int, float)):
                raise VectorStoreError(f"vector[{index}] 必须是数值")
            floats.append(float(value))
        return floats

    def _validate_query_results(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """校验 query 输出契约，供契约测试与实现自检。"""
        for index, item in enumerate(results):
            if not isinstance(item, dict):
                raise VectorStoreError(f"query 结果[{index}] 必须是 dict")
            missing = _REQUIRED_QUERY_KEYS - set(item.keys())
            if missing:
                raise VectorStoreError(
                    f"query 结果[{index}] 缺少字段: {', '.join(sorted(missing))}"
                )
            if not isinstance(item["metadata"], Mapping):
                raise VectorStoreError(f"query 结果[{index}].metadata 必须是 mapping")
        return results

    def _validate_get_by_ids_results(
        self,
        results: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """校验 get_by_ids 输出契约。"""
        for index, item in enumerate(results):
            if not isinstance(item, dict):
                raise VectorStoreError(f"get_by_ids 结果[{index}] 必须是 dict")
            missing = _REQUIRED_GET_BY_IDS_KEYS - set(item.keys())
            if missing:
                raise VectorStoreError(
                    f"get_by_ids 结果[{index}] 缺少字段: {', '.join(sorted(missing))}"
                )
            if not isinstance(item["metadata"], Mapping):
                raise VectorStoreError(f"get_by_ids 结果[{index}].metadata 必须是 mapping")
        return results
