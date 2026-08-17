"""VectorUpserter 幂等性测试：稳定 ID 与批量写入顺序。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import pytest

from core.types import Chunk
from ingestion.storage.vector_upserter import (
    generate_stable_chunk_id,
    VectorUpserter,
    VectorUpserterError,
)
from libs.vector_store.base_vector_store import BaseVectorStore


class RecordingVectorStore(BaseVectorStore):
    """记录 upsert 调用次数与最终存储条数的 Fake VectorStore。"""

    def __init__(self) -> None:
        self.upsert_calls = 0
        self._records: dict[str, dict[str, Any]] = {}

    def upsert(
        self,
        records: Sequence[Mapping[str, Any]],
        trace: Any | None = None,
    ) -> None:
        validated = self._validate_upsert_records(records)
        self.upsert_calls += 1
        for record in validated:
            record_id = str(record["id"])
            self._records[record_id] = {
                "id": record_id,
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
    ) -> list[dict[str, Any]]:
        return []

    def get_by_ids(
        self,
        ids: Sequence[str],
        trace: Any | None = None,
    ) -> list[dict[str, Any]]:
        return []

    @property
    def record_count(self) -> int:
        return len(self._records)

    def get_record(self, record_id: str) -> dict[str, Any] | None:
        return self._records.get(record_id)


def _chunk(text: str, source_path: str, chunk_index: int, chunk_id: str = "legacy_id") -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={"source_path": source_path, "chunk_index": chunk_index},
        start_offset=0,
        end_offset=len(text),
    )


@pytest.mark.unit
class TestVectorUpserterIdempotency:
    """验证稳定 ID 生成、重复 upsert 与批量顺序。"""

    def test_same_chunk_twice_upsert_produces_same_id(self) -> None:
        store = RecordingVectorStore()
        upserter = VectorUpserter(store)
        chunk = _chunk("Azure OpenAI 配置", "docs/guide.pdf", 0)
        vector = [0.1, 0.2, 0.3]

        first_ids = upserter.upsert([chunk], [vector])
        second_ids = upserter.upsert([chunk], [vector])

        assert first_ids == second_ids
        assert store.record_count == 1
        assert store.upsert_calls == 2

    def test_content_change_produces_different_id(self) -> None:
        source_path = "docs/guide.pdf"
        chunk_index = 1
        id_original = generate_stable_chunk_id(source_path, chunk_index, "原始内容")
        id_changed = generate_stable_chunk_id(source_path, chunk_index, "变更后内容")
        assert id_original != id_changed

    def test_batch_upsert_preserves_order(self) -> None:
        store = RecordingVectorStore()
        upserter = VectorUpserter(store)
        chunks = [
            _chunk("第一段", "docs/a.pdf", 0, "c0"),
            _chunk("第二段", "docs/a.pdf", 1, "c1"),
            _chunk("第三段", "docs/a.pdf", 2, "c2"),
        ]
        vectors = [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ]

        ids = upserter.upsert(chunks, vectors)

        assert len(ids) == 3
        assert ids[0] == generate_stable_chunk_id("docs/a.pdf", 0, "第一段")
        assert ids[1] == generate_stable_chunk_id("docs/a.pdf", 1, "第二段")
        assert ids[2] == generate_stable_chunk_id("docs/a.pdf", 2, "第三段")
        assert store.record_count == 3

    def test_missing_source_path_raises(self) -> None:
        upserter = VectorUpserter(RecordingVectorStore())
        chunk = Chunk(
            id="bad",
            text="无路径",
            metadata={"chunk_index": 0},
            start_offset=0,
            end_offset=2,
        )
        with pytest.raises(VectorUpserterError, match="source_path"):
            upserter.upsert([chunk], [[0.1, 0.2]])

    def test_mismatched_lengths_raise(self) -> None:
        upserter = VectorUpserter(RecordingVectorStore())
        chunk = _chunk("内容", "docs/x.pdf", 0)
        with pytest.raises(VectorUpserterError, match="不一致"):
            upserter.upsert([chunk], [])

    def test_empty_input_returns_empty(self) -> None:
        upserter = VectorUpserter(RecordingVectorStore())
        assert upserter.upsert([], []) == []
