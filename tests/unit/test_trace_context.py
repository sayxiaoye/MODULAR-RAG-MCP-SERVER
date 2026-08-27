"""TraceContext / TraceCollector 单元测试：finish、耗时、to_dict 与收集。"""

from __future__ import annotations

import json
import time

import pytest

from core.trace.trace_collector import TraceCollector
from core.trace.trace_context import TraceContext


@pytest.mark.unit
class TestTraceContextFinishAndSerialize:
    """验证 F1：finish 后 to_dict 字段完整且可 JSON 序列化。"""

    def test_default_trace_type_is_query(self) -> None:
        """未指定类型时应默认为 query。"""
        trace = TraceContext()
        assert trace.trace_type == "query"

    def test_invalid_trace_type_raises(self) -> None:
        with pytest.raises(ValueError, match="trace_type"):
            TraceContext(trace_type="other")

    def test_finish_then_to_dict_contains_required_fields(self) -> None:
        """finish 后 to_dict 应含 trace_id / trace_type / 时间戳 / 总耗时 / stages。"""
        trace = TraceContext(trace_type="query")
        trace.record_stage("dense", elapsed_ms=12.5, method="chroma")
        time.sleep(0.01)
        trace.finish()

        payload = trace.to_dict()
        assert payload["trace_id"] == trace.trace_id
        assert payload["trace_type"] == "query"
        assert payload["started_at"]
        assert payload["finished_at"]
        assert payload["total_elapsed_ms"] >= 10
        assert payload["stages"][0]["name"] == "dense"
        assert payload["stages"][0]["elapsed_ms"] == 12.5

        serialized = json.dumps(payload, ensure_ascii=False)
        restored = json.loads(serialized)
        assert restored["trace_id"] == payload["trace_id"]
        assert restored["trace_type"] == "query"

    def test_finish_is_idempotent(self) -> None:
        """重复 finish 不应改写总耗时。"""
        trace = TraceContext(trace_type="ingestion")
        trace.finish()
        first = trace.to_dict()["total_elapsed_ms"]
        time.sleep(0.01)
        trace.finish()
        assert trace.to_dict()["total_elapsed_ms"] == first

    def test_elapsed_ms_by_stage_and_total(self) -> None:
        """elapsed_ms 可按阶段名查询，缺省返回总耗时。"""
        trace = TraceContext(trace_type="query")
        trace.record_stage("sparse", elapsed_ms=7.0)
        assert trace.elapsed_ms("sparse") == 7.0
        assert trace.elapsed_ms("missing") == 0.0
        assert trace.elapsed_ms() >= 0.0
        trace.finish()
        assert trace.elapsed_ms() == trace.to_dict()["total_elapsed_ms"]


@pytest.mark.unit
class TestTraceCollector:
    """验证 collect 会 finish 并触发 persist 钩子。"""

    def test_collect_finishes_and_stores_payload(self) -> None:
        persisted: list[dict] = []
        collector = TraceCollector(persist_fn=persisted.append)
        trace = TraceContext(trace_type="ingestion")
        trace.record_stage("load", elapsed_ms=3.0)

        collector.collect(trace)

        assert trace.is_finished
        assert len(collector.traces) == 1
        assert collector.traces[0]["trace_type"] == "ingestion"
        assert persisted[0]["stages"][0]["name"] == "load"
        json.dumps(persisted[0])
