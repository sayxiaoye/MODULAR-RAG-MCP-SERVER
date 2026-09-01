"""TraceService 单元测试：解析 traces.jsonl、过滤类型、跳过损坏行。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from observability.dashboard.services.trace_service import TraceService, parse_trace_payload
from observability.logger import write_trace


def _ingestion_payload(
    trace_id: str,
    *,
    started_at: str,
    source: str = "/data/a.pdf",
    collection: str = "knowledge_hub",
    finished: bool = True,
) -> dict:
    stages = [
        {"name": "load", "elapsed_ms": 10.0, "method": "markitdown", "source_path": source},
        {"name": "split", "elapsed_ms": 5.0, "method": "recursive", "chunk_count": 3},
        {"name": "transform", "elapsed_ms": 8.0, "method": "sequential"},
        {"name": "embed", "elapsed_ms": 20.0, "method": "llamacpp", "provider": "llamacpp"},
        {"name": "upsert", "elapsed_ms": 4.0, "method": "chroma", "provider": "chroma", "vector_count": 3},
        {
            "name": "pipeline_complete",
            "elapsed_ms": 0.0,
            "method": "pipeline",
            "chunk_count": 3,
            "image_count": 1,
            "source_path": source,
            "collection": collection,
        },
    ]
    return {
        "trace_id": trace_id,
        "trace_type": "ingestion",
        "started_at": started_at,
        "finished_at": "2026-08-31T04:00:01+00:00" if finished else None,
        "total_elapsed_ms": 47.0,
        "stages": stages,
    }


@pytest.mark.unit
class TestTraceService:
    """验证 JSONL 解析、倒序、类型过滤与瀑布图行。"""

    def test_list_traces_skips_log_lines_and_sorts_desc(self, tmp_path: Path) -> None:
        """应忽略 JSONFormatter 日志行与坏 JSON，并按 started_at 倒序。"""
        path = tmp_path / "traces.jsonl"
        write_trace(_ingestion_payload("old", started_at="2026-08-30T00:00:00+00:00"), path=path)
        path.write_text(
            path.read_text(encoding="utf-8")
            + json.dumps({"timestamp": "x", "level": "INFO", "logger": "x", "message": "hi"})
            + "\nnot-json\n",
            encoding="utf-8",
        )
        write_trace(_ingestion_payload("new", started_at="2026-08-31T00:00:00+00:00"), path=path)
        write_trace(
            {
                "trace_id": "q1",
                "trace_type": "query",
                "started_at": "2026-08-31T01:00:00+00:00",
                "finished_at": "2026-08-31T01:00:01+00:00",
                "total_elapsed_ms": 1.0,
                "stages": [{"name": "dense", "elapsed_ms": 1.0}],
            },
            path=path,
        )

        service = TraceService(path)
        ingestion = service.list_traces("ingestion")
        assert [item.trace_id for item in ingestion] == ["new", "old"]
        all_types = service.list_traces()
        assert {item.trace_type for item in all_types} == {"ingestion", "query"}

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        """jsonl 不存在时应返回空列表而不是抛错。"""
        service = TraceService(tmp_path / "missing.jsonl")
        assert service.list_traces("ingestion") == []

    def test_record_extracts_source_collection_and_waterfall(self) -> None:
        """应从 pipeline_complete / 规范阶段提取文件名、集合、chunk 数与瀑布图。"""
        payload = _ingestion_payload("t1", started_at="2026-08-31T00:00:00+00:00")
        record = parse_trace_payload(payload)
        assert record is not None
        assert record.status == "成功"
        assert record.source_name == "a.pdf"
        assert record.collection == "knowledge_hub"
        assert record.chunk_count == 3
        assert record.image_count == 1
        assert [name for name, _ in record.waterfall_rows()] == [
            "load",
            "split",
            "transform",
            "embed",
            "upsert",
        ]

    def test_skipped_and_error_status(self) -> None:
        """skipped / error 阶段应映射为跳过与失败。"""
        skipped = parse_trace_payload(
            {
                "trace_id": "s1",
                "trace_type": "ingestion",
                "started_at": "t",
                "finished_at": "t2",
                "total_elapsed_ms": 1,
                "stages": [{"name": "skipped", "elapsed_ms": 0, "source_path": "/x.pdf"}],
            }
        )
        failed = parse_trace_payload(
            {
                "trace_id": "e1",
                "trace_type": "ingestion",
                "started_at": "t",
                "finished_at": "t2",
                "total_elapsed_ms": 1,
                "stages": [{"name": "error", "elapsed_ms": 0, "error": "boom"}],
            }
        )
        assert skipped is not None and skipped.status == "跳过"
        assert failed is not None and failed.status == "失败"

    def test_deleted_status_from_deleted_stage(self) -> None:
        """deleted 阶段应映射为删除，并带出文件、集合与 chunk 数。"""
        record = parse_trace_payload(
            {
                "trace_id": "d1",
                "trace_type": "ingestion",
                "started_at": "t",
                "finished_at": "t2",
                "total_elapsed_ms": 2.5,
                "stages": [
                    {
                        "name": "deleted",
                        "elapsed_ms": 0,
                        "method": "document_manager",
                        "source_path": "/data/week1.pdf",
                        "collection": "col_b",
                        "chunk_count": 19,
                    }
                ],
            }
        )
        assert record is not None
        assert record.status == "删除"
        assert record.source_name == "week1.pdf"
        assert record.collection == "col_b"
        assert record.chunk_count == 19

    def test_get_trace_by_id(self, tmp_path: Path) -> None:
        """get_trace 应按 trace_id 返回单条记录。"""
        path = tmp_path / "traces.jsonl"
        write_trace(_ingestion_payload("abc", started_at="2026-08-31T00:00:00+00:00"), path=path)
        service = TraceService(path)
        found = service.get_trace("abc")
        assert found is not None
        assert found.trace_id == "abc"
        assert service.get_trace("missing") is None


def _query_payload() -> dict:
    return {
        "trace_id": "q-1",
        "trace_type": "query",
        "started_at": "2026-08-31T05:00:00+00:00",
        "finished_at": "2026-08-31T05:00:01+00:00",
        "total_elapsed_ms": 33.0,
        "stages": [
            {"name": "query_processor", "elapsed_ms": 2.0, "method": "keyword"},
            {"name": "dense_retriever", "elapsed_ms": 10.0, "method": "vector"},
            {"name": "sparse_retriever", "elapsed_ms": 8.0, "method": "bm25"},
            {"name": "fusion", "elapsed_ms": 1.0, "method": "rrf"},
            {"name": "rerank", "elapsed_ms": 12.0, "method": "none", "provider": "none"},
            {
                "name": "query_complete",
                "elapsed_ms": 0.0,
                "query": "Azure 配置指南",
                "collection": "knowledge_hub",
                "dense_hits": [
                    {"rank": 1, "chunk_id": "d1", "score": 0.9, "title": "Dense", "source_path": "a.pdf"},
                ],
                "sparse_hits": [
                    {"rank": 1, "chunk_id": "s1", "score": 4.2, "title": "Sparse", "source_path": "b.pdf"},
                ],
                "fusion_hits": [
                    {"rank": 1, "chunk_id": "s1", "score": 0.03, "title": "Sparse", "source_path": "b.pdf"},
                    {"rank": 2, "chunk_id": "d1", "score": 0.02, "title": "Dense", "source_path": "a.pdf"},
                ],
                "rerank_hits": [
                    {"rank": 1, "chunk_id": "d1", "score": 0.99, "title": "Dense", "source_path": "a.pdf"},
                    {"rank": 2, "chunk_id": "s1", "score": 0.1, "title": "Sparse", "source_path": "b.pdf"},
                ],
            },
        ],
    }


@pytest.mark.unit
class TestQueryTraceHelpers:
    """验证 Query 瀑布图别名、关键词筛选与 Rerank 排名变化。"""

    def test_query_waterfall_uses_component_aliases(self) -> None:
        """dense_retriever 等组件名应映射到 F3 规范阶段。"""
        record = parse_trace_payload(_query_payload())
        assert record is not None
        names = [name for name, _ in record.query_waterfall_rows()]
        assert names == [
            "query_processing",
            "dense_retrieval",
            "sparse_retrieval",
            "fusion",
            "rerank",
        ]

    def test_keyword_filter_matches_query_text(self, tmp_path: Path) -> None:
        """list_traces 应按 Query 文本过滤。"""
        path = tmp_path / "traces.jsonl"
        write_trace(_query_payload(), path=path)
        write_trace(
            {
                "trace_id": "q-2",
                "trace_type": "query",
                "started_at": "2026-08-31T06:00:00+00:00",
                "finished_at": "2026-08-31T06:00:01+00:00",
                "total_elapsed_ms": 1,
                "stages": [{"name": "query_complete", "query": "完全无关的问题"}],
            },
            path=path,
        )
        service = TraceService(path)
        matched = service.list_traces("query", keyword="Azure")
        assert [item.trace_id for item in matched] == ["q-1"]

    def test_rerank_rank_changes_marks_jumps(self) -> None:
        """精排后排名上升应标记 ↑，下降标记 ↓。"""
        record = parse_trace_payload(_query_payload())
        assert record is not None
        changes = {item["chunk_id"]: item for item in record.rerank_rank_changes()}
        assert changes["d1"]["fusion_rank"] == 2
        assert changes["d1"]["rerank_rank"] == 1
        assert changes["d1"]["delta"] == 1
        assert changes["d1"]["mark"] == "↑"
        assert changes["s1"]["mark"] == "↓"
        assert record.lane_hits("dense")[0]["chunk_id"] == "d1"
