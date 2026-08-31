"""Dashboard Query 追踪页渲染测试：历史列表、关键词框与阶段对比。"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest


@pytest.mark.unit
class TestQueryTracesPage:
    """验证空状态与查询详情渲染。"""

    def test_empty_file_shows_placeholder(self) -> None:
        """没有 query trace 时应展示空状态。"""

        def page_script() -> None:
            from pathlib import Path

            from observability.dashboard.pages.query_traces import render_query_traces
            from observability.dashboard.services.trace_service import TraceService

            render_query_traces(
                TraceService(Path("missing-query-traces.jsonl")),
                load_data=False,
            )

        app = AppTest.from_function(page_script, default_timeout=30)
        app.run()
        assert not app.exception
        assert any("暂无查询追踪记录" in str(item.value) for item in app.info)
        assert "按 Query 关键词筛选" in [item.label for item in app.text_input]

    def test_renders_history_comparison_and_expanders(self, tmp_path) -> None:
        """有 query trace 时应展示历史、Dense/Sparse 标题与阶段展开项。"""
        from observability.logger import write_trace

        path = tmp_path / "traces.jsonl"
        write_trace(
            {
                "trace_id": "q-page",
                "trace_type": "query",
                "started_at": "2026-08-31T05:00:00+00:00",
                "finished_at": "2026-08-31T05:00:01+00:00",
                "total_elapsed_ms": 20.0,
                "stages": [
                    {"name": "query_processing", "elapsed_ms": 2.0, "method": "keyword"},
                    {"name": "dense_retrieval", "elapsed_ms": 8.0, "method": "vector"},
                    {"name": "sparse_retrieval", "elapsed_ms": 5.0, "method": "bm25"},
                    {"name": "fusion", "elapsed_ms": 1.0, "method": "rrf"},
                    {"name": "rerank", "elapsed_ms": 4.0, "method": "none"},
                    {
                        "name": "query_complete",
                        "elapsed_ms": 0.0,
                        "query": "Azure 配置",
                        "collection": "knowledge_hub",
                        "dense_hits": [
                            {
                                "rank": 1,
                                "chunk_id": "d1",
                                "score": 0.9,
                                "title": "DenseDoc",
                                "source_path": "a.pdf",
                            }
                        ],
                        "sparse_hits": [
                            {
                                "rank": 1,
                                "chunk_id": "s1",
                                "score": 3.0,
                                "title": "SparseDoc",
                                "source_path": "b.pdf",
                            }
                        ],
                        "fusion_hits": [
                            {
                                "rank": 1,
                                "chunk_id": "s1",
                                "score": 0.03,
                                "title": "SparseDoc",
                                "source_path": "b.pdf",
                            },
                            {
                                "rank": 2,
                                "chunk_id": "d1",
                                "score": 0.02,
                                "title": "DenseDoc",
                                "source_path": "a.pdf",
                            },
                        ],
                        "rerank_hits": [
                            {
                                "rank": 1,
                                "chunk_id": "d1",
                                "score": 0.99,
                                "title": "DenseDoc",
                                "source_path": "a.pdf",
                            }
                        ],
                    },
                ],
            },
            path=path,
        )
        os.environ["G6_TRACE_FILE"] = str(path.resolve())
        try:

            def page_script() -> None:
                import os
                from pathlib import Path

                from observability.dashboard.pages.query_traces import render_query_traces
                from observability.dashboard.services.trace_service import TraceService

                trace_path = os.environ.get("G6_TRACE_FILE", "")
                render_query_traces(TraceService(Path(trace_path)), load_data=False)

            app = AppTest.from_function(page_script, default_timeout=30)
            app.run()
        finally:
            os.environ.pop("G6_TRACE_FILE", None)

        assert not app.exception
        assert "查看详情" in [item.label for item in app.selectbox]
        metrics = [item.label for item in app.metric]
        assert "状态" in metrics
        expander_labels = [str(item.label) for item in app.expander]
        assert any("query_processing" in label or "query_complete" in label for label in expander_labels)
        markdowns = [str(item.value) for item in app.markdown]
        assert any("Dense vs Sparse" in text for text in markdowns)
        assert any("Rerank 前后" in text for text in markdowns)
