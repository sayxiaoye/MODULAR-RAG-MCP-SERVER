"""Dashboard Ingestion 追踪页渲染测试：历史列表与阶段详情。"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest


@pytest.mark.unit
class TestIngestionTracesPage:
    """验证空状态、历史表与阶段折叠项。"""

    def test_empty_file_shows_placeholder(self) -> None:
        """没有 ingestion trace 时应展示空状态。"""

        def page_script() -> None:
            from pathlib import Path

            from observability.dashboard.pages.ingestion_traces import render_ingestion_traces
            from observability.dashboard.services.trace_service import TraceService

            render_ingestion_traces(
                TraceService(Path("missing-traces.jsonl")),
                load_data=False,
            )

        app = AppTest.from_function(page_script, default_timeout=30)
        app.run()
        assert not app.exception
        assert any("暂无摄取追踪记录" in str(item.value) for item in app.info)

    def test_renders_history_and_stage_expanders(self, tmp_path) -> None:
        """有 ingestion trace 时应展示历史与 load 等阶段展开项。"""
        from observability.logger import write_trace

        path = tmp_path / "traces.jsonl"
        write_trace(
            {
                "trace_id": "trace-1",
                "trace_type": "ingestion",
                "started_at": "2026-08-31T04:00:00+00:00",
                "finished_at": "2026-08-31T04:00:01+00:00",
                "total_elapsed_ms": 42.5,
                "stages": [
                    {
                        "name": "load",
                        "elapsed_ms": 10.0,
                        "method": "markitdown",
                        "source_path": "/data/sample.pdf",
                    },
                    {"name": "split", "elapsed_ms": 5.0, "method": "recursive", "chunk_count": 2},
                    {"name": "transform", "elapsed_ms": 3.0, "method": "sequential"},
                    {"name": "embed", "elapsed_ms": 20.0, "method": "llamacpp", "provider": "llamacpp"},
                    {"name": "upsert", "elapsed_ms": 4.0, "method": "chroma", "provider": "chroma"},
                    {
                        "name": "pipeline_complete",
                        "elapsed_ms": 0.0,
                        "chunk_count": 2,
                        "image_count": 0,
                        "source_path": "/data/sample.pdf",
                        "collection": "knowledge_hub",
                    },
                ],
            },
            path=path,
        )
        os.environ["G5_TRACE_FILE"] = str(path.resolve())
        try:

            def page_script() -> None:
                import os
                from pathlib import Path

                from observability.dashboard.pages.ingestion_traces import render_ingestion_traces
                from observability.dashboard.services.trace_service import TraceService

                # AppTest 抽取源码后无法闭包捕获路径，通过环境变量注入
                trace_path = os.environ.get("G5_TRACE_FILE", "")
                render_ingestion_traces(TraceService(Path(trace_path)), load_data=False)

            app = AppTest.from_function(page_script, default_timeout=30)
            app.run()
        finally:
            os.environ.pop("G5_TRACE_FILE", None)

        assert not app.exception
        assert "查看详情" in [item.label for item in app.selectbox]
        metrics = [item.label for item in app.metric]
        assert "状态" in metrics
        assert "Chunk 数" in metrics
        expander_labels = [str(item.label) for item in app.expander]
        assert any(label.startswith("load") for label in expander_labels)
        assert any("markitdown" in label for label in expander_labels)
