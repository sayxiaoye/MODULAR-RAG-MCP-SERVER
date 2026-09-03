"""Dashboard 冒烟 E2E：六页面在有数据时均可渲染且无 Python 异常。

对应 spec I2：用 Streamlit AppTest 覆盖总览 / 数据浏览器 / Ingestion 管理 /
Ingestion 追踪 / Query 追踪 / 评估面板，以及 app.py 导航入口。
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[2]
DASHBOARD_APP = REPO_ROOT / "src" / "observability" / "dashboard" / "app.py"


def _assert_clean(app: AppTest) -> None:
    """冒烟最低标准：页面跑完没有未捕获异常。"""
    assert not app.exception, app.exception


@pytest.mark.e2e
class TestDashboardSmoke:
    """有注入数据时，六个页面都能完成一次渲染。"""

    def test_overview_with_stats(self) -> None:
        """总览页应展示组件卡片与集合统计。"""

        def page_script() -> None:
            from core.settings import load_settings
            from libs.vector_store.chroma_store import CollectionStats
            from observability.dashboard.pages.overview import render_overview
            from observability.dashboard.services.config_service import ConfigService

            render_overview(
                config_service=ConfigService(load_settings()),
                stats=CollectionStats(
                    collection="knowledge_hub",
                    chunk_count=8,
                    document_count=2,
                ),
                load_stats=False,
            )

        app = AppTest.from_function(page_script, default_timeout=30)
        app.run()
        _assert_clean(app)
        assert "系统总览" in [str(item.value) for item in app.header]
        assert "LLM" in [item.label for item in app.metric]

    def test_data_browser_with_documents(self) -> None:
        """数据浏览器有文档时应列出集合筛选与 chunk。"""

        def page_script() -> None:
            from observability.dashboard.pages.data_browser import render_data_browser
            from observability.dashboard.services.data_service import DataService

            class FakeChroma:
                def get_by_metadata(self, filters=None, trace=None, collection=None):
                    return [
                        {
                            "id": "chunk-smoke",
                            "text": "smoke chunk text",
                            "metadata": {
                                "source_path": "/data/smoke.pdf",
                                "collection": "knowledge_hub",
                                "chunk_index": 0,
                            },
                        }
                    ]

                def list_collection_names(self):
                    return ["knowledge_hub"]

            class FakeImages:
                def list_images(self, collection=None, doc_hash=None):
                    return []

                def get_path(self, image_id):
                    return None

            render_data_browser(
                DataService(FakeChroma(), FakeImages(), default_collection="knowledge_hub"),
                load_data=False,
            )

        app = AppTest.from_function(page_script, default_timeout=30)
        app.run()
        _assert_clean(app)
        assert "数据浏览器" in [str(item.value) for item in app.header]
        assert "集合" in [item.label for item in app.selectbox]

    def test_ingestion_manager_with_documents(self) -> None:
        """Ingestion 管理页有文档时应出现删除按钮。"""

        def page_script() -> None:
            from ingestion.document_manager import DeleteResult
            from ingestion.pipeline import IngestionResult
            from observability.dashboard.pages.ingestion_manager import render_ingestion_manager
            from observability.dashboard.services.data_service import DataService

            class FakeChroma:
                def get_by_metadata(self, filters=None, trace=None, collection=None):
                    return [
                        {
                            "id": "c1",
                            "text": "hello",
                            "metadata": {
                                "source_path": "/data/sample.pdf",
                                "collection": "knowledge_hub",
                            },
                        }
                    ]

            class FakeImages:
                def list_images(self, collection=None, doc_hash=None):
                    return []

                def get_path(self, image_id):
                    return None

            class FakePipeline:
                def run(self, source_path, collection="default", force=False, on_progress=None, trace=None):
                    return IngestionResult(
                        skipped=False,
                        source_path=source_path,
                        collection=collection,
                        chunk_count=1,
                    )

            class FakeManager:
                def delete_document(self, source_path, collection):
                    return DeleteResult(source_path, collection, 1, True, 0, True)

            render_ingestion_manager(
                pipeline=FakePipeline(),
                document_manager=FakeManager(),
                data_service=DataService(
                    FakeChroma(),
                    FakeImages(),
                    default_collection="knowledge_hub",
                ),
                load_deps=False,
                default_collection="knowledge_hub",
            )

        app = AppTest.from_function(page_script, default_timeout=30)
        app.run()
        _assert_clean(app)
        assert "Ingestion 管理" in [str(item.value) for item in app.header]
        assert "删除" in [item.label for item in app.button]

    def test_ingestion_traces_with_records(self) -> None:
        """Ingestion 追踪页有 trace 时应展示历史与详情选择。"""

        def page_script() -> None:
            import tempfile
            from pathlib import Path

            from observability.dashboard.pages.ingestion_traces import render_ingestion_traces
            from observability.dashboard.services.trace_service import TraceService
            from observability.logger import write_trace

            path = Path(tempfile.mkdtemp()) / "traces.jsonl"
            write_trace(
                {
                    "trace_id": "ing-smoke",
                    "trace_type": "ingestion",
                    "started_at": "2026-09-03T00:00:00+00:00",
                    "finished_at": "2026-09-03T00:00:01+00:00",
                    "total_elapsed_ms": 12.0,
                    "stages": [
                        {
                            "name": "load",
                            "elapsed_ms": 4.0,
                            "method": "markitdown",
                            "source_path": "/data/sample.pdf",
                        },
                        {
                            "name": "pipeline_complete",
                            "elapsed_ms": 0.0,
                            "chunk_count": 2,
                            "source_path": "/data/sample.pdf",
                            "collection": "knowledge_hub",
                        },
                    ],
                },
                path=path,
            )
            render_ingestion_traces(TraceService(path), load_data=False)

        app = AppTest.from_function(page_script, default_timeout=30)
        app.run()
        _assert_clean(app)
        assert "Ingestion 追踪" in [str(item.value) for item in app.header]
        assert "查看详情" in [item.label for item in app.selectbox]

    def test_query_traces_with_records(self) -> None:
        """Query 追踪页有 trace 时应展示筛选框与详情。"""

        def page_script() -> None:
            import tempfile
            from pathlib import Path

            from observability.dashboard.pages.query_traces import render_query_traces
            from observability.dashboard.services.trace_service import TraceService
            from observability.logger import write_trace

            path = Path(tempfile.mkdtemp()) / "traces.jsonl"
            write_trace(
                {
                    "trace_id": "q-smoke",
                    "trace_type": "query",
                    "started_at": "2026-09-03T00:00:00+00:00",
                    "finished_at": "2026-09-03T00:00:01+00:00",
                    "total_elapsed_ms": 9.0,
                    "stages": [
                        {"name": "dense_retrieval", "elapsed_ms": 5.0, "method": "vector"},
                        {
                            "name": "query_complete",
                            "elapsed_ms": 0.0,
                            "query": "smoke query",
                            "collection": "knowledge_hub",
                            "dense_hits": [
                                {
                                    "rank": 1,
                                    "chunk_id": "d1",
                                    "score": 0.8,
                                    "title": "Doc",
                                    "source_path": "a.pdf",
                                }
                            ],
                            "sparse_hits": [],
                            "fusion_hits": [],
                            "rerank_hits": [],
                        },
                    ],
                },
                path=path,
            )
            render_query_traces(TraceService(path), load_data=False)

        app = AppTest.from_function(page_script, default_timeout=30)
        app.run()
        _assert_clean(app)
        assert "Query 追踪" in [str(item.value) for item in app.header]
        assert "按 Query 关键词筛选" in [item.label for item in app.text_input]

    def test_evaluation_panel_with_golden_set(self) -> None:
        """评估面板应展示集合/后端/黄金集控件。"""

        def page_script() -> None:
            from core.settings import REPO_ROOT
            from observability.dashboard.pages.evaluation_panel import render_evaluation_panel

            render_evaluation_panel(
                golden_sets=[REPO_ROOT / "tests" / "fixtures" / "golden_test_set.json"],
                history_records=[
                    {
                        "ran_at": "2026-09-01T00:00:00+00:00",
                        "backend": "Custom",
                        "collection": "knowledge_hub",
                        "hit_rate": 0.5,
                        "mrr": 0.4,
                    },
                    {
                        "ran_at": "2026-09-02T00:00:00+00:00",
                        "backend": "All",
                        "collection": "knowledge_hub",
                        "hit_rate": 0.7,
                        "mrr": 0.6,
                    },
                ],
                collections=["knowledge_hub"],
                load_deps=False,
            )

        app = AppTest.from_function(page_script, default_timeout=30)
        app.run()
        _assert_clean(app)
        assert "评估面板" in [str(item.value) for item in app.header]
        labels = [item.label for item in app.selectbox]
        assert "集合" in labels
        assert "评估后端" in labels
        assert "运行评估" in [item.label for item in app.button]

    def test_app_entry_loads_without_exception(self) -> None:
        """app.py 导航入口应能完成默认页（系统总览）渲染。"""
        app = AppTest.from_file(str(DASHBOARD_APP), default_timeout=60)
        app.run()
        _assert_clean(app)
        headers = [str(item.value) for item in app.header]
        assert any("系统总览" in text for text in headers)
