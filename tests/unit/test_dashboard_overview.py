"""Dashboard 总览页渲染测试：用 AppTest 验证配置卡片与统计指标。"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest


@pytest.mark.unit
class TestOverviewPage:
    """验证总览页渲染组件配置与注入的集合统计。"""

    def test_overview_renders_component_metrics(self) -> None:
        """总览页应展示 LLM 等组件 metric，以及注入的集合统计。"""
        def page_script() -> None:
            # AppTest.from_function 抽取源码后无测试模块全局名，需在函数内导入
            from core.settings import load_settings
            from libs.vector_store.chroma_store import CollectionStats
            from observability.dashboard.pages.overview import render_overview
            from observability.dashboard.services.config_service import ConfigService

            render_overview(
                config_service=ConfigService(load_settings()),
                stats=CollectionStats(
                    collection="knowledge_hub",
                    chunk_count=12,
                    document_count=3,
                ),
                load_stats=False,
            )

        app = AppTest.from_function(page_script, default_timeout=30)
        app.run()
        assert not app.exception

        labels = [item.label for item in app.metric]
        values = [str(item.value) for item in app.metric]
        assert "LLM" in labels
        assert "Embedding" in labels
        assert "集合" in labels
        assert "knowledge_hub" in values
        assert "12" in values
        assert "3" in values
