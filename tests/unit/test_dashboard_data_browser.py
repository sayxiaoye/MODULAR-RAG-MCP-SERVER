"""Dashboard 数据浏览器渲染测试：用 AppTest 验证列表与空状态。"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest


@pytest.mark.unit
class TestDataBrowserPage:
    """验证数据浏览器渲染文档列表、筛选控件与空状态。"""

    def test_empty_store_shows_placeholder(self) -> None:
        """无文档时应展示空状态提示。"""

        def page_script() -> None:
            # AppTest.from_function 会抽取源码到临时脚本，必须在函数内导入
            from observability.dashboard.pages.data_browser import render_data_browser
            from observability.dashboard.services.data_service import DataService

            class FakeChroma:
                def get_by_metadata(self, filters=None, trace=None, collection=None):
                    return []

            class FakeImageStorage:
                def list_images(self, collection=None, doc_hash=None):
                    return []

                def get_path(self, image_id):
                    return None

            render_data_browser(
                DataService(FakeChroma(), FakeImageStorage()),
                load_data=False,
            )

        app = AppTest.from_function(page_script, default_timeout=30)
        app.run()
        assert not app.exception
        texts = [item.value for item in app.info]
        assert any("暂无已摄入文档" in str(text) for text in texts)

    def test_renders_document_list_and_chunk_expanders(self) -> None:
        """有文档时应展示集合筛选、文档列表与 chunk 折叠项。"""

        def page_script() -> None:
            from observability.dashboard.pages.data_browser import render_data_browser
            from observability.dashboard.services.data_service import DataService

            class FakeChroma:
                def get_by_metadata(self, filters=None, trace=None, collection=None):
                    records = [
                        {
                            "id": "chunk-aaa",
                            "text": "hello world from sample pdf",
                            "metadata": {
                                "source_path": "/data/sample.pdf",
                                "collection": "knowledge_hub",
                                "chunk_index": 0,
                            },
                        }
                    ]
                    if not filters:
                        return list(records)
                    return [
                        item
                        for item in records
                        if all(
                            item.get("metadata", {}).get(key) == value
                            for key, value in filters.items()
                        )
                    ]

            class FakeImageStorage:
                def list_images(self, collection=None, doc_hash=None):
                    return []

                def get_path(self, image_id):
                    return None

            render_data_browser(
                DataService(
                    FakeChroma(),
                    FakeImageStorage(),
                    default_collection="knowledge_hub",
                ),
                load_data=False,
            )

        app = AppTest.from_function(page_script, default_timeout=30)
        app.run()
        assert not app.exception

        select_labels = [item.label for item in app.selectbox]
        assert "集合" in select_labels
        input_labels = [item.label for item in app.text_input]
        assert "关键词搜索" in input_labels

        expander_labels = [item.label for item in app.expander]
        assert any("sample.pdf" in str(label) for label in expander_labels)
        assert any("hello world from sample pdf" in str(label) for label in expander_labels)
