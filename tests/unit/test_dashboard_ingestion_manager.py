"""Dashboard Ingestion 管理页渲染测试：摄取触发、进度与删除按钮。"""

from __future__ import annotations

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest


@pytest.mark.unit
class TestIngestionManagerPage:
    """验证管理页控件、空列表、摄取成功提示与删除按钮。"""

    def test_empty_list_shows_uploader_and_placeholder(self) -> None:
        """无文档时应展示上传控件与空状态提示。"""

        def page_script() -> None:
            from ingestion.document_manager import DeleteResult
            from ingestion.pipeline import IngestionResult
            from observability.dashboard.pages.ingestion_manager import render_ingestion_manager
            from observability.dashboard.services.data_service import DataService

            class FakeChroma:
                def get_by_metadata(self, filters=None, trace=None, collection=None):
                    return []

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
                        chunk_count=0,
                    )

            class FakeManager:
                def delete_document(self, source_path, collection):
                    return DeleteResult(source_path, collection, 0, True, 0, True)

            render_ingestion_manager(
                pipeline=FakePipeline(),
                document_manager=FakeManager(),
                data_service=DataService(FakeChroma(), FakeImages()),
                load_deps=False,
                default_collection="knowledge_hub",
            )

        app = AppTest.from_function(page_script, default_timeout=30)
        app.run()
        assert not app.exception
        assert "开始摄取" in [item.label for item in app.button]
        assert any("暂无已摄入文档" in str(item.value) for item in app.info)
        assert app.file_uploader
        assert "目标集合" in [item.label for item in app.selectbox]

    def test_ingest_button_runs_pipeline_and_shows_success(self, tmp_path) -> None:
        """填写本地 PDF 路径并点击开始摄取后，应展示完成提示。"""
        pdf = tmp_path / "sample.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        pdf_literal = str(pdf.resolve())

        def page_script() -> None:
            from ingestion.document_manager import DeleteResult
            from ingestion.pipeline import IngestionResult
            from observability.dashboard.pages.ingestion_manager import render_ingestion_manager
            from observability.dashboard.services.data_service import DataService

            class FakeChroma:
                def get_by_metadata(self, filters=None, trace=None, collection=None):
                    return []

            class FakeImages:
                def list_images(self, collection=None, doc_hash=None):
                    return []

                def get_path(self, image_id):
                    return None

            class FakePipeline:
                def run(self, source_path, collection="default", force=False, on_progress=None, trace=None):
                    if on_progress is not None:
                        on_progress("load", 1, 1)
                        on_progress("upsert", 1, 1)
                    return IngestionResult(
                        skipped=False,
                        source_path=source_path,
                        collection=collection,
                        chunk_count=4,
                    )

            class FakeManager:
                def delete_document(self, source_path, collection):
                    return DeleteResult(source_path, collection, 0, True, 0, True)

            render_ingestion_manager(
                pipeline=FakePipeline(),
                document_manager=FakeManager(),
                data_service=DataService(FakeChroma(), FakeImages()),
                load_deps=False,
                default_collection="knowledge_hub",
            )

        app = AppTest.from_function(page_script, default_timeout=30)
        app.run()
        assert not app.exception
        app.text_input[0].input(pdf_literal).run()
        ingest = next(item for item in app.button if item.label == "开始摄取")
        ingest.click().run()
        assert not app.exception
        successes = [str(item.value) for item in app.success]
        assert any("摄取完成" in text and "4 chunks" in text for text in successes)

    def test_delete_button_shows_confirmation(self) -> None:
        """文档列表中的删除按钮点击后应展示删除成功提示。"""

        def page_script() -> None:
            from ingestion.document_manager import DeleteResult
            from ingestion.pipeline import IngestionResult
            from observability.dashboard.pages.ingestion_manager import render_ingestion_manager
            from observability.dashboard.services.data_service import DataService

            class FakeChroma:
                def get_by_metadata(self, filters=None, trace=None, collection=None):
                    records = [
                        {
                            "id": "c1",
                            "text": "hello",
                            "metadata": {
                                "source_path": "/data/sample.pdf",
                                "collection": "knowledge_hub",
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
                    )

            class FakeManager:
                def delete_document(self, source_path, collection):
                    return DeleteResult(source_path, collection, 2, True, 0, True)

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
        assert not app.exception
        assert "删除" in [item.label for item in app.button]
        delete = next(item for item in app.button if item.label == "删除")
        delete.click().run()
        assert not app.exception
        successes = [str(item.value) for item in app.success]
        assert any("已删除" in text and "2 条 chunk" in text for text in successes)
