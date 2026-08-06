"""Smoke tests for package imports.

Verifies that all key packages can be imported after editable install.
"""

import pytest


@pytest.mark.unit
class TestSmokeImports:
    """Smoke tests to verify all key packages are importable."""

    def test_import_mcp_server(self) -> None:
        import mcp_server

        assert mcp_server is not None

    def test_import_mcp_server_tools(self) -> None:
        from mcp_server import tools

        assert tools is not None

    def test_import_core(self) -> None:
        import core

        assert core is not None

    def test_import_core_query_engine(self) -> None:
        from core import query_engine

        assert query_engine is not None

    def test_import_core_response(self) -> None:
        from core import response

        assert response is not None

    def test_import_core_trace(self) -> None:
        from core import trace

        assert trace is not None

    def test_import_ingestion(self) -> None:
        import ingestion

        assert ingestion is not None

    def test_import_ingestion_chunking(self) -> None:
        from ingestion import chunking

        assert chunking is not None

    def test_import_ingestion_embedding(self) -> None:
        from ingestion import embedding

        assert embedding is not None

    def test_import_ingestion_storage(self) -> None:
        from ingestion import storage

        assert storage is not None

    def test_import_ingestion_transform(self) -> None:
        from ingestion import transform

        assert transform is not None

    def test_import_libs(self) -> None:
        import libs

        assert libs is not None

    def test_import_libs_embedding(self) -> None:
        from libs import embedding

        assert embedding is not None

    def test_import_libs_evaluator(self) -> None:
        from libs import evaluator

        assert evaluator is not None

    def test_import_libs_llm(self) -> None:
        from libs import llm

        assert llm is not None

    def test_import_libs_loader(self) -> None:
        from libs import loader

        assert loader is not None

    def test_import_libs_reranker(self) -> None:
        from libs import reranker

        assert reranker is not None

    def test_import_libs_splitter(self) -> None:
        from libs import splitter

        assert splitter is not None

    def test_import_libs_vector_store(self) -> None:
        from libs import vector_store

        assert vector_store is not None

    def test_import_observability(self) -> None:
        import observability

        assert observability is not None

    def test_import_observability_dashboard(self) -> None:
        from observability import dashboard

        assert dashboard is not None

    def test_import_observability_evaluation(self) -> None:
        from observability import evaluation

        assert evaluation is not None
