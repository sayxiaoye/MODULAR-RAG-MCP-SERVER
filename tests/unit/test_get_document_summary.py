"""get_document_summary Tool 单元测试。"""

from __future__ import annotations

import pytest

from core.settings import load_settings
from mcp_server.protocol_handler import INVALID_PARAMS, ProtocolHandlerError
from mcp_server.tools.get_document_summary import (
    DocumentSummary,
    build_get_document_summary_tool,
    get_document_summary,
)


@pytest.mark.unit
class TestGetDocumentSummary:
    """验证文档摘要查询与错误处理。"""

    def test_returns_structured_summary(self) -> None:
        """存在 doc_id 时应返回 title/summary/tags。"""

        def _lookup(doc_id: str, settings, collection):  # noqa: ANN001
            assert doc_id == "abc123"
            return DocumentSummary(
                doc_id=doc_id,
                title="Sample Document",
                summary="A short summary.",
                tags=["rag", "test"],
                source_path="tests/fixtures/sample_documents/simple.pdf",
                chunk_count=3,
            )

        result = get_document_summary(
            {"doc_id": "abc123"},
            settings=load_settings(),
            lookup_fn=_lookup,
        )

        markdown = result["content"][0]["text"]
        assert "Sample Document" in markdown
        assert result["isError"] is False

        payload = result["structuredContent"]
        assert payload["title"] == "Sample Document"
        assert payload["summary"] == "A short summary."
        assert payload["tags"] == ["rag", "test"]
        assert payload["chunk_count"] == 3

    def test_missing_doc_id_raises_invalid_params(self) -> None:
        with pytest.raises(ProtocolHandlerError) as exc_info:
            get_document_summary({"doc_id": "  "})
        assert exc_info.value.code == INVALID_PARAMS

    def test_unknown_doc_id_raises_invalid_params(self) -> None:
        def _lookup(doc_id, settings, collection):  # noqa: ANN001
            return None

        with pytest.raises(ProtocolHandlerError) as exc_info:
            get_document_summary(
                {"doc_id": "missing.pdf"},
                lookup_fn=_lookup,
            )
        assert exc_info.value.code == INVALID_PARAMS
        assert "未找到文档" in exc_info.value.message

    def test_tool_definition_registers_handler(self) -> None:
        tool = build_get_document_summary_tool(
            lookup_fn=lambda doc_id, settings, collection: DocumentSummary(
                doc_id=doc_id,
                title="T",
                summary="S",
                tags=[],
            )
        )
        assert tool.name == "get_document_summary"
        result = tool.handler({"doc_id": "doc-1"})
        assert result["structuredContent"]["title"] == "T"
