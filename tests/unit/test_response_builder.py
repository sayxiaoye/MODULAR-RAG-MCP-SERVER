"""ResponseBuilder / CitationGenerator 单元测试。"""

from __future__ import annotations

import pytest

from core.response.citation_generator import CitationGenerator
from core.response.response_builder import ResponseBuilder
from core.types import RetrievalResult


def _result(
    chunk_id: str,
    *,
    score: float = 0.91,
    text: str = "RAG pipeline processes documents into chunks.",
    source_path: str = "docs/guide.pdf",
    page: int | None = 3,
) -> RetrievalResult:
    metadata: dict = {"source_path": source_path}
    if page is not None:
        metadata["page"] = page
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text,
        metadata=metadata,
    )


@pytest.mark.unit
class TestCitationGenerator:
    """验证引用字段完整性。"""

    def test_generate_contains_required_fields(self) -> None:
        citations = CitationGenerator.generate([_result("chunk-1")])

        assert len(citations) == 1
        payload = citations[0].to_dict()
        assert payload["id"] == 1
        assert payload["source"] == "guide.pdf"
        assert payload["page"] == 3
        assert payload["chunk_id"] == "chunk-1"
        assert payload["score"] == pytest.approx(0.91)
        assert "RAG pipeline" in payload["text"]


@pytest.mark.unit
class TestResponseBuilder:
    """验证 MCP 响应 Markdown 与 structuredContent。"""

    def test_build_markdown_contains_citation_markers(self) -> None:
        response = ResponseBuilder.build(
            [
                _result("chunk-1"),
                _result("chunk-2", source_path="docs/other.pdf", page=5, score=0.75),
            ],
            "RAG pipeline",
        )
        tool_result = response.to_tool_result()

        markdown = tool_result["content"][0]["text"]
        assert "[1]" in markdown
        assert "[2]" in markdown
        assert "guide.pdf" in markdown
        assert tool_result["isError"] is False

        citations = tool_result["structuredContent"]["citations"]
        assert len(citations) == 2
        assert {item["chunk_id"] for item in citations} == {"chunk-1", "chunk-2"}
        for key in ("source", "page", "chunk_id", "score"):
            assert key in citations[0]

    def test_build_empty_returns_friendly_hint(self) -> None:
        response = ResponseBuilder.build([], "empty query")
        tool_result = response.to_tool_result()

        assert tool_result["content"][0]["text"]
        assert "未找到相关文档" in tool_result["content"][0]["text"]
        assert tool_result["structuredContent"]["citations"] == []
