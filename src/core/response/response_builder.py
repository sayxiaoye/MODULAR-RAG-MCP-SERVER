"""MCP Tool 响应构建：Markdown 正文 + structuredContent.citations。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from core.response.citation_generator import Citation, CitationGenerator
from core.types import RetrievalResult

_EMPTY_HINT = "未找到相关文档，请先运行 ingest.py 摄取数据。"


@dataclass(frozen=True)
class MCPResponse:
    """MCP tools/call 结果载体，可序列化为 Tool handler 返回值。"""

    content: list[dict[str, Any]]
    structured_content: dict[str, Any]
    is_error: bool = False

    def to_tool_result(self) -> dict[str, Any]:
        """转为 ProtocolHandler tools/call 期望的字典结构。"""
        return {
            "content": list(self.content),
            "structuredContent": dict(self.structured_content),
            "isError": self.is_error,
        }


class ResponseBuilder:
    """将检索结果组装为带引用标注的 MCP 响应。"""

    @classmethod
    def build(cls, retrieval_results: Sequence[RetrievalResult], query: str) -> MCPResponse:
        """
        构建 MCP 响应。

        Args:
            retrieval_results: 最终 Top-K 检索结果（已融合/重排）。
            query: 用户原始查询，用于 structuredContent.answer 上下文。
        """
        if not retrieval_results:
            return cls._build_empty_response(query)

        citations = CitationGenerator.generate(retrieval_results)
        markdown = cls._build_markdown(query, citations)
        return MCPResponse(
            content=[{"type": "text", "text": markdown}],
            structured_content={
                "query": query,
                "answer": markdown,
                "citations": [item.to_dict() for item in citations],
            },
            is_error=False,
        )

    @classmethod
    def _build_empty_response(cls, query: str) -> MCPResponse:
        """无命中时返回友好提示，避免空 content 数组。"""
        return MCPResponse(
            content=[{"type": "text", "text": _EMPTY_HINT}],
            structured_content={
                "query": query,
                "answer": _EMPTY_HINT,
                "citations": [],
            },
            is_error=False,
        )

    @staticmethod
    def _build_markdown(query: str, citations: Sequence[Citation]) -> str:
        """生成含 [1]、[2] 引用标注的 Markdown 文本。"""
        lines = [f"## 检索结果：{query.strip()}", ""]
        for citation in citations:
            page_suffix = f"（第 {citation.page} 页）" if citation.page is not None else ""
            lines.append(
                f"**[{citation.id}]** `{citation.source}`{page_suffix} "
                f"(score={citation.score:.4f})"
            )
            snippet = " ".join(citation.text.split())
            if len(snippet) > 280:
                snippet = f"{snippet[:277]}..."
            lines.append(f"> {snippet}")
            lines.append("")
        return "\n".join(lines).strip()
