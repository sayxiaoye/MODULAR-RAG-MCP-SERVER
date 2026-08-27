"""MCP Tools 子包：对外暴露的知识库工具实现。"""

from mcp_server.tools.get_document_summary import (
    TOOL_NAME as GET_DOCUMENT_SUMMARY_TOOL_NAME,
    DocumentSummary,
    build_get_document_summary_tool,
    get_document_summary,
    resolve_document_summary,
)
from mcp_server.tools.list_collections import (
    TOOL_NAME as LIST_COLLECTIONS_TOOL_NAME,
    build_list_collections_tool,
    discover_collections,
    list_collections,
)
from mcp_server.tools.query_knowledge_hub import (
    TOOL_NAME,
    build_query_knowledge_hub_tool,
    query_knowledge_hub,
)
from mcp_server.tools.registry import build_default_protocol_handler, build_default_tools

__all__ = [
    "GET_DOCUMENT_SUMMARY_TOOL_NAME",
    "LIST_COLLECTIONS_TOOL_NAME",
    "DocumentSummary",
    "TOOL_NAME",
    "build_default_protocol_handler",
    "build_default_tools",
    "build_get_document_summary_tool",
    "build_list_collections_tool",
    "build_query_knowledge_hub_tool",
    "discover_collections",
    "get_document_summary",
    "list_collections",
    "query_knowledge_hub",
    "resolve_document_summary",
]
