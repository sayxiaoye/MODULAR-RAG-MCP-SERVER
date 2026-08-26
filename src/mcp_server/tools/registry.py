"""MCP Tools 注册表：集中注册对外暴露的 ToolDefinition。"""

from __future__ import annotations

from typing import Sequence

from core.settings import Settings, load_settings
from mcp_server.protocol_handler import ProtocolHandler, ToolDefinition
from mcp_server.tools.list_collections import build_list_collections_tool
from mcp_server.tools.query_knowledge_hub import build_query_knowledge_hub_tool


def build_default_tools(settings: Settings | None = None) -> list[ToolDefinition]:
    """构建默认 Tool 列表（query_knowledge_hub + list_collections）。"""
    resolved = settings or load_settings()
    return [
        build_query_knowledge_hub_tool(settings=resolved),
        build_list_collections_tool(settings=resolved),
    ]


def build_default_protocol_handler(settings: Settings | None = None) -> ProtocolHandler:
    """创建已注册默认 Tools 的 ProtocolHandler，供 MCPServer 使用。"""
    resolved = settings or load_settings()
    tools: Sequence[ToolDefinition] = build_default_tools(resolved)
    return ProtocolHandler(settings=resolved, tools=tools)
