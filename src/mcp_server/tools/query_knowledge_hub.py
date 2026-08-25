"""query_knowledge_hub Tool：混合检索 + 引用透明 MCP 响应。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Mapping

from core.response.response_builder import ResponseBuilder
from core.settings import Settings, SettingsError, load_settings
from core.trace.trace_context import TraceContext
from mcp_server.protocol_handler import INVALID_PARAMS, ProtocolHandlerError, ToolDefinition
from observability.logger import get_logger

# 复用 scripts/query.py 中的查询流水线，避免重复编排逻辑
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_ROOT = _REPO_ROOT / "scripts"
if str(_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_ROOT))

from query import DEFAULT_TOP_K, execute_query_pipeline, _settings_for_query  # noqa: E402

logger = get_logger("mcp_server.tools.query_knowledge_hub")

TOOL_NAME = "query_knowledge_hub"
TOOL_DESCRIPTION = "在知识库中执行混合检索（Dense + Sparse + Rerank），返回带引用的 Markdown 结果。"

INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "用户查询文本",
        },
        "top_k": {
            "type": "integer",
            "minimum": 1,
            "description": "返回结果数量，默认 10",
        },
        "collection": {
            "type": "string",
            "description": "限定检索集合名称，需与 ingest 时一致",
        },
    },
    "required": ["query"],
}

PipelineRunner = Callable[..., Any]


def query_knowledge_hub(
    arguments: Mapping[str, Any],
    *,
    settings: Settings | None = None,
    pipeline_runner: PipelineRunner | None = None,
) -> dict[str, Any]:
    """
    MCP Tool 入口：执行检索并构建带 citations 的响应。

    Args:
        arguments: tools/call 传入的参数字典。
        settings: 可选配置注入，测试时可覆盖。
        pipeline_runner: 可选流水线函数注入，默认 execute_query_pipeline。
    """
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ProtocolHandlerError(INVALID_PARAMS, "query 必须是非空字符串")

    top_k_raw = arguments.get("top_k", DEFAULT_TOP_K)
    if not isinstance(top_k_raw, int) or top_k_raw <= 0:
        raise ProtocolHandlerError(INVALID_PARAMS, "top_k 必须是大于 0 的整数")

    collection = arguments.get("collection")
    if collection is not None and (not isinstance(collection, str) or not collection.strip()):
        raise ProtocolHandlerError(INVALID_PARAMS, "collection 必须是非空字符串")

    try:
        base_settings = settings or load_settings()
    except SettingsError as exc:
        raise ProtocolHandlerError(INVALID_PARAMS, f"配置加载失败: {exc}") from exc

    resolved_settings, bm25_root = _settings_for_query(
        base_settings,
        collection.strip() if isinstance(collection, str) else None,
        data_root=None,
    )
    trace = (
        TraceContext(trace_type="query")
        if resolved_settings.observability.trace_enabled
        else None
    )
    filters = {"collection": collection.strip()} if isinstance(collection, str) and collection.strip() else None

    runner = pipeline_runner or execute_query_pipeline
    pipeline_result = runner(
        resolved_settings,
        query.strip(),
        top_k_raw,
        filters=filters,
        no_rerank=not resolved_settings.rerank.enabled,
        trace=trace,
        bm25_root=bm25_root,
    )

    logger.info(
        "query_knowledge_hub 完成 — query=%r hits=%d collection=%s",
        query.strip(),
        len(pipeline_result.final_results),
        resolved_settings.vector_store.collection_name,
    )
    return ResponseBuilder.build(pipeline_result.final_results, query.strip()).to_tool_result()


def build_query_knowledge_hub_tool(
    *,
    settings: Settings | None = None,
    pipeline_runner: PipelineRunner | None = None,
) -> ToolDefinition:
    """构造可注册到 ProtocolHandler 的 ToolDefinition。"""

    def _handler(arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        return query_knowledge_hub(
            arguments,
            settings=settings,
            pipeline_runner=pipeline_runner,
        )

    return ToolDefinition(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=INPUT_SCHEMA,
        handler=_handler,
    )
