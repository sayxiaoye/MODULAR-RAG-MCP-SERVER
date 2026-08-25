"""MCP JSON-RPC 协议处理：initialize / tools/list / tools/call 与错误码映射。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping, Sequence

from core.settings import Settings, load_settings
from observability.logger import get_logger

logger = get_logger("mcp_server.protocol")

SERVER_NAME = "modular-rag-mcp-server"
SERVER_VERSION = "0.1.0"
# MCP 规范发布日版本号；initialize 时与客户端协商，须与官方协议版本字符串一致
SUPPORTED_PROTOCOL_VERSION = "2024-11-05"

# JSON-RPC 2.0 标准错误码（MCP 沿用的保留区间 -32700 ~ -32600）
INVALID_REQUEST = -32600  # 无效请求
METHOD_NOT_FOUND = -32601  # 方法未找到
INVALID_PARAMS = -32602  # 无效参数
INTERNAL_ERROR = -32603  # 内部错误

ToolHandler = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class ToolDefinition:
    """已注册 MCP Tool 的元数据与执行入口。"""

    name: str
    description: str
    input_schema: Mapping[str, Any]
    handler: ToolHandler


class ProtocolHandlerError(Exception):
    """协议层可预期错误，携带 JSON-RPC 错误码。"""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class ProtocolHandler:
    """封装 MCP 核心方法分发与能力协商。"""

    def __init__(
        self,
        settings: Settings | None = None,
        tools: Sequence[ToolDefinition] | None = None,
    ) -> None:
        self._settings = settings
        self._initialized = False
        self._tools: MutableMapping[str, ToolDefinition] = {
            tool.name: tool for tool in (tools or ())
        }

    @property
    def settings(self) -> Settings:
        """懒加载配置，initialize 阶段触发 fail-fast 校验。"""
        if self._settings is None:
            self._settings = load_settings()
        return self._settings

    def dispatch(self, method: str, params: Mapping[str, Any] | None) -> Any:
        """按 MCP method 路由到具体 handler；未知 method 抛出 ProtocolHandlerError。"""
        payload = params or {}
        if not isinstance(payload, Mapping):
            raise ProtocolHandlerError(INVALID_PARAMS, "params must be an object")

        if method == "initialize":
            return self.handle_initialize(payload)
        if method == "tools/list":
            return self.handle_tools_list()
        if method == "tools/call":
            return self.handle_tools_call(payload)

        raise ProtocolHandlerError(METHOD_NOT_FOUND, f"Method not found: {method}")

    def handle_notification(self, method: str, params: Mapping[str, Any] | None) -> None:
        """处理无响应的通知消息。"""
        if method == "notifications/initialized":
            logger.info("收到 notifications/initialized，握手完成")
            return
        logger.warning("收到未处理的通知: %s params=%s", method, params)

    def handle_initialize(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """能力协商：返回 protocolVersion、capabilities 与 serverInfo。"""
        client_version = params.get("protocolVersion")
        if isinstance(client_version, str) and client_version:
            logger.info("MCP initialize — client protocolVersion=%s", client_version)

        _ = self.settings
        self._initialized = True
        logger.info("MCP initialize 完成 — server=%s tools=%d", SERVER_NAME, len(self._tools))

        return {
            "protocolVersion": SUPPORTED_PROTOCOL_VERSION,
            "capabilities": {
                "tools": {
                    "listChanged": False,
                },
            },
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
            },
        }

    def handle_tools_list(self) -> dict[str, Any]:
        """返回已注册 tools 的 name/description/inputSchema。"""
        return {
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": dict(tool.input_schema),
                }
                for tool in self._tools.values()
            ],
        }

    def handle_tools_call(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """路由 tools/call 到具体 tool handler，并规范异常为 JSON-RPC 错误。"""
        name = params.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ProtocolHandlerError(INVALID_PARAMS, "Missing or invalid tool name")

        arguments = params.get("arguments", {})
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, Mapping):
            raise ProtocolHandlerError(INVALID_PARAMS, "arguments must be an object")

        tool = self._tools.get(name)
        if tool is None:
            raise ProtocolHandlerError(METHOD_NOT_FOUND, f"Unknown tool: {name}")

        try:
            result = tool.handler(dict(arguments))
        except ProtocolHandlerError:
            raise
        except Exception as exc:  # noqa: BLE001 — 统一转换为 -32603，避免堆栈泄露
            logger.exception("Tool 执行失败: %s", name)
            raise ProtocolHandlerError(INTERNAL_ERROR, str(exc)) from exc

        if not isinstance(result, Mapping):
            raise ProtocolHandlerError(INTERNAL_ERROR, "Tool handler must return an object")

        return dict(result)

    def register_tool(self, tool: ToolDefinition) -> None:
        """运行时注册 tool，供后续 E3-E5 扩展。"""
        self._tools[tool.name] = tool
