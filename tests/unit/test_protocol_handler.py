"""ProtocolHandler 单元测试：验证 initialize / tools/list / tools/call 与错误码。"""

from __future__ import annotations

import pytest

from mcp_server.protocol_handler import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    SERVER_NAME,
    SERVER_VERSION,
    SUPPORTED_PROTOCOL_VERSION,
    ProtocolHandler,
    ProtocolHandlerError,
    ToolDefinition,
)


def _echo_handler(arguments: dict) -> dict:
    """测试用 echo tool：回显 message 字段。"""
    return {
        "content": [{"type": "text", "text": str(arguments.get("message", ""))}],
        "isError": False,
    }


def _failing_handler(_arguments: dict) -> dict:
    """故意抛异常，验证 -32603 转换。"""
    raise RuntimeError("boom")


@pytest.fixture
def handler() -> ProtocolHandler:
    """带 echo tool 的 ProtocolHandler，覆盖 list/call 路由。"""
    return ProtocolHandler(
        tools=[
            ToolDefinition(
                name="echo",
                description="回显输入消息",
                input_schema={
                    "type": "object",
                    "properties": {"message": {"type": "string"}},
                },
                handler=_echo_handler,
            ),
            ToolDefinition(
                name="fail",
                description="故意失败",
                input_schema={"type": "object"},
                handler=_failing_handler,
            ),
        ],
    )


@pytest.mark.unit
class TestProtocolHandlerInitialize:
    """验证 initialize 能力协商。"""

    def test_handle_initialize_returns_server_info(self, handler: ProtocolHandler) -> None:
        result = handler.handle_initialize(
            {
                "protocolVersion": SUPPORTED_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "pytest", "version": "1.0"},
            }
        )

        assert result["protocolVersion"] == SUPPORTED_PROTOCOL_VERSION
        assert result["serverInfo"]["name"] == SERVER_NAME
        assert result["serverInfo"]["version"] == SERVER_VERSION
        assert "tools" in result["capabilities"]


@pytest.mark.unit
class TestProtocolHandlerToolsList:
    """验证 tools/list 返回 schema。"""

    def test_handle_tools_list_returns_registered_tools(self, handler: ProtocolHandler) -> None:
        result = handler.handle_tools_list()

        names = {tool["name"] for tool in result["tools"]}
        assert names == {"echo", "fail"}
        echo = next(item for item in result["tools"] if item["name"] == "echo")
        assert echo["description"] == "回显输入消息"
        assert echo["inputSchema"]["type"] == "object"


@pytest.mark.unit
class TestProtocolHandlerToolsCall:
    """验证 tools/call 路由与参数校验。"""

    def test_handle_tools_call_routes_to_handler(self, handler: ProtocolHandler) -> None:
        result = handler.handle_tools_call(
            {"name": "echo", "arguments": {"message": "hello"}}
        )

        assert result["isError"] is False
        assert result["content"][0]["text"] == "hello"

    def test_handle_tools_call_unknown_tool(self, handler: ProtocolHandler) -> None:
        with pytest.raises(ProtocolHandlerError) as exc_info:
            handler.handle_tools_call({"name": "missing", "arguments": {}})

        assert exc_info.value.code == METHOD_NOT_FOUND

    def test_handle_tools_call_missing_name(self, handler: ProtocolHandler) -> None:
        with pytest.raises(ProtocolHandlerError) as exc_info:
            handler.handle_tools_call({"arguments": {}})

        assert exc_info.value.code == INVALID_PARAMS

    def test_handle_tools_call_invalid_arguments_type(self, handler: ProtocolHandler) -> None:
        with pytest.raises(ProtocolHandlerError) as exc_info:
            handler.handle_tools_call({"name": "echo", "arguments": "bad"})

        assert exc_info.value.code == INVALID_PARAMS

    def test_handle_tools_call_internal_error_without_stack_leak(
        self, handler: ProtocolHandler
    ) -> None:
        with pytest.raises(ProtocolHandlerError) as exc_info:
            handler.handle_tools_call({"name": "fail", "arguments": {}})

        assert exc_info.value.code == INTERNAL_ERROR
        assert exc_info.value.message == "boom"
        assert "Traceback" not in exc_info.value.message


@pytest.mark.unit
class TestProtocolHandlerDispatch:
    """验证 dispatch 对未知 method 的错误处理。"""

    def test_dispatch_unknown_method(self, handler: ProtocolHandler) -> None:
        with pytest.raises(ProtocolHandlerError) as exc_info:
            handler.dispatch("unknown/method", {})

        assert exc_info.value.code == METHOD_NOT_FOUND

    def test_dispatch_tools_list(self, handler: ProtocolHandler) -> None:
        result = handler.dispatch("tools/list", {})
        assert "tools" in result

    def test_handle_notification_initialized(self, handler: ProtocolHandler) -> None:
        # 通知类消息不应抛异常
        handler.handle_notification("notifications/initialized", {})
