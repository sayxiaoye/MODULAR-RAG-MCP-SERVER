"""MCP Server 入口：Stdio Transport，stdout 仅输出 JSON-RPC，日志写入 stderr。"""

from __future__ import annotations

import json
import sys
from typing import Any, Mapping, TextIO

from core.settings import Settings, SettingsError, load_settings
from mcp_server.protocol_handler import (
    ProtocolHandler,
    ProtocolHandlerError,
    INVALID_REQUEST,
    SERVER_NAME,
    SERVER_VERSION,
)
from mcp_server.tools.registry import build_default_protocol_handler
from observability.logger import get_logger

# JSON-RPC 解析错误在 transport 层处理
PARSE_ERROR = -32700

logger = get_logger("mcp_server")


class MCPServer:
    """MCP Server 主类：Stdio 读写 + 委托 ProtocolHandler 处理业务方法。"""

    def __init__(
        self,
        settings: Settings | None = None,
        protocol_handler: ProtocolHandler | None = None,
    ) -> None:
        self._settings = settings
        self._handler = protocol_handler or build_default_protocol_handler(settings=settings)

    def run_stdio(self, input_stream: TextIO | None = None, output_stream: TextIO | None = None) -> None:
        """启动 Stdio 主循环，直到输入流关闭。"""
        stdin = input_stream or sys.stdin
        stdout = output_stream or sys.stdout

        logger.info("MCP Server 启动 — name=%s version=%s", SERVER_NAME, SERVER_VERSION)

        while True:
            request = self._read_message(stdin)
            if request is None:
                logger.info("MCP Server 输入流结束，退出")
                break

            if "_parse_error" in request:
                self._write_message(stdout, request["_parse_error"])
                continue

            response = self.handle_request(request)
            if response is not None:
                self._write_message(stdout, response)

    def handle_request(self, request: Mapping[str, Any]) -> dict[str, Any] | None:
        """校验 JSON-RPC 信封后委托 ProtocolHandler。"""
        if not isinstance(request, Mapping):
            return self._error_response(None, INVALID_REQUEST, "Request must be a JSON object")

        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params")

        if request.get("jsonrpc") != "2.0":
            return self._error_response(request_id, INVALID_REQUEST, "Invalid JSON-RPC version")

        if not isinstance(method, str) or not method:
            return self._error_response(request_id, INVALID_REQUEST, "Missing method")

        if self._is_notification(request):
            self._handler.handle_notification(method, params if isinstance(params, Mapping) else {})
            return None

        try:
            result = self._handler.dispatch(method, params if isinstance(params, Mapping) else {})
            return self._success_response(request_id, result)
        except ProtocolHandlerError as exc:
            return self._error_response(request_id, exc.code, exc.message)
        except Exception as exc:  # noqa: BLE001 — transport 层兜底
            logger.exception("处理 MCP 请求失败: method=%s", method)
            from mcp_server.protocol_handler import INTERNAL_ERROR

            return self._error_response(request_id, INTERNAL_ERROR, str(exc))

    @staticmethod
    def _read_message(stream: TextIO) -> dict[str, Any] | None:
        """从 stdin 读取一行 NDJSON；空行跳过，EOF 返回 None。"""
        while True:
            line = stream.readline()
            if line == "":
                return None
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                logger.error("JSON 解析失败: %s", exc)
                return {"_parse_error": MCPServer._error_response(None, PARSE_ERROR, "Parse error")}
            if not isinstance(payload, dict):
                return {
                    "_parse_error": MCPServer._error_response(
                        None, INVALID_REQUEST, "Invalid request payload"
                    ),
                }
            return payload

    @staticmethod
    def _write_message(stream: TextIO, message: Mapping[str, Any]) -> None:
        """向 stdout 写入单条 JSON-RPC 消息（NDJSON）。"""
        stream.write(json.dumps(message, ensure_ascii=False) + "\n")
        stream.flush()

    @staticmethod
    def _is_notification(request: Mapping[str, Any]) -> bool:
        """JSON-RPC 通知没有 id 字段。"""
        return "id" not in request

    @staticmethod
    def _success_response(request_id: Any, result: Mapping[str, Any]) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    @staticmethod
    def _error_response(request_id: Any, code: int, message: str) -> dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }


def run_stdio_server() -> None:
    """CLI 入口：加载配置后启动 Stdio Server。"""
    try:
        settings = load_settings()
    except SettingsError as exc:
        logger.error("配置加载失败，无法启动 MCP Server: %s", exc)
        sys.exit(1)

    MCPServer(settings=settings, protocol_handler=build_default_protocol_handler(settings)).run_stdio()


def main() -> None:
    """`python -m mcp_server.server` 时的入口函数。"""
    run_stdio_server()


if __name__ == "__main__":
    main()
