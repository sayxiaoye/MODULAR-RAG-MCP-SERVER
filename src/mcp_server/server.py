"""MCP Server 入口：Stdio Transport，stdout 仅输出 JSON-RPC，日志写入 stderr。"""

from __future__ import annotations

import json
import sys
from typing import Any, Mapping, TextIO

from core.settings import Settings, SettingsError, load_settings
from observability.logger import get_logger

# MCP 协议与 Server 元信息（E2 将扩展 capabilities / tools 协商）
SERVER_NAME = "modular-rag-mcp-server"
SERVER_VERSION = "0.1.0"
SUPPORTED_PROTOCOL_VERSION = "2024-11-05"

# JSON-RPC 2.0 标准错误码
PARSE_ERROR = -32700  # 解析错误
INVALID_REQUEST = -32600  # 无效请求
METHOD_NOT_FOUND = -32601  # 方法未找到
INTERNAL_ERROR = -32603  # 内部错误

logger = get_logger("mcp_server")


class MCPServer:
    """MCP Server 主类：读取 stdin 上的 NDJSON 消息并写回 stdout。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings
        self._initialized = False

    @property
    def settings(self) -> Settings:
        """懒加载配置，避免测试构造时必须访问磁盘。"""
        if self._settings is None:
            self._settings = load_settings()
        return self._settings

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
        """分发单条 JSON-RPC 请求；通知类消息返回 None。"""
        if not isinstance(request, Mapping):
            return self._error_response(None, INVALID_REQUEST, "Request must be a JSON object")

        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params") or {}

        if request.get("jsonrpc") != "2.0":
            return self._error_response(request_id, INVALID_REQUEST, "Invalid JSON-RPC version")

        if not isinstance(method, str) or not method:
            return self._error_response(request_id, INVALID_REQUEST, "Missing method")

        if self._is_notification(request):
            self._handle_notification(method, params)
            return None

        try:
            if method == "initialize":
                return self._success_response(request_id, self._handle_initialize(params))
            return self._error_response(request_id, METHOD_NOT_FOUND, f"Method not found: {method}")
        except Exception as exc:  # noqa: BLE001 — MCP 层需兜底，避免堆栈泄露到 Client
            logger.exception("处理 MCP 请求失败: method=%s", method)
            return self._error_response(request_id, INTERNAL_ERROR, str(exc))

    def _handle_initialize(self, params: Mapping[str, Any]) -> dict[str, Any]:
        """处理 initialize 握手，声明 Server 能力与版本信息。"""
        client_version = params.get("protocolVersion")
        if isinstance(client_version, str) and client_version:
            logger.info("MCP initialize — client protocolVersion=%s", client_version)

        # 触发配置加载，确保启动阶段 fail-fast
        _ = self.settings

        self._initialized = True
        logger.info("MCP initialize 完成 — server=%s", SERVER_NAME)

        return {
            "protocolVersion": SUPPORTED_PROTOCOL_VERSION,
            "capabilities": {
                "tools": {},
            },
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
            },
        }

    def _handle_notification(self, method: str, params: Mapping[str, Any]) -> None:
        """处理无 id 的通知消息（如 notifications/initialized）。"""
        if method == "notifications/initialized":
            logger.info("收到 notifications/initialized，握手完成")
            return
        logger.warning("收到未处理的通知: %s params=%s", method, params)

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
                return {
                    "_parse_error": MCPServer._error_response(None, PARSE_ERROR, "Parse error"),
                }
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

    MCPServer(settings=settings).run_stdio()


def main() -> None:
    """`python -m mcp_server.server` 时的入口函数。"""
    run_stdio_server()


if __name__ == "__main__":
    main()
