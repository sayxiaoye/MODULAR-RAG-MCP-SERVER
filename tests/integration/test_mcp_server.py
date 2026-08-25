"""MCP Server 集成测试：子进程验证 Stdio 握手与 stdout/stderr 隔离。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from mcp_server.server import MCPServer, SERVER_NAME, SERVER_VERSION, SUPPORTED_PROTOCOL_VERSION

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVER_MODULE = "mcp_server.server"


def _run_server_exchange(
    messages: list[dict],
    *,
    timeout: float = 10.0,
) -> subprocess.CompletedProcess[str]:
    """以子进程启动 Server，按 NDJSON 发送请求并收集 stdout/stderr。"""
    payload = "\n".join(json.dumps(msg, ensure_ascii=False) for msg in messages) + "\n"
    return subprocess.run(
        [sys.executable, "-m", SERVER_MODULE],
        input=payload,
        text=True,
        capture_output=True,
        cwd=str(REPO_ROOT),
        timeout=timeout,
        env={**dict(**__import__("os").environ), "PYTHONPATH": str(REPO_ROOT / "src")},
    )


@pytest.mark.integration
class TestMCPServerStdio:
    """验证 E1：Stdio 约束与 initialize 握手。"""

    def test_initialize_returns_server_info(self) -> None:
        """发送 initialize 应返回 serverInfo 与 capabilities。"""
        result = _run_server_exchange(
            [
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": SUPPORTED_PROTOCOL_VERSION,
                        "capabilities": {},
                        "clientInfo": {"name": "pytest", "version": "1.0"},
                    },
                }
            ]
        )

        assert result.returncode == 0, result.stderr
        lines = [line for line in result.stdout.splitlines() if line.strip()]
        assert len(lines) == 1

        response = json.loads(lines[0])
        assert response["jsonrpc"] == "2.0"
        assert response["id"] == 1
        assert response["result"]["protocolVersion"] == SUPPORTED_PROTOCOL_VERSION
        assert response["result"]["serverInfo"]["name"] == SERVER_NAME
        assert response["result"]["serverInfo"]["version"] == SERVER_VERSION
        assert "tools" in response["result"]["capabilities"]

    def test_stderr_has_logs_but_stdout_is_clean_json(self) -> None:
        """日志应写入 stderr，stdout 仅包含合法 JSON-RPC 响应。"""
        result = _run_server_exchange(
            [
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": SUPPORTED_PROTOCOL_VERSION,
                        "capabilities": {},
                        "clientInfo": {"name": "pytest", "version": "1.0"},
                    },
                }
            ]
        )

        assert "[INFO]" in result.stderr or "MCP Server" in result.stderr
        assert "[INFO]" not in result.stdout
        assert "MCP Server 启动" not in result.stdout

        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            parsed = json.loads(line)
            assert parsed.get("jsonrpc") == "2.0"

    def test_initialized_notification_does_not_write_response(self) -> None:
        """notifications/initialized 为通知，不应产生 stdout 响应。"""
        result = _run_server_exchange(
            [
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": SUPPORTED_PROTOCOL_VERSION,
                        "capabilities": {},
                        "clientInfo": {"name": "pytest", "version": "1.0"},
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {},
                },
            ]
        )

        lines = [line for line in result.stdout.splitlines() if line.strip()]
        assert len(lines) == 1
        assert json.loads(lines[0])["id"] == 3

    def test_unknown_method_returns_json_rpc_error(self) -> None:
        """未知 method 应返回 -32601，且错误信息出现在 JSON 响应中。"""
        server = MCPServer()
        response = server.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 99,
                "method": "tools/list",
                "params": {},
            }
        )

        assert response is not None
        assert response["error"]["code"] == -32601
