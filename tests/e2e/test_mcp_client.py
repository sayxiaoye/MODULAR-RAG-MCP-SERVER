"""E2E：以子进程启动 MCP Server，模拟 Client 的 tools/list 与 tools/call。

对应 spec I1：Stdio NDJSON 握手后调用 ``query_knowledge_hub``，断言返回 citations。
子进程注入固定检索结果，避免 E2E 依赖 llama-server / 真实向量库。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from mcp_server.protocol_handler import SUPPORTED_PROTOCOL_VERSION
from mcp_server.tools.get_document_summary import TOOL_NAME as GET_DOCUMENT_SUMMARY
from mcp_server.tools.list_collections import TOOL_NAME as LIST_COLLECTIONS
from mcp_server.tools.query_knowledge_hub import TOOL_NAME as QUERY_KNOWLEDGE_HUB

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
THIS_FILE = Path(__file__).resolve()

_FAKE_CHUNK_ID = "chunk-e2e-mcp-001"
_FAKE_SOURCE = "config_guide.pdf"
_FAKE_QUERY = "如何配置 Azure OpenAI？"


def _run_stdio_harness() -> None:
    """子进程入口：真实 MCPServer Stdio + 注入检索结果。"""
    src = str(Path(__file__).resolve().parents[2] / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    from core.query_engine.query_pipeline import QueryPipelineResult
    from core.types import RetrievalResult
    from mcp_server.protocol_handler import ProtocolHandler
    from mcp_server.server import MCPServer
    from mcp_server.tools.get_document_summary import build_get_document_summary_tool
    from mcp_server.tools.list_collections import build_list_collections_tool
    from mcp_server.tools.query_knowledge_hub import build_query_knowledge_hub_tool

    fake_hit = RetrievalResult(
        chunk_id=_FAKE_CHUNK_ID,
        score=0.91,
        text="在 Azure 门户创建 OpenAI 资源，填写 endpoint 与 api_key。",
        metadata={"source_path": f"docs/{_FAKE_SOURCE}", "page": 3},
    )

    def _fake_pipeline(*_args: Any, **_kwargs: Any) -> QueryPipelineResult:
        # 固定命中，保证 Client 侧一定能拿到 citations
        return QueryPipelineResult(
            fusion_results=[fake_hit],
            final_results=[fake_hit],
            dense_results=[fake_hit],
            sparse_results=[],
        )

    tools = [
        build_query_knowledge_hub_tool(pipeline_runner=_fake_pipeline),
        build_list_collections_tool(),
        build_get_document_summary_tool(),
    ]
    MCPServer(protocol_handler=ProtocolHandler(tools=tools)).run_stdio()


def _client_exchange(messages: list[dict[str, Any]], *, timeout: float = 20.0) -> list[dict[str, Any]]:
    """模拟 MCP Client：向子进程 Server 写入 NDJSON，解析 stdout 响应。"""
    payload = "\n".join(json.dumps(msg, ensure_ascii=False) for msg in messages) + "\n"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["MCP_E2E_ROLE"] = "server"
    completed = subprocess.run(
        [sys.executable, str(THIS_FILE)],
        input=payload,
        text=True,
        capture_output=True,
        cwd=str(REPO_ROOT),
        timeout=timeout,
        env=env,
    )
    assert completed.returncode == 0, (
        f"MCP Server 子进程退出码 {completed.returncode}\nstderr={completed.stderr}"
    )
    responses: list[dict[str, Any]] = []
    for line in completed.stdout.splitlines():
        if not line.strip():
            continue
        parsed = json.loads(line)
        assert parsed.get("jsonrpc") == "2.0"
        responses.append(parsed)
    return responses


def _by_id(responses: list[dict[str, Any]], request_id: int) -> dict[str, Any]:
    """按 JSON-RPC id 取响应；缺失则失败。"""
    for item in responses:
        if item.get("id") == request_id:
            return item
    raise AssertionError(f"没有 id={request_id} 的响应: {responses!r}")


def _initialize_message(request_id: int = 1) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": "initialize",
        "params": {
            "protocolVersion": SUPPORTED_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": "e2e-mcp-client", "version": "1.0"},
        },
    }


@pytest.mark.e2e
class TestMCPClientSimulation:
    """模拟 Copilot/Claude 侧 MCP Client：list tools 后调用 query_knowledge_hub。"""

    def test_tools_list_exposes_query_knowledge_hub(self) -> None:
        """tools/list 应包含 query_knowledge_hub 及其 query 入参。"""
        responses = _client_exchange(
            [
                _initialize_message(1),
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            ]
        )
        listed = _by_id(responses, 2)["result"]["tools"]
        names = {item["name"] for item in listed}
        assert QUERY_KNOWLEDGE_HUB in names
        assert LIST_COLLECTIONS in names
        assert GET_DOCUMENT_SUMMARY in names
        query_tool = next(item for item in listed if item["name"] == QUERY_KNOWLEDGE_HUB)
        assert "query" in query_tool["inputSchema"]["required"]

    def test_query_knowledge_hub_returns_citations(self) -> None:
        """完整走通 tools/call query_knowledge_hub，structuredContent 含 citations。"""
        responses = _client_exchange(
            [
                _initialize_message(1),
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": QUERY_KNOWLEDGE_HUB,
                        "arguments": {"query": _FAKE_QUERY, "top_k": 3},
                    },
                },
            ]
        )
        call = _by_id(responses, 3)
        assert "error" not in call, call
        result = call["result"]
        assert result["isError"] is False
        markdown = result["content"][0]["text"]
        assert "[1]" in markdown
        citations = result["structuredContent"]["citations"]
        assert len(citations) == 1
        assert citations[0]["chunk_id"] == _FAKE_CHUNK_ID
        assert citations[0]["source"] == _FAKE_SOURCE
        assert citations[0]["page"] == 3
        assert citations[0]["score"] == pytest.approx(0.91)


if __name__ == "__main__":
    # pytest 导入本模块时不会走这里；子进程带 MCP_E2E_ROLE=server 时充当 Server
    if os.environ.get("MCP_E2E_ROLE") == "server":
        _run_stdio_harness()
    else:
        raise SystemExit("请通过 pytest 运行，或设置 MCP_E2E_ROLE=server 启动 Stdio harness")
