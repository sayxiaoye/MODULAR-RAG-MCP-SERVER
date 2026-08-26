"""list_collections Tool：列举知识库集合及基础统计信息。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.query_engine.query_pipeline import resolve_data_paths
from core.settings import Settings, SettingsError, load_settings, resolve_path
from mcp_server.protocol_handler import ToolDefinition
from observability.logger import get_logger

logger = get_logger("mcp_server.tools.list_collections")

TOOL_NAME = "list_collections"
TOOL_DESCRIPTION = "列举知识库中可用的文档集合，并返回文档数与 chunk 数统计。"

INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
}


_EMPTY_HINT = "当前未发现任何文档集合。请先运行 ingest.py 摄取数据。"


@dataclass(frozen=True)
class CollectionSummary:
    """单个集合的汇总信息，供 MCP structuredContent 使用。"""

    name: str
    document_count: int
    chunk_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "document_count": self.document_count,
            "chunk_count": self.chunk_count,
        }


def _count_pdf_files(collection_dir: Path) -> int:
    """统计集合目录下的 PDF 文件数量。"""
    if not collection_dir.is_dir():
        return 0
    return sum(1 for path in collection_dir.glob("*.pdf") if path.is_file())


def _read_bm25_chunk_count(collection: str, bm25_root: Path) -> int:
    """从 BM25 索引文件读取 chunk 数（N 字段），不存在则返回 0。"""
    index_path = bm25_root / f"{collection}.json"
    if not index_path.is_file():
        return 0
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    chunk_count = payload.get("N", 0)
    return int(chunk_count) if isinstance(chunk_count, int) else 0


def discover_collections(
    documents_root: Path | None = None,
    bm25_root: Path | None = None,
) -> list[CollectionSummary]:
    """
    发现可用集合名称并汇总统计。

    集合来源：
    1. ``data/documents/`` 下的子目录名
    2. ``data/db/bm25/*.json`` 的索引文件名（覆盖仅 ingest 未落盘文档目录的场景）
    """
    docs_root = documents_root or resolve_path("data/documents")
    bm25_dir = bm25_root or resolve_data_paths(None)["bm25"]

    names: set[str] = set()
    if docs_root.is_dir():
        names.update(item.name for item in docs_root.iterdir() if item.is_dir())
    if bm25_dir.is_dir():
        names.update(path.stem for path in bm25_dir.glob("*.json") if path.is_file())

    summaries: list[CollectionSummary] = []
    for name in sorted(names):
        document_count = _count_pdf_files(docs_root / name)
        chunk_count = _read_bm25_chunk_count(name, bm25_dir)
        summaries.append(
            CollectionSummary(
                name=name,
                document_count=document_count,
                chunk_count=chunk_count,
            )
        )
    return summaries


def _build_markdown(collections: Sequence[CollectionSummary]) -> str:
    """将集合列表格式化为 Markdown 文本。"""
    if not collections:
        return _EMPTY_HINT

    lines = ["## 可用文档集合", ""]
    for index, item in enumerate(collections, start=1):
        lines.append(
            f"{index}. **{item.name}** — 文档 {item.document_count} 个，"
            f"chunk {item.chunk_count} 个"
        )
    return "\n".join(lines)


def list_collections(
    arguments: Mapping[str, Any] | None = None,
    *,
    settings: Settings | None = None,
    documents_root: Path | None = None,
    bm25_root: Path | None = None,
) -> dict[str, Any]:
    """
    MCP Tool 入口：返回集合列表与统计信息。

    Args:
        arguments: tools/call 参数字典（当前无必填参数，保留扩展位）。
        settings: 可选配置注入。
        documents_root: 测试用文档根目录覆盖。
        bm25_root: 测试用 BM25 根目录覆盖。
    """
    _ = arguments or {}
    try:
        _ = settings or load_settings()
    except SettingsError as exc:
        from mcp_server.protocol_handler import INVALID_PARAMS, ProtocolHandlerError

        raise ProtocolHandlerError(INVALID_PARAMS, f"配置加载失败: {exc}") from exc

    resolved_bm25 = bm25_root
    if resolved_bm25 is None:
        resolved_bm25 = resolve_data_paths(None)["bm25"]

    collections = discover_collections(
        documents_root=documents_root,
        bm25_root=resolved_bm25,
    )
    markdown = _build_markdown(collections)
    logger.info("list_collections 完成 — count=%d", len(collections))

    return {
        "content": [{"type": "text", "text": markdown}],
        "structuredContent": {
            "collections": [item.to_dict() for item in collections],
        },
        "isError": False,
    }


def build_list_collections_tool(
    *,
    settings: Settings | None = None,
    documents_root: Path | None = None,
    bm25_root: Path | None = None,
) -> ToolDefinition:
    """构造可注册到 ProtocolHandler 的 list_collections ToolDefinition。"""

    def _handler(arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        return list_collections(
            arguments,
            settings=settings,
            documents_root=documents_root,
            bm25_root=bm25_root,
        )

    return ToolDefinition(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=INPUT_SCHEMA,
        handler=_handler,
    )
