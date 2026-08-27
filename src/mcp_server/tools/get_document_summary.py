"""get_document_summary Tool：按 doc_id 返回文档 title/summary/tags。"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import chromadb

from core.settings import Settings, SettingsError, load_settings, resolve_path
from libs.loader.file_integrity import DEFAULT_INGESTION_HISTORY_DB
from mcp_server.protocol_handler import INVALID_PARAMS, ProtocolHandlerError, ToolDefinition
from observability.logger import get_logger

logger = get_logger("mcp_server.tools.get_document_summary")

TOOL_NAME = "get_document_summary"
TOOL_DESCRIPTION = "按 doc_id（文件哈希或文件名）获取文档的 title、summary 与 tags。"

INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "doc_id": {
            "type": "string",
            "description": "文档标识，可为 doc_hash（SHA256）或文件名（如 sample.pdf）",
        },
        "collection": {
            "type": "string",
            "description": "可选集合名，默认使用 settings.vector_store.collection_name",
        },
    },
    "required": ["doc_id"],
}

SummaryLookup = Callable[[str, Settings, str | None], "DocumentSummary | None"]


@dataclass(frozen=True)
class DocumentSummary:
    """文档摘要结构，对应 MCP structuredContent。"""

    doc_id: str
    title: str
    summary: str
    tags: list[str]
    source_path: str | None = None
    chunk_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "doc_id": self.doc_id,
            "title": self.title,
            "summary": self.summary,
            "tags": list(self.tags),
        }
        if self.source_path is not None:
            payload["source_path"] = self.source_path
        if self.chunk_count is not None:
            payload["chunk_count"] = self.chunk_count
        return payload


def _parse_tags(raw_value: Any) -> list[str]:
    """解析 metadata 中的 tags，兼容 list 与 JSON 字符串。"""
    if isinstance(raw_value, list):
        return [str(item) for item in raw_value if str(item).strip()]
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    return []


def _metadata_to_summary(doc_id: str, metadata: Mapping[str, Any]) -> DocumentSummary:
    """从 Chroma chunk metadata 构造 DocumentSummary。"""
    title = str(metadata.get("title") or metadata.get("doc_title") or "Untitled").strip()
    summary = str(metadata.get("summary") or metadata.get("doc_summary") or title).strip()
    tags = _parse_tags(metadata.get("tags"))
    source_path = metadata.get("source_path")
    source = str(source_path).strip() if isinstance(source_path, str) and source_path.strip() else None
    chunk_count_raw = metadata.get("chunk_count")
    chunk_count = int(chunk_count_raw) if isinstance(chunk_count_raw, int) else None
    return DocumentSummary(
        doc_id=doc_id,
        title=title or "Untitled",
        summary=summary or title,
        tags=tags,
        source_path=source,
        chunk_count=chunk_count,
    )


def _lookup_chroma_metadata(
    settings: Settings,
    doc_id: str,
    collection: str | None,
) -> DocumentSummary | None:
    """从 Chroma 中按 doc_hash 或 source_path 后缀匹配文档 metadata。"""
    collection_name = collection or settings.vector_store.collection_name
    chroma_dir = resolve_path(settings.vector_store.persist_directory)
    try:
        client = chromadb.PersistentClient(path=str(chroma_dir))
        coll = client.get_collection(collection_name)
    except Exception:
        return None

    for where in ({"doc_hash": doc_id}, {"source_path": doc_id}):
        try:
            raw = coll.get(where=where, limit=1, include=["metadatas"])
        except Exception:
            continue
        metadatas = raw.get("metadatas") or []
        if metadatas and isinstance(metadatas[0], Mapping):
            return _metadata_to_summary(doc_id, metadatas[0])

    # 文件名匹配：扫描少量 metadata（MVP 规模可接受）
    try:
        raw = coll.get(limit=200, include=["metadatas"])
    except Exception:
        return None
    for metadata in raw.get("metadatas") or []:
        if not isinstance(metadata, Mapping):
            continue
        source_path = metadata.get("source_path")
        if isinstance(source_path, str) and source_path.replace("\\", "/").endswith(doc_id):
            return _metadata_to_summary(doc_id, metadata)
    return None


def _lookup_ingestion_history(
    doc_id: str,
    history_db: Path | None = None,
) -> DocumentSummary | None:
    """从 ingestion_history 表按 file_hash 或 file_path 后缀回退查询。"""
    db_path = history_db or resolve_path(DEFAULT_INGESTION_HISTORY_DB)
    if not db_path.is_file():
        return None

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            """
            SELECT file_hash, file_path, chunk_count
            FROM ingestion_history
            WHERE status = 'success' AND (file_hash = ? OR file_path LIKE ?)
            ORDER BY processed_at DESC
            LIMIT 1
            """,
            (doc_id, f"%{doc_id}"),
        ).fetchone()

    if row is None:
        return None

    source_path = str(row["file_path"])
    title = Path(source_path).name or doc_id
    chunk_count = row["chunk_count"]
    return DocumentSummary(
        doc_id=doc_id,
        title=title,
        summary=f"文档已摄取，共 {chunk_count} 个 chunk。" if chunk_count else "文档已摄取。",
        tags=[],
        source_path=source_path,
        chunk_count=int(chunk_count) if isinstance(chunk_count, int) else None,
    )


def resolve_document_summary(
    doc_id: str,
    settings: Settings,
    collection: str | None = None,
    *,
    history_db: Path | None = None,
) -> DocumentSummary | None:
    """组合 Chroma metadata 与 ingestion_history 回退，解析文档摘要。"""
    from_chroma = _lookup_chroma_metadata(settings, doc_id, collection)
    if from_chroma is not None:
        return from_chroma
    return _lookup_ingestion_history(doc_id, history_db=history_db)


def _build_markdown(summary: DocumentSummary) -> str:
    """将文档摘要格式化为 Markdown。"""
    tags_text = ", ".join(summary.tags) if summary.tags else "（无）"
    lines = [
        f"## {summary.title}",
        "",
        f"**doc_id**: `{summary.doc_id}`",
        f"**summary**: {summary.summary}",
        f"**tags**: {tags_text}",
    ]
    if summary.source_path:
        lines.append(f"**source**: `{summary.source_path}`")
    if summary.chunk_count is not None:
        lines.append(f"**chunks**: {summary.chunk_count}")
    return "\n".join(lines)


def get_document_summary(
    arguments: Mapping[str, Any],
    *,
    settings: Settings | None = None,
    lookup_fn: SummaryLookup | None = None,
) -> dict[str, Any]:
    """
    MCP Tool 入口：按 doc_id 返回文档摘要。

    Raises:
        ProtocolHandlerError: doc_id 无效或文档不存在时抛出 -32602。
    """
    doc_id = arguments.get("doc_id")
    if not isinstance(doc_id, str) or not doc_id.strip():
        raise ProtocolHandlerError(INVALID_PARAMS, "doc_id 必须是非空字符串")

    collection = arguments.get("collection")
    if collection is not None and (not isinstance(collection, str) or not collection.strip()):
        raise ProtocolHandlerError(INVALID_PARAMS, "collection 必须是非空字符串")

    try:
        resolved_settings = settings or load_settings()
    except SettingsError as exc:
        raise ProtocolHandlerError(INVALID_PARAMS, f"配置加载失败: {exc}") from exc

    lookup = lookup_fn or resolve_document_summary
    summary = lookup(doc_id.strip(), resolved_settings, collection.strip() if collection else None)
    if summary is None:
        raise ProtocolHandlerError(INVALID_PARAMS, f"未找到文档: {doc_id.strip()}")

    logger.info("get_document_summary 完成 — doc_id=%s title=%s", summary.doc_id, summary.title)
    return {
        "content": [{"type": "text", "text": _build_markdown(summary)}],
        "structuredContent": summary.to_dict(),
        "isError": False,
    }


def build_get_document_summary_tool(
    *,
    settings: Settings | None = None,
    lookup_fn: SummaryLookup | None = None,
) -> ToolDefinition:
    """构造可注册到 ProtocolHandler 的 get_document_summary ToolDefinition。"""

    def _handler(arguments: Mapping[str, Any]) -> Mapping[str, Any]:
        return get_document_summary(
            arguments,
            settings=settings,
            lookup_fn=lookup_fn,
        )

    return ToolDefinition(
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
        input_schema=INPUT_SCHEMA,
        handler=_handler,
    )
