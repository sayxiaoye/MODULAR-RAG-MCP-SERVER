"""list_collections Tool 单元测试：验证集合发现与 MCP 响应格式。"""

from __future__ import annotations

import json

import pytest

from mcp_server.tools.list_collections import (
    discover_collections,
    list_collections,
)


def _write_bm25_index(path, collection: str, chunk_count: int) -> None:
    """写入最小 BM25 索引文件，供 discover_collections 统计 chunk 数。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "collection": collection,
        "N": chunk_count,
        "avg_doc_length": 10.0,
        "doc_lengths": {f"chunk-{index}": 10 for index in range(chunk_count)},
        "terms": {},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.unit
class TestDiscoverCollections:
    """验证从 fixtures 目录结构发现集合名称。"""

    def test_discover_collections_from_documents_and_bm25(self, tmp_path) -> None:
        """应合并 documents 子目录与 bm25 索引文件名。"""
        documents_root = tmp_path / "documents"
        bm25_root = tmp_path / "db" / "bm25"

        (documents_root / "default").mkdir(parents=True)
        (documents_root / "default" / "a.pdf").write_bytes(b"%PDF-1.4")
        (documents_root / "default" / "b.pdf").write_bytes(b"%PDF-1.4")
        (documents_root / "research").mkdir(parents=True)
        (documents_root / "research" / "paper.pdf").write_bytes(b"%PDF-1.4")

        _write_bm25_index(bm25_root / "default.json", "default", chunk_count=3)
        _write_bm25_index(bm25_root / "archived.json", "archived", chunk_count=1)

        summaries = discover_collections(documents_root=documents_root, bm25_root=bm25_root)
        names = {item.name for item in summaries}
        assert names == {"archived", "default", "research"}

        default = next(item for item in summaries if item.name == "default")
        assert default.document_count == 2
        assert default.chunk_count == 3

        research = next(item for item in summaries if item.name == "research")
        assert research.document_count == 1
        assert research.chunk_count == 0

    def test_discover_collections_empty_roots(self, tmp_path) -> None:
        """空目录应返回空列表。"""
        summaries = discover_collections(
            documents_root=tmp_path / "documents",
            bm25_root=tmp_path / "bm25",
        )
        assert summaries == []


@pytest.mark.unit
class TestListCollectionsTool:
    """验证 MCP Tool 返回 Markdown 与 structuredContent。"""

    def test_list_collections_returns_structured_content(self, tmp_path) -> None:
        documents_root = tmp_path / "documents"
        bm25_root = tmp_path / "db" / "bm25"
        (documents_root / "default").mkdir(parents=True)
        (documents_root / "default" / "sample.pdf").write_bytes(b"%PDF-1.4")
        _write_bm25_index(bm25_root / "default.json", "default", chunk_count=2)

        result = list_collections(
            {},
            documents_root=documents_root,
            bm25_root=bm25_root,
        )

        markdown = result["content"][0]["text"]
        assert "default" in markdown
        assert result["isError"] is False

        collections = result["structuredContent"]["collections"]
        assert len(collections) == 1
        assert collections[0]["name"] == "default"
        assert collections[0]["document_count"] == 1
        assert collections[0]["chunk_count"] == 2

    def test_list_collections_empty_returns_hint(self, tmp_path) -> None:
        result = list_collections(
            {},
            documents_root=tmp_path / "documents",
            bm25_root=tmp_path / "bm25",
        )

        assert "未发现任何文档集合" in result["content"][0]["text"]
        assert result["structuredContent"]["collections"] == []
