"""引用生成：将 RetrievalResult 转为 MCP structuredContent.citations。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.types import RetrievalResult


@dataclass(frozen=True)
class Citation:
    """单条引用信息，对应 MCP structuredContent 中 citations 元素。"""

    id: int
    source: str
    page: int | None
    chunk_id: str
    score: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "source": self.source,
            "chunk_id": self.chunk_id,
            "score": self.score,
            "text": self.text,
        }
        if self.page is not None:
            payload["page"] = self.page
        return payload


class CitationGenerator:
    """从检索结果列表生成带序号的引用列表。"""

    @staticmethod
    def _resolve_source(metadata: Mapping[str, Any]) -> str:
        """从 metadata 提取来源文件名，优先使用 source_path。"""
        for key in ("source_path", "source_file", "file_name"):
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return Path(value.strip()).name
        return "unknown"

    @classmethod
    def generate(cls, retrieval_results: Sequence[RetrievalResult]) -> list[Citation]:
        """按检索顺序生成 [1]、[2]... 对应的引用结构。"""
        citations: list[Citation] = []
        for index, item in enumerate(retrieval_results, start=1):
            page = item.metadata.get("page")
            page_value = int(page) if isinstance(page, int) else None
            citations.append(
                Citation(
                    id=index,
                    source=cls._resolve_source(item.metadata),
                    page=page_value,
                    chunk_id=item.chunk_id,
                    score=float(item.score),
                    text=item.text,
                )
            )
        return citations
