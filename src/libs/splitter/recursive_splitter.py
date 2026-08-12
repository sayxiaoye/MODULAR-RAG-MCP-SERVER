"""RecursiveCharacterTextSplitter 默认切分器：封装 LangChain，适配 Markdown 结构。"""

from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.settings import IngestionSettings
from libs.splitter.base_splitter import BaseSplitter

# Markdown 语义断点：优先在标题与段落边界切分，尽量保持代码块在 chunk_size 内完整
DEFAULT_MARKDOWN_SEPARATORS = [
    "\n## ",
    "\n### ",
    "\n#### ",
    "\n# ",
    "\n\n",
    "\n",
    " ",
    "",
]


class RecursiveSplitter(BaseSplitter):
    """基于 LangChain RecursiveCharacterTextSplitter 的 Markdown 感知切分器。"""

    def __init__(self, settings: IngestionSettings) -> None:
        self.settings = settings
        chunk_size = max(1, settings.chunk_size)
        chunk_overlap = max(0, settings.chunk_overlap)
        if chunk_overlap >= chunk_size:
            chunk_overlap = max(0, chunk_size // 5)
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=DEFAULT_MARKDOWN_SEPARATORS,
            length_function=len,
            is_separator_regex=False,
        )

    def split_text(self, text: str, trace: Any | None = None) -> list[str]:
        validated = self._validate_text(text)
        return self._splitter.split_text(validated)
