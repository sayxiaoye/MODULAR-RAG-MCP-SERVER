"""RecursiveSplitter 单元测试：验证 Markdown 结构感知切分与工厂路由。"""

from __future__ import annotations

import pytest

from core.settings import IngestionSettings, Settings, load_settings
from libs.splitter.base_splitter import SplitterError
from libs.splitter.recursive_splitter import RecursiveSplitter
from libs.splitter.splitter_factory import SplitterFactory


def _ingestion(**overrides: object) -> IngestionSettings:
    base = load_settings()
    data = {
        "chunk_size": base.ingestion.chunk_size,
        "chunk_overlap": base.ingestion.chunk_overlap,
        "splitter": base.ingestion.splitter,
        "batch_size": base.ingestion.batch_size,
    }
    data.update(overrides)
    return IngestionSettings(**data)


@pytest.mark.unit
class TestRecursiveSplitterMarkdown:
    """验证 Markdown 标题与代码块切分行为。"""

    def test_splits_at_heading_boundaries(self) -> None:
        """较长文档应在二级标题边界切分，而非标题行中间。"""
        text = (
            "# Doc\n\n"
            "Intro paragraph with enough text to force a split when combined.\n\n"
            "## Section A\n\n"
            "Content of section A that continues with more detail here.\n\n"
            "## Section B\n\n"
            "Content of section B with additional explanation text."
        )
        splitter = RecursiveSplitter(_ingestion(chunk_size=120, chunk_overlap=0))
        chunks = splitter.split_text(text)
        assert len(chunks) >= 2
        for chunk in chunks:
            assert "## Section" not in chunk or chunk.strip().startswith("## Section")
        combined = "".join(chunks)
        assert "Section A" in combined
        assert "Section B" in combined

    def test_code_block_preserved_when_within_chunk_size(self) -> None:
        """在 chunk_size 允许时，代码块应作为完整片段保留。"""
        code_block = "```python\ndef hello():\n    return 42\n```"
        text = f"# Title\n\nParagraph before code.\n\n{code_block}\n\nAfter code."
        splitter = RecursiveSplitter(_ingestion(chunk_size=500, chunk_overlap=0))
        chunks = splitter.split_text(text)
        assert any("def hello():" in chunk and "return 42" in chunk for chunk in chunks)
        assert not any(chunk.strip() == "return 42" for chunk in chunks)

    def test_empty_text_raises(self) -> None:
        """空文本应抛出可读错误。"""
        splitter = RecursiveSplitter(_ingestion())
        with pytest.raises(SplitterError, match="text 必须是非空字符串"):
            splitter.split_text("   ")

    def test_respects_chunk_size(self) -> None:
        """chunk_size 较小时应产出更多片段。"""
        text = "word " * 200
        small = RecursiveSplitter(_ingestion(chunk_size=50, chunk_overlap=0))
        large = RecursiveSplitter(_ingestion(chunk_size=500, chunk_overlap=0))
        assert len(small.split_text(text)) > len(large.split_text(text))


@pytest.mark.unit
class TestRecursiveSplitterFactoryRouting:
    """验证 SplitterFactory 能创建 RecursiveSplitter。"""

    def test_factory_creates_recursive(self) -> None:
        """provider=recursive 时应返回 RecursiveSplitter 实例。"""
        base = load_settings()
        settings = Settings(
            llm=base.llm,
            embedding=base.embedding,
            vector_store=base.vector_store,
            retrieval=base.retrieval,
            rerank=base.rerank,
            evaluation=base.evaluation,
            observability=base.observability,
            ingestion=_ingestion(splitter="recursive"),
            vision_llm=base.vision_llm,
        )
        splitter = SplitterFactory.create(settings)
        assert isinstance(splitter, RecursiveSplitter)
