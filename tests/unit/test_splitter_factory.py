"""Splitter 工厂与 BaseSplitter 契约的单元测试。"""

from __future__ import annotations

import pytest

from core.settings import IngestionSettings, Settings, load_settings
from libs.splitter.base_splitter import BaseSplitter, SplitterError
from libs.splitter.splitter_factory import (
    SplitterFactory,
    SplitterFactoryError,
    register_splitter,
)


class FakeSplitter(BaseSplitter):
    """测试用 Fake Splitter：按固定长度切分，不依赖外部库。"""

    def __init__(self, settings: IngestionSettings) -> None:
        self.chunk_size = settings.chunk_size

    def split_text(self, text: str, trace=None) -> list[str]:
        validated = self._validate_text(text)
        size = max(1, self.chunk_size)
        return [validated[i : i + size] for i in range(0, len(validated), size)]


@pytest.fixture(autouse=True)
def _reset_splitter_factory() -> None:
    """每个用例前后恢复工厂默认构造器，避免测试间污染。"""
    SplitterFactory.reset_constructor()
    yield
    SplitterFactory.reset_constructor()


@pytest.mark.unit
class TestFakeSplitter:
    """验证 Fake Splitter 基本切分行为。"""

    def test_split_by_chunk_size(self) -> None:
        """应按 ingestion.chunk_size 将文本切成多段。"""
        base = load_settings()
        splitter = FakeSplitter(
            IngestionSettings(
                chunk_size=4,
                chunk_overlap=0,
                splitter="fake",
                batch_size=10,
            )
        )
        chunks = splitter.split_text("abcdefghij")
        assert chunks == ["abcd", "efgh", "ij"]

    def test_empty_text_raises(self) -> None:
        """空文本应抛出可读错误。"""
        base = load_settings()
        splitter = FakeSplitter(base.ingestion)
        with pytest.raises(SplitterError, match="text 必须是非空字符串"):
            splitter.split_text("   ")


@pytest.mark.unit
class TestSplitterFactoryRouting:
    """验证工厂按 ingestion.splitter 路由到已注册实现。"""

    def test_fake_splitter_routing(self) -> None:
        """注册 fake 策略后，工厂应返回 FakeSplitter 并能切分文本。"""
        register_splitter("fake", FakeSplitter)
        base = load_settings()
        fake_settings = Settings(
            llm=base.llm,
            embedding=base.embedding,
            vector_store=base.vector_store,
            retrieval=base.retrieval,
            rerank=base.rerank,
            evaluation=base.evaluation,
            observability=base.observability,
            ingestion=IngestionSettings(
                chunk_size=3,
                chunk_overlap=0,
                splitter="fake",
                batch_size=10,
            ),
            vision_llm=base.vision_llm,
        )
        splitter = SplitterFactory.create(fake_settings)
        assert isinstance(splitter, FakeSplitter)
        assert splitter.split_text("hello") == ["hel", "lo"]

    def test_unknown_splitter_raises(self) -> None:
        """未注册的策略名应抛出包含名称的错误。"""
        base = load_settings()
        unknown = Settings(
            llm=base.llm,
            embedding=base.embedding,
            vector_store=base.vector_store,
            retrieval=base.retrieval,
            rerank=base.rerank,
            evaluation=base.evaluation,
            observability=base.observability,
            ingestion=IngestionSettings(
                chunk_size=100,
                chunk_overlap=0,
                splitter="unknown_splitter_xyz",
                batch_size=10,
            ),
            vision_llm=base.vision_llm,
        )
        with pytest.raises(SplitterFactoryError, match="unknown_splitter_xyz"):
            SplitterFactory.create(unknown)

    def test_missing_ingestion_raises(self) -> None:
        """无 ingestion 配置时不应创建 Splitter。"""
        base = load_settings()
        no_ingestion = Settings(
            llm=base.llm,
            embedding=base.embedding,
            vector_store=base.vector_store,
            retrieval=base.retrieval,
            rerank=base.rerank,
            evaluation=base.evaluation,
            observability=base.observability,
            ingestion=None,
            vision_llm=base.vision_llm,
        )
        with pytest.raises(SplitterFactoryError, match="ingestion"):
            SplitterFactory.create(no_ingestion)
