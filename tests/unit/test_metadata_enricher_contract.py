"""MetadataEnricher 契约测试：规则兜底、LLM mock 与降级。"""

from __future__ import annotations

import json

import pytest

from core.settings import IngestionSettings, Settings, load_settings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.metadata_enricher import (
    MetadataEnricher,
    load_metadata_enrichment_prompt,
)
from libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse, LLMError


class FakeLLM(BaseLLM):
    """测试用 LLM：返回预设 JSON metadata。"""

    def __init__(self, payload: dict[str, object], should_fail: bool = False) -> None:
        self.payload = payload
        self.should_fail = should_fail
        self.last_messages: list[ChatMessage] | None = None

    def chat(self, messages, trace=None) -> ChatResponse:
        self.last_messages = list(messages)
        if self.should_fail:
            raise LLMError("mock llm failure")
        return ChatResponse(content=json.dumps(self.payload, ensure_ascii=False), model="fake")


def _ingestion(**overrides: object) -> IngestionSettings:
    base = load_settings()
    data = {
        "chunk_size": base.ingestion.chunk_size,
        "chunk_overlap": base.ingestion.chunk_overlap,
        "splitter": base.ingestion.splitter,
        "batch_size": base.ingestion.batch_size,
        "chunk_refiner": base.ingestion.chunk_refiner,
        "metadata_enricher": base.ingestion.metadata_enricher,
    }
    data.update(overrides)
    return IngestionSettings(**data)


def _settings(**ingestion_overrides: object) -> Settings:
    base = load_settings()
    ingestion = _ingestion(**ingestion_overrides)
    return Settings(
        llm=base.llm,
        embedding=base.embedding,
        vector_store=base.vector_store,
        retrieval=base.retrieval,
        rerank=base.rerank,
        evaluation=base.evaluation,
        observability=base.observability,
        ingestion=ingestion,
        vision_llm=base.vision_llm,
    )


def _chunk(text: str) -> Chunk:
    return Chunk(
        id="chunk_meta_001",
        text=text,
        metadata={"source_path": "docs/sample.pdf", "chunk_index": 0},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-001",
    )


@pytest.mark.unit
class TestMetadataEnricherRuleMode:
    """规则模式必须产出非空 title/summary/tags。"""

    def test_rule_mode_from_markdown_heading(self) -> None:
        text = "# Azure 配置指南\n\n本节介绍 Azure OpenAI 部署步骤。"
        enricher = MetadataEnricher(_settings(metadata_enricher={"use_llm": False}))
        result = enricher.transform([_chunk(text)])[0]

        assert result.metadata["title"] == "Azure 配置指南"
        assert result.metadata["summary"]
        assert isinstance(result.metadata["tags"], list)
        assert len(result.metadata["tags"]) > 0
        assert result.metadata["enriched_by"] == "rule"

    def test_rule_mode_with_hashtags(self) -> None:
        text = "讨论 #RAG #Retrieval 的基础概念。"
        enricher = MetadataEnricher(_settings())
        tags = enricher._rule_based_enrich(text)["tags"]
        assert "RAG" in tags or "rag" in tags

    def test_rule_mode_plain_text_title(self) -> None:
        text = "简短段落没有标题"
        result = MetadataEnricher(_settings()).transform([_chunk(text)])[0]
        assert result.metadata["title"]
        assert result.metadata["summary"]
        assert result.metadata["tags"]


@pytest.mark.unit
class TestMetadataEnricherLLMMode:
    """LLM 模式与降级契约。"""

    def test_llm_mode_uses_mock_and_sets_enriched_by_llm(self) -> None:
        payload = {
            "title": "LLM Title",
            "summary": "LLM generated summary.",
            "tags": ["azure", "config"],
        }
        enricher = MetadataEnricher(
            _settings(metadata_enricher={"use_llm": True}),
            llm=FakeLLM(payload),
        )
        result = enricher.transform([_chunk("一些正文内容")])[0]

        assert result.metadata["title"] == "LLM Title"
        assert result.metadata["summary"] == "LLM generated summary."
        assert result.metadata["tags"] == ["azure", "config"]
        assert result.metadata["enriched_by"] == "llm"

    def test_llm_failure_falls_back_to_rule(self) -> None:
        enricher = MetadataEnricher(
            _settings(metadata_enricher={"use_llm": True}),
            llm=FakeLLM({}, should_fail=True),
        )
        chunk = _chunk("# 回退标题\n正文内容")
        result = enricher.transform([chunk])[0]

        assert result.metadata["enriched_by"] == "rule"
        assert result.metadata.get("enrichment_fallback") == "llm_failed_or_invalid"
        assert result.metadata["title"] == "回退标题"
        assert result.metadata["tags"]

    def test_llm_invalid_json_falls_back(self) -> None:
        class BadJSONLLM(BaseLLM):
            def chat(self, messages, trace=None) -> ChatResponse:
                return ChatResponse(content="not json", model="fake")

        enricher = MetadataEnricher(
            _settings(metadata_enricher={"use_llm": True}),
            llm=BadJSONLLM(),
        )
        result = enricher.transform([_chunk("正文")])[0]
        assert result.metadata["enriched_by"] == "rule"

    def test_llm_integration_style_mock_in_contract(self) -> None:
        """契约内集成风格用例：模拟真实 LLM JSON 输出结构。"""
        payload = {
            "title": "Integration Style",
            "summary": "Valid summary for integration style test.",
            "tags": ["integration", "metadata"],
        }
        enricher = MetadataEnricher(
            _settings(metadata_enricher={"use_llm": True}),
            llm=FakeLLM(payload),
        )
        trace = TraceContext(trace_type="ingestion")
        result = enricher.transform([_chunk("integration body")], trace=trace)[0]
        summary = trace.finish()

        assert result.metadata["enriched_by"] == "llm"
        assert any(stage["name"] == "metadata_enricher_llm" for stage in summary["stages"])

    def test_transform_preserves_chunk_text_and_id(self) -> None:
        chunk = _chunk("保持不变的内容")
        enricher = MetadataEnricher(_settings())
        result = enricher.transform([chunk])[0]
        assert result.id == chunk.id
        assert result.text == chunk.text
        assert result.source_ref == chunk.source_ref

    def test_load_prompt_contains_text_placeholder(self) -> None:
        template = load_metadata_enrichment_prompt()
        assert "{text}" in template
