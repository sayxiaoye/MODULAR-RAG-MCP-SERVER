"""QueryProcessor 单元测试：关键词提取与 filters 结构。"""

from __future__ import annotations

import pytest

from core.query_engine.query_processor import QueryProcessor, QueryProcessorError
from core.trace.trace_context import TraceContext


@pytest.mark.unit
class TestQueryProcessor:
    """验证关键词非空、停用词策略与 filters 合并。"""

    def test_extracts_keywords_from_mixed_query(self) -> None:
        processor = QueryProcessor()
        result = processor.process("什么是 Azure OpenAI 配置指南？")

        assert result.keywords
        assert "azure" in result.keywords
        assert "openai" in result.keywords
        assert any(k in result.keywords for k in ("配置", "指南", "配置指南"))
        assert result.dense_query == "什么是 Azure OpenAI 配置指南？"
        assert isinstance(result.filters, dict)

    def test_filters_default_empty_dict(self) -> None:
        processor = QueryProcessor()
        result = processor.process("BM25 检索原理")

        assert result.filters == {}

    def test_explicit_filters_are_preserved(self) -> None:
        processor = QueryProcessor()
        result = processor.process(
            "Azure 向量数据库",
            filters={"collection": "docs", "doc_type": "pdf"},
        )

        assert result.filters["collection"] == "docs"
        assert result.filters["doc_type"] == "pdf"

    def test_inline_filters_parsed_from_query(self) -> None:
        processor = QueryProcessor()
        result = processor.process("collection:handbook Azure 部署步骤")

        assert result.filters["collection"] == "handbook"
        assert "azure" in result.keywords
        assert any(k in result.keywords for k in ("部署", "步骤", "部署步骤"))
        assert "collection:handbook" not in result.dense_query

    def test_explicit_filters_override_inline_filters(self) -> None:
        processor = QueryProcessor()
        result = processor.process(
            "collection:inline Azure 配置",
            filters={"collection": "override"},
        )

        assert result.filters["collection"] == "override"

    def test_stopwords_removed_but_keywords_remain_non_empty(self) -> None:
        processor = QueryProcessor()
        result = processor.process("what is the Azure configuration")

        assert result.keywords
        assert "what" not in result.keywords
        assert "the" not in result.keywords
        assert "azure" in result.keywords

    def test_all_stopwords_fallback_to_tokens(self) -> None:
        processor = QueryProcessor()
        result = processor.process("what is the")

        assert result.keywords == ["what", "is", "the"]

    def test_empty_query_raises(self) -> None:
        processor = QueryProcessor()
        with pytest.raises(QueryProcessorError, match="不能为空"):
            processor.process("   ")

    def test_records_trace_stage(self) -> None:
        processor = QueryProcessor()
        trace = TraceContext(trace_type="query")
        processor.process("Azure BM25", trace=trace)

        trace.finish()
        stage_names = [stage["name"] for stage in trace.to_dict()["stages"]]
        assert "query_processor" in stage_names

    def test_processed_query_serializable(self) -> None:
        processor = QueryProcessor()
        payload = processor.process("RAG pipeline").to_dict()

        assert payload["original_query"] == "RAG pipeline"
        assert isinstance(payload["keywords"], list)
        assert isinstance(payload["filters"], dict)
