"""LLM Reranker 单元测试（mock LLM，不走真实 API）。"""

from __future__ import annotations

import pytest

from core.settings import RerankSettings, Settings, load_settings
from libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse, LLMError
from libs.reranker.base_reranker import RerankerError, RerankerFallbackSignal
from libs.reranker.llm_reranker import LLMReranker, load_rerank_prompt_template
from libs.reranker.reranker_factory import RerankerFactory


def _sample_candidates() -> list[dict[str, object]]:
    return [
        {
            "id": "a",
            "score": 0.9,
            "text": "first passage",
            "metadata": {"source_path": "doc1.pdf"},
        },
        {
            "id": "b",
            "score": 0.5,
            "text": "second passage",
            "metadata": {"source_path": "doc2.pdf"},
        },
    ]


class FakeLLM(BaseLLM):
    """测试用 LLM：返回预设响应内容。"""

    def __init__(self, content: str, should_fail: bool = False) -> None:
        self.content = content
        self.should_fail = should_fail

    def chat(self, messages, trace=None) -> ChatResponse:
        if self.should_fail:
            raise LLMError("mock llm failure")
        return ChatResponse(content=self.content, model="fake", usage={})


@pytest.mark.unit
class TestLLMRerankerRerank:
    """验证 LLM Reranker 结构化输出与错误处理。"""

    def test_rerank_by_ranked_ids(self) -> None:
        """合法 ranked_ids 应按 LLM 顺序重排候选。"""
        reranker = LLMReranker(
            RerankSettings(enabled=True, provider="llm", model="m", top_k=5),
            llm=FakeLLM('{"ranked_ids": ["b", "a"]}'),
            prompt_template="Q:{query}\nP:{passages}",
        )
        result = reranker.rerank("azure 配置", _sample_candidates())
        assert [item["id"] for item in result] == ["b", "a"]

    def test_custom_prompt_template_injected(self) -> None:
        """测试应能注入替代 prompt 模板。"""
        template = "CUSTOM {query} :: {passages}"
        reranker = LLMReranker(
            RerankSettings(enabled=True, provider="llm", model="m", top_k=5),
            llm=FakeLLM('{"ranked_ids": ["a"]}'),
            prompt_template=template,
        )
        captured: list[str] = []

        class CaptureLLM(FakeLLM):
            def chat(self, messages, trace=None) -> ChatResponse:
                captured.append(messages[0].content)
                return super().chat(messages, trace=trace)

        reranker._llm = CaptureLLM('{"ranked_ids": ["a"]}')
        reranker.rerank("hello", _sample_candidates())
        assert "CUSTOM hello ::" in captured[0]
        assert "id=a" in captured[0]

    def test_invalid_schema_raises(self) -> None:
        """不满足 schema 的 JSON 应抛出可读错误。"""
        reranker = LLMReranker(
            RerankSettings(enabled=True, provider="llm", model="m", top_k=5),
            llm=FakeLLM('{"wrong_field": ["a"]}'),
            prompt_template="Q:{query}\nP:{passages}",
        )
        with pytest.raises(RerankerError, match="ranked_ids"):
            reranker.rerank("q", _sample_candidates())

    def test_llm_failure_raises_fallback_signal(self) -> None:
        """LLM 调用失败时应抛出 RerankerFallbackSignal。"""
        reranker = LLMReranker(
            RerankSettings(enabled=True, provider="llm", model="m", top_k=5),
            llm=FakeLLM("", should_fail=True),
            prompt_template="Q:{query}\nP:{passages}",
        )
        with pytest.raises(RerankerFallbackSignal, match="回退"):
            reranker.rerank("q", _sample_candidates())

    def test_top_k_limits_output(self) -> None:
        """rerank.top_k 应限制返回条数。"""
        reranker = LLMReranker(
            RerankSettings(enabled=True, provider="llm", model="m", top_k=1),
            llm=FakeLLM('{"ranked_ids": ["b", "a"]}'),
            prompt_template="Q:{query}\nP:{passages}",
        )
        result = reranker.rerank("q", _sample_candidates())
        assert len(result) == 1
        assert result[0]["id"] == "b"

    def test_load_default_prompt_file(self) -> None:
        """默认 rerank.txt 应包含 query 与 passages 占位符。"""
        template = load_rerank_prompt_template()
        assert "{query}" in template
        assert "{passages}" in template


@pytest.mark.unit
class TestLLMRerankerFactoryRouting:
    """验证 RerankerFactory 在 backend=llm 时可创建 LLMReranker。"""

    def test_factory_creates_llm_reranker(self) -> None:
        """enabled=true 且 provider=llm 时应返回 LLMReranker。"""
        base = load_settings()
        settings = Settings(
            llm=base.llm,
            embedding=base.embedding,
            vector_store=base.vector_store,
            retrieval=base.retrieval,
            rerank=RerankSettings(
                enabled=True,
                provider="llm",
                model=base.llm.model,
                top_k=5,
            ),
            evaluation=base.evaluation,
            observability=base.observability,
            ingestion=base.ingestion,
            vision_llm=base.vision_llm,
        )
        reranker = RerankerFactory.create(settings)
        assert isinstance(reranker, LLMReranker)
