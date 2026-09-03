"""黄金集评估前生成答案、以及 Ragas Judge 适配器。"""

from __future__ import annotations

from typing import Any

import pytest

from libs.llm.base_llm import BaseLLM, ChatResponse
from observability.evaluation.answer_generator import generate_rag_answer


class FakeLLM(BaseLLM):
    """记录 chat 入参并返回固定文本。"""

    def __init__(self, content: str = "根据资料，应在 Azure 门户创建资源。") -> None:
        self.content = content
        self.messages: list[Any] = []

    def chat(self, messages, trace=None) -> ChatResponse:
        self.messages.append(list(messages))
        return ChatResponse(content=self.content)


class FakeEmbedding:
    """固定维度的假向量，验证适配器走 embed()。"""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts, trace=None) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(item)), 0.1] for item in texts]


@pytest.mark.unit
class TestAnswerGenerator:
    """验证 RAG 答案 prompt 含 query 与检索片段。"""

    def test_includes_query_and_context(self) -> None:
        """生成时应把问题和资料交给 LLM。"""
        llm = FakeLLM()
        answer = generate_rag_answer("如何配置 Azure？", ["在门户创建资源。"], llm)
        assert answer == "根据资料，应在 Azure 门户创建资源。"
        payload = llm.messages[0][0]
        content = payload["content"] if isinstance(payload, dict) else payload.content
        assert "如何配置 Azure？" in content
        assert "在门户创建资源。" in content

    def test_empty_contexts_still_calls_llm(self) -> None:
        """无检索资料时仍应生成（prompt 标明无资料）。"""
        llm = FakeLLM("不知道")
        answer = generate_rag_answer("q", [], llm)
        assert answer == "不知道"
        content = llm.messages[0][0]["content"]
        assert "无检索资料" in content


@pytest.mark.unit
class TestRagasAdapters:
    """验证项目 LLM / Embedding 被包成 Ragas 可调用对象。"""

    def test_wrap_llm_routes_to_project_chat(self) -> None:
        """Judge 调用应落到 BaseLLM.chat，而不是 OpenAI。"""
        pytest.importorskip("ragas")
        pytest.importorskip("langchain_core")
        from langchain_core.messages import HumanMessage

        from observability.evaluation.ragas_adapters import wrap_project_llm_for_ragas

        llm = FakeLLM("judge-ok")
        wrapped = wrap_project_llm_for_ragas(llm)
        result = wrapped.langchain_llm._generate([HumanMessage(content="打分")])
        text = result.generations[0].message.content
        assert text == "judge-ok"
        assert llm.messages

    def test_wrap_embeddings_routes_to_project_embed(self) -> None:
        """Answer Relevancy 用的向量应走项目 embed()。"""
        pytest.importorskip("ragas")
        from observability.evaluation.ragas_adapters import wrap_project_embeddings_for_ragas

        inner = FakeEmbedding()
        wrapped = wrap_project_embeddings_for_ragas(inner)
        vectors = wrapped.embed_documents(["hello"])
        assert vectors[0][0] == float(len("hello"))
        assert inner.calls == [["hello"]]
