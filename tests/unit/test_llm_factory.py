"""LLM 工厂与 BaseLLM 契约的单元测试。"""

from __future__ import annotations

import pytest

from core.settings import LLMSettings, Settings, load_settings
from libs.llm.base_llm import (
    BaseLLM,
    ChatMessage,
    ChatResponse,
    LLMError,
    normalize_messages,
)
from libs.llm.llm_factory import LLMFactory, LLMFactoryError, register_llm_provider


class FakeLLM(BaseLLM):
    """测试用 Fake LLM：记录 provider 并返回固定响应，不发起网络请求。"""

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings

    def chat(self, messages, trace=None) -> ChatResponse:
        normalized = normalize_messages(messages)
        last = normalized[-1].content
        return ChatResponse(content=f"fake:{last}", model=self.settings.model)


@pytest.fixture(autouse=True)
def _reset_llm_factory() -> None:
    """每个用例前后恢复工厂默认构造器，避免测试间污染。"""
    LLMFactory.reset_constructor()
    yield
    LLMFactory.reset_constructor()


@pytest.mark.unit
class TestChatMessageValidation:
    """验证消息 shape 校验与 normalize_messages 行为。"""

    def test_normalize_dict_messages(self) -> None:
        """dict 形式的消息应被规范化为 ChatMessage。"""
        result = normalize_messages([{"role": "user", "content": "你好"}])
        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].content == "你好"

    def test_empty_messages_raises(self) -> None:
        """空 messages 列表应抛出可读错误。"""
        with pytest.raises(LLMError, match="messages 不能为空"):
            normalize_messages([])

    def test_invalid_role_raises(self) -> None:
        """非法 role 应被拒绝。"""
        with pytest.raises(LLMError, match="无效的消息角色"):
            ChatMessage(role="invalid", content="test")


@pytest.mark.unit
class TestLLMFactoryRouting:
    """验证工厂按 provider 路由到已注册实现。"""

    def test_fake_provider_routing(self) -> None:
        """注册 fake provider 后，工厂应返回 FakeLLM 实例。"""
        register_llm_provider("fake", FakeLLM)
        base = load_settings()
        fake_settings = Settings(
            llm=LLMSettings(
                provider="fake",
                model="fake-model",
                temperature=0.0,
                max_tokens=100,
            ),
            embedding=base.embedding,
            vector_store=base.vector_store,
            retrieval=base.retrieval,
            rerank=base.rerank,
            evaluation=base.evaluation,
            observability=base.observability,
            ingestion=base.ingestion,
            vision_llm=base.vision_llm,
        )
        llm = LLMFactory.create(fake_settings)
        assert isinstance(llm, FakeLLM)
        response = llm.chat([ChatMessage(role="user", content="ping")])
        assert response.content == "fake:ping"
        assert response.model == "fake-model"

    def test_unknown_provider_raises(self) -> None:
        """未注册的 provider 应抛出包含 provider 名称的错误。"""
        base = load_settings()
        unknown = Settings(
            llm=LLMSettings(
                provider="not_registered_xyz",
                model="m",
                temperature=0.0,
                max_tokens=100,
            ),
            embedding=base.embedding,
            vector_store=base.vector_store,
            retrieval=base.retrieval,
            rerank=base.rerank,
            evaluation=base.evaluation,
            observability=base.observability,
            ingestion=base.ingestion,
            vision_llm=base.vision_llm,
        )
        with pytest.raises(LLMFactoryError, match="not_registered_xyz"):
            LLMFactory.create(unknown)
