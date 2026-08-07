"""OpenAI / Azure / DeepSeek LLM Provider 冒烟测试（mock HTTP，不走真实网络）。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from core.settings import LLMSettings, Settings, load_settings
from libs.llm.azure_llm import AzureLLM
from libs.llm.base_llm import ChatMessage, LLMError
from libs.llm.deepseek_llm import DeepSeekLLM
from libs.llm.llm_factory import LLMFactory
from libs.llm.openai_llm import OpenAILLM


def _openai_style_response(content: str = "hello from model") -> dict:
    return {
        "id": "chatcmpl-test",
        "model": "gpt-4o-mini",
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _mock_http_response(status_code: int = 200, json_data: dict | None = None, text: str = "") -> MagicMock:
    mock = MagicMock()
    mock.status_code = status_code
    mock.text = text
    mock.json.return_value = json_data or _openai_style_response()
    return mock


@pytest.mark.unit
class TestLLMFactoryRouting:
    """验证工厂能按 provider 路由到 OpenAI/Azure/DeepSeek 实现。"""

    def test_factory_openai_provider(self) -> None:
        base = load_settings()
        settings = _settings_with_llm(base, provider="openai", model="gpt-4o-mini")
        llm = LLMFactory.create(settings)
        assert isinstance(llm, OpenAILLM)

    def test_factory_azure_provider(self) -> None:
        base = load_settings()
        settings = _settings_with_llm(
            base,
            provider="azure",
            model="gpt-4o",
            azure_endpoint="https://my.openai.azure.com",
            deployment_name="gpt-4o-deploy",
            api_version="2024-02-15-preview",
            api_key="test-key",
        )
        llm = LLMFactory.create(settings)
        assert isinstance(llm, AzureLLM)

    def test_factory_deepseek_provider(self) -> None:
        base = load_settings()
        settings = _settings_with_llm(base, provider="deepseek", model="deepseek-chat")
        llm = LLMFactory.create(settings)
        assert isinstance(llm, DeepSeekLLM)


@pytest.mark.unit
class TestOpenAICompatibleChat:
    """验证 chat 调用与错误信息可读性。"""

    @patch("httpx.post")
    def test_openai_chat_success(self, mock_post: MagicMock) -> None:
        """mock 成功响应时应返回 ChatResponse.content。"""
        mock_post.return_value = _mock_http_response()
        llm = OpenAILLM(
            LLMSettings(
                provider="openai",
                model="gpt-4o-mini",
                temperature=0.0,
                max_tokens=128,
                api_key="sk-test",
            )
        )
        response = llm.chat([ChatMessage(role="user", content="你好")])
        assert response.content == "hello from model"
        mock_post.assert_called_once()

    @patch("httpx.post")
    def test_azure_chat_success(self, mock_post: MagicMock) -> None:
        """Azure 应使用 api-key 头并解析 deployment URL。"""
        mock_post.return_value = _mock_http_response(json_data=_openai_style_response("azure ok"))
        llm = AzureLLM(
            LLMSettings(
                provider="azure",
                model="gpt-4o",
                temperature=0.0,
                max_tokens=128,
                api_key="azure-key",
                azure_endpoint="https://my.openai.azure.com",
                deployment_name="gpt-4o-deploy",
                api_version="2024-02-15-preview",
            )
        )
        response = llm.chat([{"role": "user", "content": "test"}])
        assert response.content == "azure ok"
        called_url = mock_post.call_args.args[0]
        assert "deployments/gpt-4o-deploy/chat/completions" in called_url
        assert mock_post.call_args.kwargs["headers"]["api-key"] == "azure-key"

    @patch("httpx.post")
    def test_api_error_contains_provider_name(self, mock_post: MagicMock) -> None:
        """HTTP 4xx 时错误信息应包含 provider 标识。"""
        mock_post.return_value = _mock_http_response(status_code=401, text="Unauthorized")
        llm = DeepSeekLLM(
            LLMSettings(
                provider="deepseek",
                model="deepseek-chat",
                temperature=0.0,
                max_tokens=64,
                api_key="bad-key",
            )
        )
        with pytest.raises(LLMError, match=r"\[deepseek\].*HTTP 401"):
            llm.chat([ChatMessage(role="user", content="q")])

    def test_invalid_messages_raise_clear_error(self) -> None:
        """非法 messages shape 应在调用前失败。"""
        llm = OpenAILLM(
            LLMSettings(
                provider="openai",
                model="m",
                temperature=0.0,
                max_tokens=10,
                api_key="k",
            )
        )
        with pytest.raises(LLMError, match="messages 不能为空"):
            llm.chat([])

    @patch("httpx.post", side_effect=httpx.ConnectError("connection refused"))
    def test_network_error_contains_error_type(self, mock_post: MagicMock) -> None:
        """网络异常应包含异常类型名，便于排查。"""
        llm = OpenAILLM(
            LLMSettings(
                provider="openai",
                model="m",
                temperature=0.0,
                max_tokens=10,
                api_key="k",
            )
        )
        with pytest.raises(LLMError, match=r"\[openai\].*ConnectError"):
            llm.chat([ChatMessage(role="user", content="q")])


def _settings_with_llm(base: Settings, **llm_overrides: object) -> Settings:
    """构造带自定义 llm 段的 Settings，便于工厂路由测试。"""
    llm_data = {
        "provider": base.llm.provider,
        "model": base.llm.model,
        "temperature": base.llm.temperature,
        "max_tokens": base.llm.max_tokens,
        "api_key": base.llm.api_key,
        "api_version": base.llm.api_version,
        "azure_endpoint": base.llm.azure_endpoint,
        "deployment_name": base.llm.deployment_name,
        "base_url": base.llm.base_url,
    }
    llm_data.update(llm_overrides)
    llm = LLMSettings(**llm_data)
    return Settings(
        llm=llm,
        embedding=base.embedding,
        vector_store=base.vector_store,
        retrieval=base.retrieval,
        rerank=base.rerank,
        evaluation=base.evaluation,
        observability=base.observability,
        ingestion=base.ingestion,
        vision_llm=base.vision_llm,
    )
