"""LlamaCpp LLM Provider 单元测试（mock HTTP，不走真实 llama-server）。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from core.settings import LLMSettings, Settings, load_settings
from libs.llm.base_llm import ChatMessage, LLMError
from libs.llm.llamacpp_llm import (
    DEFAULT_LLAMACPP_BASE_URL,
    DEFAULT_LLAMACPP_TIMEOUT,
    LlamaCppLLM,
)
from libs.llm.llm_factory import LLMFactory


def _openai_style_response(content: str = "llamacpp says hi") -> dict:
    return {
        "id": "chatcmpl-test",
        "model": "qwen2.5-7b-instruct",
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }


@pytest.mark.unit
class TestLlamaCppLLMChat:
    """验证 LlamaCpp chat 调用、默认配置与错误处理。"""

    @patch("httpx.post")
    def test_chat_success_without_api_key(self, mock_post: MagicMock) -> None:
        """无 api_key 时应使用占位符并成功解析 OpenAI 兼容响应。"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = ""
        mock_resp.json.return_value = _openai_style_response("本地模型回复")
        mock_post.return_value = mock_resp

        llm = LlamaCppLLM(
            LLMSettings(
                provider="llamacpp",
                model="qwen2.5-7b-instruct",
                temperature=0.0,
                max_tokens=128,
            )
        )
        response = llm.chat([ChatMessage(role="user", content="你好")])

        assert response.content == "本地模型回复"
        called_url = mock_post.call_args.args[0]
        assert called_url == f"{DEFAULT_LLAMACPP_BASE_URL}/chat/completions"
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer not-needed"
        assert mock_post.call_args.kwargs["timeout"] == DEFAULT_LLAMACPP_TIMEOUT
        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "qwen2.5-7b-instruct"

    @patch("httpx.post")
    def test_extra_chat_payload_sends_grammar(self, mock_post: MagicMock) -> None:
        """Judge 注入的 extra_chat_payload 应出现在 llama-server 请求体。"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = ""
        mock_resp.json.return_value = _openai_style_response("{}")
        mock_post.return_value = mock_resp

        llm = LlamaCppLLM(
            LLMSettings(
                provider="llamacpp",
                model="qwen2.5-7b-instruct",
                temperature=0.0,
                max_tokens=128,
            )
        )
        llm.extra_chat_payload = {"grammar": "root ::= object"}
        llm.chat([ChatMessage(role="user", content="输出 JSON")])
        payload = mock_post.call_args.kwargs["json"]
        assert payload["grammar"] == "root ::= object"

    @patch("httpx.post", side_effect=httpx.ConnectError("connection refused"))
    def test_connect_error_hints_llama_server(self, mock_post: MagicMock) -> None:
        """连接失败时应提示启动 llama-server，且不泄露 api_key。"""
        llm = LlamaCppLLM(
            LLMSettings(
                provider="llamacpp",
                model="qwen2.5-7b-instruct",
                temperature=0.0,
                max_tokens=64,
                base_url="http://localhost:8080/v1",
                api_key="secret-should-not-appear",
            )
        )
        with pytest.raises(LLMError, match=r"\[llamacpp\].*llama-server") as exc_info:
            llm.chat([ChatMessage(role="user", content="q")])
        assert "secret-should-not-appear" not in str(exc_info.value)

    @patch("httpx.post")
    def test_api_http_error(self, mock_post: MagicMock) -> None:
        """HTTP 4xx/5xx 应包含 provider 与状态码。"""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "internal error"
        mock_post.return_value = mock_resp

        llm = LlamaCppLLM(
            LLMSettings(
                provider="llamacpp",
                model="qwen2.5-7b-instruct",
                temperature=0.0,
                max_tokens=64,
            )
        )
        with pytest.raises(LLMError, match=r"\[llamacpp\].*HTTP 500"):
            llm.chat([ChatMessage(role="user", content="q")])


@pytest.mark.unit
class TestLlamaCppFactoryRouting:
    """验证 LLMFactory 能创建 LlamaCppLLM。"""

    def test_factory_creates_llamacpp(self) -> None:
        """provider=llamacpp 时应返回 LlamaCppLLM 实例。"""
        base = load_settings()
        settings = Settings(
            llm=LLMSettings(
                provider="llamacpp",
                model="qwen2.5-7b-instruct",
                temperature=0.0,
                max_tokens=4096,
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
        llm = LLMFactory.create(settings)
        assert isinstance(llm, LlamaCppLLM)
        assert llm.base_url == DEFAULT_LLAMACPP_BASE_URL.rstrip("/")
