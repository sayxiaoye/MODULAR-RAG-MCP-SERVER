"""Ollama LLM Provider 单元测试（mock HTTP，不走真实网络）。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from core.settings import LLMSettings, Settings, load_settings
from libs.llm.base_llm import ChatMessage, LLMError
from libs.llm.llm_factory import LLMFactory
from libs.llm.ollama_llm import DEFAULT_OLLAMA_BASE_URL, OllamaLLM


def _ollama_response(content: str = "ollama says hi") -> dict:
    return {
        "model": "llama3",
        "message": {"role": "assistant", "content": content},
        "done": True,
    }


@pytest.mark.unit
class TestOllamaLLMChat:
    """验证 Ollama chat 调用与错误处理。"""

    @patch("httpx.post")
    def test_chat_success(self, mock_post: MagicMock) -> None:
        """mock 成功响应时应返回 assistant 文本。"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = ""
        mock_resp.json.return_value = _ollama_response("本地模型回复")
        mock_post.return_value = mock_resp

        llm = OllamaLLM(
            LLMSettings(
                provider="ollama",
                model="llama3",
                temperature=0.0,
                max_tokens=128,
                base_url=DEFAULT_OLLAMA_BASE_URL,
            )
        )
        response = llm.chat([ChatMessage(role="user", content="你好")])
        assert response.content == "本地模型回复"
        called_url = mock_post.call_args.args[0]
        assert called_url.endswith("/api/chat")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "llama3"
        assert payload["stream"] is False

    @patch("httpx.post", side_effect=httpx.ConnectError("connection refused"))
    def test_connect_error_readable_without_secrets(self, mock_post: MagicMock) -> None:
        """连接失败时应给出可读提示，且不暴露 api_key 等敏感字段。"""
        llm = OllamaLLM(
            LLMSettings(
                provider="ollama",
                model="llama3",
                temperature=0.0,
                max_tokens=64,
                base_url="http://localhost:11434",
                api_key="secret-should-not-appear",
            )
        )
        with pytest.raises(LLMError, match=r"\[ollama\].*ConnectError") as exc_info:
            llm.chat([ChatMessage(role="user", content="q")])
        assert "secret-should-not-appear" not in str(exc_info.value)

    @patch("httpx.post")
    def test_api_http_error(self, mock_post: MagicMock) -> None:
        """HTTP 4xx/5xx 应包含 provider 与状态码。"""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "internal error"
        mock_post.return_value = mock_resp

        llm = OllamaLLM(
            LLMSettings(
                provider="ollama",
                model="llama3",
                temperature=0.0,
                max_tokens=64,
            )
        )
        with pytest.raises(LLMError, match=r"\[ollama\].*HTTP 500"):
            llm.chat([ChatMessage(role="user", content="q")])


@pytest.mark.unit
class TestOllamaFactoryRouting:
    """验证 LLMFactory 能创建 OllamaLLM。"""

    def test_factory_creates_ollama(self) -> None:
        """provider=ollama 时应返回 OllamaLLM 实例。"""
        base = load_settings()
        settings = Settings(
            llm=LLMSettings(
                provider="ollama",
                model="llama3",
                temperature=0.0,
                max_tokens=4096,
                base_url="http://localhost:11434",
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
        assert isinstance(llm, OllamaLLM)
