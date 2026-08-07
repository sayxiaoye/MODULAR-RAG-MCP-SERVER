"""OpenAI Chat Completions 兼容调用基类，供 OpenAI / DeepSeek 等 Provider 复用。"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

import httpx

from core.settings import LLMSettings
from libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse, LLMError, normalize_messages


class OpenAICompatibleLLM(BaseLLM):
    """通过 HTTP 调用 OpenAI 兼容 chat/completions 端点的通用实现。"""

    def __init__(
        self,
        settings: LLMSettings,
        provider_name: str,
        base_url: str,
        api_key_env: str,
    ) -> None:
        self.settings = settings
        self.provider_name = provider_name
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env

    def _resolve_api_key(self) -> str:
        """优先使用 settings.api_key，否则读取环境变量。"""
        if self.settings.api_key:
            return self.settings.api_key
        env_key = os.environ.get(self.api_key_env)
        if not env_key:
            raise LLMError(
                f"[{self.provider_name}] 缺少 API Key：请配置 llm.api_key 或环境变量 {self.api_key_env}"
            )
        return env_key

    def _chat_completions_url(self) -> str:
        """构建 chat/completions 请求 URL。"""
        return f"{self.base_url}/chat/completions"

    def _build_payload(self, messages: list[ChatMessage]) -> dict[str, Any]:
        """组装与 OpenAI Chat API 一致的请求体。"""
        return {
            "model": self.settings.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }

    def _build_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._resolve_api_key()}",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        messages: Sequence[ChatMessage | Mapping[str, Any]],
        trace: Any | None = None,
    ) -> ChatResponse:
        normalized = normalize_messages(messages)
        url = self._chat_completions_url()
        payload = self._build_payload(normalized)
        headers = self._build_headers()

        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=60.0)
        except httpx.HTTPError as exc:
            raise LLMError(
                f"[{self.provider_name}] 网络请求失败 ({type(exc).__name__}): {exc}"
            ) from exc

        if response.status_code >= 400:
            raise LLMError(
                f"[{self.provider_name}] API 错误 HTTP {response.status_code}: "
                f"{response.text[:300]}"
            )

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(
                f"[{self.provider_name}] 响应格式异常，无法解析 choices.message.content"
            ) from exc

        return ChatResponse(
            content=str(content),
            model=str(data.get("model", self.settings.model)),
            usage=dict(data.get("usage") or {}),
        )
