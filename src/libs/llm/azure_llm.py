"""Azure OpenAI LLM 实现。"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

import httpx

from core.settings import LLMSettings
from libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse, LLMError, normalize_messages


class AzureLLM(BaseLLM):
    """调用 Azure OpenAI 部署的 Chat 模型（deployment + api-version）。"""

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings

    def _resolve_api_key(self) -> str:
        if self.settings.api_key:
            return self.settings.api_key
        env_key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not env_key:
            raise LLMError(
                "[azure] 缺少 API Key：请配置 llm.api_key 或环境变量 AZURE_OPENAI_API_KEY"
            )
        return env_key

    def _chat_url(self) -> str:
        endpoint = (self.settings.azure_endpoint or "").rstrip("/")
        deployment = self.settings.deployment_name or self.settings.model
        api_version = self.settings.api_version or "2024-02-15-preview"
        if not endpoint:
            raise LLMError("[azure] 缺少必填配置 llm.azure_endpoint")
        if not deployment:
            raise LLMError("[azure] 缺少必填配置 llm.deployment_name 或 llm.model")
        return (
            f"{endpoint}/openai/deployments/{deployment}/chat/completions"
            f"?api-version={api_version}"
        )

    def chat(
        self,
        messages: Sequence[ChatMessage | Mapping[str, Any]],
        trace: Any | None = None,
    ) -> ChatResponse:
        normalized = normalize_messages(messages)
        url = self._chat_url()
        payload = {
            "messages": [{"role": m.role, "content": m.content} for m in normalized],
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }
        headers = {
            "api-key": self._resolve_api_key(),
            "Content-Type": "application/json",
        }

        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=60.0)
        except httpx.HTTPError as exc:
            raise LLMError(f"[azure] 网络请求失败 ({type(exc).__name__}): {exc}") from exc

        if response.status_code >= 400:
            raise LLMError(
                f"[azure] API 错误 HTTP {response.status_code}: {response.text[:300]}"
            )

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("[azure] 响应格式异常，无法解析 choices.message.content") from exc

        return ChatResponse(
            content=str(content),
            model=str(data.get("model", self.settings.deployment_name or self.settings.model)),
            usage=dict(data.get("usage") or {}),
        )
