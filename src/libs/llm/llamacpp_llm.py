"""LlamaCpp 本地 LLM 实现（llama-server OpenAI 兼容 API）。"""

from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

import httpx

from core.settings import LLMSettings
from libs.llm.base_llm import ChatMessage, ChatResponse, LLMError, normalize_messages
from libs.llm.openai_compatible import OpenAICompatibleLLM

DEFAULT_LLAMACPP_BASE_URL = "http://localhost:8080/v1"
# 本地推理通常较慢，与 ollama_llm 一致使用较长超时
DEFAULT_LLAMACPP_TIMEOUT = 120.0


class LlamaCppLLM(OpenAICompatibleLLM):
    """通过 llama-server 调用本地 GGUF 模型（OpenAI 兼容 /v1/chat/completions）。"""

    def __init__(self, settings: LLMSettings) -> None:
        base_url = settings.base_url or DEFAULT_LLAMACPP_BASE_URL
        super().__init__(
            settings=settings,
            provider_name="llamacpp",
            base_url=base_url,
            api_key_env="LLAMACPP_API_KEY",
        )

    def _resolve_api_key(self) -> str:
        """llama-server 通常不校验 Key；无配置时使用占位符。"""
        if self.settings.api_key:
            return self.settings.api_key
        return os.environ.get(self.api_key_env, "not-needed")

    def chat(
        self,
        messages: Sequence[ChatMessage | Mapping[str, Any]],
        trace: Any | None = None,
    ) -> ChatResponse:
        """调用 chat/completions；超时与连接错误提示针对本地 llama-server 优化。"""
        normalized = normalize_messages(messages)
        url = self._chat_completions_url()
        payload = self._build_payload(normalized)
        headers = self._build_headers()

        try:
            response = httpx.post(
                url,
                json=payload,
                headers=headers,
                timeout=DEFAULT_LLAMACPP_TIMEOUT,
            )
        except httpx.HTTPError as exc:
            raise LLMError(
                f"[{self.provider_name}] 网络请求失败 ({type(exc).__name__})："
                "请确认 llama-server 已启动，例如 "
                "llama-server -m <model.gguf> --port 8080"
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
