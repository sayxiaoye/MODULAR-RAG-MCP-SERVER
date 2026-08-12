"""Ollama 本地 LLM 实现（HTTP /api/chat）。"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import httpx

from core.settings import LLMSettings
from libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse, LLMError, normalize_messages

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaLLM(BaseLLM):
    """通过 Ollama 本地服务调用 Chat 模型。"""

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        self.base_url = (settings.base_url or DEFAULT_OLLAMA_BASE_URL).rstrip("/")

    def _chat_url(self) -> str:
        return f"{self.base_url}/api/chat"

    def chat(
        self,
        messages: Sequence[ChatMessage | Mapping[str, Any]],
        trace: Any | None = None,
    ) -> ChatResponse:
        normalized = normalize_messages(messages)
        payload = {
            "model": self.settings.model,
            "messages": [{"role": m.role, "content": m.content} for m in normalized],
            "stream": False,
            "options": {
                "temperature": self.settings.temperature,
                "num_predict": self.settings.max_tokens,
            },
        }

        try:
            response = httpx.post(self._chat_url(), json=payload, timeout=120.0)
        except httpx.HTTPError as exc:
            # 错误信息不包含 base_url 以外的敏感配置
            raise LLMError(
                f"[ollama] 网络请求失败 ({type(exc).__name__})，请检查 Ollama 服务是否可用"
            ) from exc

        if response.status_code >= 400:
            raise LLMError(
                f"[ollama] API 错误 HTTP {response.status_code}: {response.text[:300]}"
            )

        data = response.json()
        try:
            content = data["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError("[ollama] 响应格式异常，无法解析 message.content") from exc

        return ChatResponse(
            content=str(content),
            model=str(data.get("model", self.settings.model)),
            usage={},
        )
