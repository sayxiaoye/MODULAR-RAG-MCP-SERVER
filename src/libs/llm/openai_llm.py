"""OpenAI 官方 API LLM 实现。"""

from __future__ import annotations

from core.settings import LLMSettings
from libs.llm.openai_compatible import OpenAICompatibleLLM

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


class OpenAILLM(OpenAICompatibleLLM):
    """调用 OpenAI Chat Completions API（api.openai.com）。"""

    def __init__(self, settings: LLMSettings) -> None:
        base_url = settings.base_url or DEFAULT_OPENAI_BASE_URL
        super().__init__(
            settings=settings,
            provider_name="openai",
            base_url=base_url,
            api_key_env="OPENAI_API_KEY",
        )
