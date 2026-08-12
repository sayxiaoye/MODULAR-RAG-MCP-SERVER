"""DeepSeek LLM 实现（OpenAI 兼容 API）。"""

from __future__ import annotations

from core.settings import LLMSettings
from libs.llm.openai_compatible import OpenAICompatibleLLM

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"


class DeepSeekLLM(OpenAICompatibleLLM):
    """调用 DeepSeek Chat API（OpenAI 兼容协议）。"""

    def __init__(self, settings: LLMSettings) -> None:
        base_url = settings.base_url or DEFAULT_DEEPSEEK_BASE_URL
        super().__init__(
            settings=settings,
            provider_name="deepseek",
            base_url=base_url,
            api_key_env="DEEPSEEK_API_KEY",
        )
