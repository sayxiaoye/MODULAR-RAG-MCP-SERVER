"""OpenAI 官方 Embedding API 实现。"""

from __future__ import annotations

from typing import Any, Sequence

from core.settings import EmbeddingSettings
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.openai_compatible import (
    request_compatible_embeddings,
    resolve_embedding_api_key,
)

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


class OpenAIEmbedding(BaseEmbedding):
    """调用 OpenAI Embeddings API（api.openai.com/v1/embeddings）。"""

    def __init__(self, settings: EmbeddingSettings) -> None:
        self.settings = settings
        self.base_url = (settings.base_url or DEFAULT_OPENAI_BASE_URL).rstrip("/")

    def embed(self, texts: Sequence[str], trace: Any | None = None) -> list[list[float]]:
        validated = self._validate_texts(texts)
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {resolve_embedding_api_key(self.settings, 'openai', 'OPENAI_API_KEY')}",
            "Content-Type": "application/json",
        }
        return request_compatible_embeddings(
            provider_name="openai",
            url=url,
            headers=headers,
            model=self.settings.model,
            texts=validated,
            dimensions=self.settings.dimensions,
        )
