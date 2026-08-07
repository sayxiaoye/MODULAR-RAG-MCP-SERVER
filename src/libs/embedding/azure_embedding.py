"""Azure OpenAI Embedding 实现（复用 OpenAI Embeddings 请求逻辑）。"""

from __future__ import annotations

from typing import Any, Sequence

from core.settings import EmbeddingSettings
from libs.embedding.base_embedding import BaseEmbedding, EmbeddingError
from libs.embedding.openai_compatible import (
    request_compatible_embeddings,
    resolve_embedding_api_key,
)


class AzureEmbedding(BaseEmbedding):
    """调用 Azure OpenAI 部署的 Embedding 模型。"""

    def __init__(self, settings: EmbeddingSettings) -> None:
        self.settings = settings

    def _embeddings_url(self) -> str:
        endpoint = (self.settings.azure_endpoint or "").rstrip("/")
        deployment = self.settings.deployment_name or self.settings.model
        api_version = self.settings.api_version or "2024-02-15-preview"
        if not endpoint:
            raise EmbeddingError("[azure] 缺少必填配置 embedding.azure_endpoint")
        if not deployment:
            raise EmbeddingError("[azure] 缺少必填配置 embedding.deployment_name 或 embedding.model")
        return (
            f"{endpoint}/openai/deployments/{deployment}/embeddings"
            f"?api-version={api_version}"
        )

    def embed(self, texts: Sequence[str], trace: Any | None = None) -> list[list[float]]:
        validated = self._validate_texts(texts)
        headers = {
            "api-key": resolve_embedding_api_key(
                self.settings,
                "azure",
                "AZURE_OPENAI_API_KEY",
            ),
            "Content-Type": "application/json",
        }
        return request_compatible_embeddings(
            provider_name="azure",
            url=self._embeddings_url(),
            headers=headers,
            model=self.settings.model,
            texts=validated,
            dimensions=self.settings.dimensions,
        )
