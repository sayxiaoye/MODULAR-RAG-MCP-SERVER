"""LlamaCpp 本地 Embedding 实现（llama-server OpenAI 兼容 /v1/embeddings）。"""

from __future__ import annotations

import os
from typing import Any, Sequence

import httpx

from core.settings import EmbeddingSettings
from libs.embedding.base_embedding import BaseEmbedding, EmbeddingError
from libs.embedding.openai_compatible import request_compatible_embeddings

DEFAULT_LLAMACPP_EMBED_BASE_URL = "http://localhost:8081/v1"
# 本地 embedding 推理较慢，与 LLM 侧一致使用较长超时
DEFAULT_LLAMACPP_TIMEOUT = 120.0


class LlamaCppEmbedding(BaseEmbedding):
    """通过 llama-server 调用本地 Embedding 模型（OpenAI 兼容 /v1/embeddings）。"""

    def __init__(self, settings: EmbeddingSettings) -> None:
        self.settings = settings
        self.base_url = (settings.base_url or DEFAULT_LLAMACPP_EMBED_BASE_URL).rstrip("/")

    def _resolve_api_key(self) -> str:
        """llama-server 通常不校验 Key；无配置时使用占位符。"""
        if self.settings.api_key:
            return self.settings.api_key
        return os.environ.get("LLAMACPP_API_KEY", "not-needed")

    def embed(self, texts: Sequence[str], trace: Any | None = None) -> list[list[float]]:
        """批量向量化；连接失败时提示启动带 --embedding 的 llama-server。"""
        validated = self._validate_texts(texts)
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self._resolve_api_key()}",
            "Content-Type": "application/json",
        }
        try:
            return request_compatible_embeddings(
                provider_name="llamacpp",
                url=url,
                headers=headers,
                model=self.settings.model,
                texts=validated,
                dimensions=self.settings.dimensions,
            )
        except EmbeddingError as exc:
            message = str(exc)
            if "ConnectError" in message or "TimeoutException" in message:
                raise EmbeddingError(
                    f"[llamacpp] 网络请求失败：请确认 embedding 专用 llama-server 已启动，例如 "
                    "llama-server -m <embed-model.gguf> --port 8081 --embedding"
                ) from exc
            raise
