"""Ollama 本地 Embedding 实现（HTTP /api/embed）。"""

from __future__ import annotations

from typing import Any, Sequence

import httpx

from core.settings import EmbeddingSettings
from libs.embedding.base_embedding import BaseEmbedding, EmbeddingError

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


class OllamaEmbedding(BaseEmbedding):
    """通过 Ollama 本地服务调用 Embedding 模型，支持单条与批量 input。"""

    def __init__(self, settings: EmbeddingSettings) -> None:
        self.settings = settings
        self.base_url = (settings.base_url or DEFAULT_OLLAMA_BASE_URL).rstrip("/")

    def _embed_url(self) -> str:
        return f"{self.base_url}/api/embed"

    def embed(self, texts: Sequence[str], trace: Any | None = None) -> list[list[float]]:
        validated = self._validate_texts(texts)
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "input": validated,
            # 超长文本由 Ollama 按模型上下文截断，避免整批请求失败
            "truncate": True,
        }
        # dimensions>0 时请求降维（Matryoshka 等模型支持）
        if self.settings.dimensions and self.settings.dimensions > 0:
            payload["dimensions"] = self.settings.dimensions

        try:
            response = httpx.post(self._embed_url(), json=payload, timeout=120.0)
        except httpx.HTTPError as exc:
            raise EmbeddingError(
                f"[ollama] 网络请求失败 ({type(exc).__name__})，请检查 Ollama 服务是否可用"
            ) from exc

        if response.status_code >= 400:
            raise EmbeddingError(
                f"[ollama] API 错误 HTTP {response.status_code}: {response.text[:300]}"
            )

        data = response.json()
        try:
            raw_embeddings = data["embeddings"]
            vectors = [list(map(float, item)) for item in raw_embeddings]
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingError(
                "[ollama] 响应格式异常，无法解析 embeddings"
            ) from exc

        if len(vectors) != len(validated):
            raise EmbeddingError(
                f"[ollama] 返回向量数量 {len(vectors)} 与输入文本数量 {len(validated)} 不一致"
            )
        return vectors
