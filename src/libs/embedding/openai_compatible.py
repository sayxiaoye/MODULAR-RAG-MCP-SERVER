"""OpenAI 兼容 Embeddings API 的共享 HTTP 逻辑（非 OpenAI 专属实现）。"""

from __future__ import annotations

import os
from typing import Any

import httpx

from core.settings import EmbeddingSettings
from libs.embedding.base_embedding import EmbeddingError


def resolve_embedding_api_key(
    settings: EmbeddingSettings,
    provider_name: str,
    env_var: str,
) -> str:
    """从配置或环境变量解析 Embedding API Key。"""
    if settings.api_key:
        return settings.api_key
    env_key = os.environ.get(env_var)
    if not env_key:
        raise EmbeddingError(
            f"[{provider_name}] 缺少 API Key：请配置 embedding.api_key 或环境变量 {env_var}"
        )
    return env_key


def request_compatible_embeddings(
    *,
    provider_name: str,
    url: str,
    headers: dict[str, str],
    model: str,
    texts: list[str],
    dimensions: int | None = None,
) -> list[list[float]]:
    """
    调用 OpenAI 兼容格式的 /embeddings 端点，返回与 texts 顺序一致的向量列表。

    供 OpenAIEmbedding、AzureEmbedding 等复用，本身不是某个 Provider 的实现类。
    """
    payload: dict[str, Any] = {
        "input": texts,
        "model": model,
    }
    if dimensions and dimensions > 0:
        payload["dimensions"] = dimensions

    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=60.0)
    except httpx.HTTPError as exc:
        raise EmbeddingError(
            f"[{provider_name}] 网络请求失败 ({type(exc).__name__}): {exc}"
        ) from exc

    if response.status_code >= 400:
        raise EmbeddingError(
            f"[{provider_name}] API 错误 HTTP {response.status_code}: {response.text[:300]}"
        )

    data = response.json()
    try:
        items = sorted(data["data"], key=lambda item: item["index"])
        vectors = [list(map(float, item["embedding"])) for item in items]
    except (KeyError, TypeError, ValueError) as exc:
        raise EmbeddingError(
            f"[{provider_name}] 响应格式异常，无法解析 data[].embedding"
        ) from exc

    if len(vectors) != len(texts):
        raise EmbeddingError(
            f"[{provider_name}] 返回向量数量 {len(vectors)} 与输入文本数量 {len(texts)} 不一致"
        )
    return vectors
