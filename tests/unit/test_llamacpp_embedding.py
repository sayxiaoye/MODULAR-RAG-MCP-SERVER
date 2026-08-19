"""LlamaCpp Embedding Provider 单元测试（mock HTTP，不走真实 llama-server）。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from core.settings import EmbeddingSettings, Settings, load_settings
from libs.embedding.base_embedding import EmbeddingError
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.llamacpp_embedding import (
    DEFAULT_LLAMACPP_EMBED_BASE_URL,
    LlamaCppEmbedding,
)


def _embedding_response(dim: int = 4, count: int = 2) -> dict:
    data = []
    for i in range(count):
        base = 0.1 * (i + 1)
        data.append({"index": i, "embedding": [base + j * 0.1 for j in range(dim)]})
    return {"data": data, "model": "nomic-embed-text-v1.5"}


@pytest.mark.unit
class TestLlamaCppEmbeddingEmbed:
    """验证 LlamaCpp embed 批量调用与错误处理。"""

    @patch("httpx.post")
    def test_embed_batch_success_without_api_key(self, mock_post: MagicMock) -> None:
        """无 api_key 时应使用占位符并成功解析 OpenAI 兼容响应。"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = ""
        mock_resp.json.return_value = _embedding_response()
        mock_post.return_value = mock_resp

        emb = LlamaCppEmbedding(
            EmbeddingSettings(
                provider="llamacpp",
                model="nomic-embed-text-v1.5",
                dimensions=4,
            )
        )
        vectors = emb.embed(["hello", "world"])
        assert len(vectors) == 2
        assert len(vectors[0]) == 4
        called_url = mock_post.call_args.args[0]
        assert called_url == f"{DEFAULT_LLAMACPP_EMBED_BASE_URL}/embeddings"
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer not-needed"
        payload = mock_post.call_args.kwargs["json"]
        assert payload["input"] == ["hello", "world"]
        assert payload["dimensions"] == 4

    def test_empty_texts_raises(self) -> None:
        """空输入应在基类校验阶段报错。"""
        emb = LlamaCppEmbedding(
            EmbeddingSettings(
                provider="llamacpp",
                model="nomic-embed-text-v1.5",
                dimensions=768,
            )
        )
        with pytest.raises(EmbeddingError, match="texts 不能为空"):
            emb.embed([])

    @patch("httpx.post", side_effect=httpx.ConnectError("connection refused"))
    def test_connect_error_hints_llama_server(self, mock_post: MagicMock) -> None:
        """连接失败时应提示启动 embedding 专用 llama-server。"""
        emb = LlamaCppEmbedding(
            EmbeddingSettings(
                provider="llamacpp",
                model="nomic-embed-text-v1.5",
                dimensions=768,
                base_url="http://localhost:8081/v1",
            )
        )
        with pytest.raises(EmbeddingError, match=r"\[llamacpp\].*--embedding"):
            emb.embed(["text"])

    @patch("httpx.post")
    def test_api_http_error(self, mock_post: MagicMock) -> None:
        """HTTP 4xx/5xx 应包含 provider 与状态码。"""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "internal error"
        mock_post.return_value = mock_resp

        emb = LlamaCppEmbedding(
            EmbeddingSettings(
                provider="llamacpp",
                model="nomic-embed-text-v1.5",
                dimensions=768,
            )
        )
        with pytest.raises(EmbeddingError, match=r"\[llamacpp\].*HTTP 500"):
            emb.embed(["text"])


@pytest.mark.unit
class TestLlamaCppEmbeddingFactoryRouting:
    """验证 EmbeddingFactory 能创建 LlamaCppEmbedding。"""

    def test_factory_creates_llamacpp(self) -> None:
        """provider=llamacpp 时应返回 LlamaCppEmbedding 实例。"""
        base = load_settings()
        settings = Settings(
            llm=base.llm,
            embedding=EmbeddingSettings(
                provider="llamacpp",
                model="nomic-embed-text-v1.5",
                dimensions=768,
            ),
            vector_store=base.vector_store,
            retrieval=base.retrieval,
            rerank=base.rerank,
            evaluation=base.evaluation,
            observability=base.observability,
            ingestion=base.ingestion,
            vision_llm=base.vision_llm,
        )
        emb = EmbeddingFactory.create(settings)
        assert isinstance(emb, LlamaCppEmbedding)
        assert emb.base_url == DEFAULT_LLAMACPP_EMBED_BASE_URL.rstrip("/")
