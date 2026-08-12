"""Ollama Embedding Provider 单元测试（mock HTTP，不走真实网络）。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from core.settings import EmbeddingSettings, Settings, load_settings
from libs.embedding.base_embedding import EmbeddingError
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.ollama_embedding import DEFAULT_OLLAMA_BASE_URL, OllamaEmbedding


def _ollama_embed_response(dim: int = 4, count: int = 2) -> dict:
    embeddings = []
    for i in range(count):
        base = 0.1 * (i + 1)
        embeddings.append([base + j * 0.1 for j in range(dim)])
    return {"model": "nomic-embed-text", "embeddings": embeddings}


@pytest.mark.unit
class TestOllamaEmbeddingEmbed:
    """验证 Ollama embed 批量调用与错误处理。"""

    @patch("httpx.post")
    def test_embed_batch_success(self, mock_post: MagicMock) -> None:
        """mock 成功响应时应返回与输入等长的向量列表。"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = ""
        mock_resp.json.return_value = _ollama_embed_response()
        mock_post.return_value = mock_resp

        emb = OllamaEmbedding(
            EmbeddingSettings(
                provider="ollama",
                model="nomic-embed-text",
                dimensions=4,
                base_url=DEFAULT_OLLAMA_BASE_URL,
            )
        )
        vectors = emb.embed(["hello", "world"])
        assert len(vectors) == 2
        assert len(vectors[0]) == 4
        called_url = mock_post.call_args.args[0]
        assert called_url.endswith("/api/embed")
        payload = mock_post.call_args.kwargs["json"]
        assert payload["model"] == "nomic-embed-text"
        assert payload["input"] == ["hello", "world"]
        assert payload["truncate"] is True
        assert payload["dimensions"] == 4

    @patch("httpx.post")
    def test_embed_single_text(self, mock_post: MagicMock) -> None:
        """单条文本也应走批量 input 列表格式。"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = ""
        mock_resp.json.return_value = _ollama_embed_response(count=1, dim=3)
        mock_post.return_value = mock_resp

        emb = OllamaEmbedding(
            EmbeddingSettings(
                provider="ollama",
                model="nomic-embed-text",
                dimensions=3,
            )
        )
        vectors = emb.embed(["single"])
        assert len(vectors) == 1
        assert mock_post.call_args.kwargs["json"]["input"] == ["single"]

    def test_empty_texts_raises(self) -> None:
        """空输入应在基类校验阶段报错。"""
        emb = OllamaEmbedding(
            EmbeddingSettings(
                provider="ollama",
                model="nomic-embed-text",
                dimensions=4,
            )
        )
        with pytest.raises(EmbeddingError, match="texts 不能为空"):
            emb.embed([])

    @patch("httpx.post", side_effect=httpx.ConnectError("connection refused"))
    def test_connect_error_readable(self, mock_post: MagicMock) -> None:
        """连接失败时应给出可读提示。"""
        emb = OllamaEmbedding(
            EmbeddingSettings(
                provider="ollama",
                model="nomic-embed-text",
                dimensions=4,
                base_url="http://localhost:11434",
            )
        )
        with pytest.raises(EmbeddingError, match=r"\[ollama\].*ConnectError"):
            emb.embed(["text"])

    @patch("httpx.post", side_effect=httpx.TimeoutException("timeout"))
    def test_timeout_readable(self, mock_post: MagicMock) -> None:
        """超时应包含 provider 与异常类型。"""
        emb = OllamaEmbedding(
            EmbeddingSettings(
                provider="ollama",
                model="nomic-embed-text",
                dimensions=4,
            )
        )
        with pytest.raises(EmbeddingError, match=r"\[ollama\].*TimeoutException"):
            emb.embed(["text"])

    @patch("httpx.post")
    def test_api_http_error(self, mock_post: MagicMock) -> None:
        """HTTP 4xx/5xx 应包含 provider 与状态码。"""
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "internal error"
        mock_post.return_value = mock_resp

        emb = OllamaEmbedding(
            EmbeddingSettings(
                provider="ollama",
                model="nomic-embed-text",
                dimensions=4,
            )
        )
        with pytest.raises(EmbeddingError, match=r"\[ollama\].*HTTP 500"):
            emb.embed(["text"])


@pytest.mark.unit
class TestOllamaEmbeddingFactoryRouting:
    """验证 EmbeddingFactory 能创建 OllamaEmbedding。"""

    def test_factory_creates_ollama(self) -> None:
        """provider=ollama 时应返回 OllamaEmbedding 实例。"""
        base = load_settings()
        settings = Settings(
            llm=base.llm,
            embedding=EmbeddingSettings(
                provider="ollama",
                model="nomic-embed-text",
                dimensions=768,
                base_url="http://localhost:11434",
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
        assert isinstance(emb, OllamaEmbedding)
