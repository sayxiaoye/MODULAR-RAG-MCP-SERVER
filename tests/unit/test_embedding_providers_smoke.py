"""OpenAI / Azure Embedding Provider 冒烟测试（mock HTTP）。"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from core.settings import EmbeddingSettings, Settings, load_settings
from libs.embedding.base_embedding import EmbeddingError
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.openai_embedding import OpenAIEmbedding
from libs.embedding.azure_embedding import AzureEmbedding


def _embedding_response(dim: int = 4, count: int = 2) -> dict:
    data = []
    for i in range(count):
        base = 0.1 * (i + 1)
        data.append({"index": i, "embedding": [base + j * 0.1 for j in range(dim)]})
    return {"data": data, "model": "text-embedding-3-small"}


@pytest.mark.unit
class TestEmbeddingFactoryRouting:
    """验证 EmbeddingFactory 按 provider 路由。"""

    def test_factory_openai_provider(self) -> None:
        base = load_settings()
        settings = _settings_with_embedding(base, provider="openai", model="text-embedding-3-small")
        emb = EmbeddingFactory.create(settings)
        assert isinstance(emb, OpenAIEmbedding)

    def test_factory_azure_provider(self) -> None:
        base = load_settings()
        settings = _settings_with_embedding(
            base,
            provider="azure",
            model="text-embedding-ada-002",
            azure_endpoint="https://my.openai.azure.com",
            deployment_name="embed-deploy",
            api_version="2024-02-15-preview",
            api_key="azure-key",
        )
        emb = EmbeddingFactory.create(settings)
        assert isinstance(emb, AzureEmbedding)


@pytest.mark.unit
class TestOpenAIEmbedding:
    """验证 OpenAI embed 批量调用与错误处理。"""

    @patch("httpx.post")
    def test_embed_batch_success(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = ""
        mock_resp.json.return_value = _embedding_response()
        mock_post.return_value = mock_resp

        emb = OpenAIEmbedding(
            EmbeddingSettings(
                provider="openai",
                model="text-embedding-3-small",
                dimensions=4,
                api_key="sk-test",
            )
        )
        vectors = emb.embed(["hello", "world"])
        assert len(vectors) == 2
        assert len(vectors[0]) == 4
        payload = mock_post.call_args.kwargs["json"]
        assert payload["input"] == ["hello", "world"]
        assert payload["dimensions"] == 4

    def test_empty_texts_raises(self) -> None:
        emb = OpenAIEmbedding(
            EmbeddingSettings(
                provider="openai",
                model="m",
                dimensions=4,
                api_key="k",
            )
        )
        with pytest.raises(EmbeddingError, match="texts 不能为空"):
            emb.embed([])

    @patch("httpx.post")
    def test_api_error_contains_provider(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"
        mock_post.return_value = mock_resp

        emb = OpenAIEmbedding(
            EmbeddingSettings(
                provider="openai",
                model="m",
                dimensions=4,
                api_key="bad",
            )
        )
        with pytest.raises(EmbeddingError, match=r"\[openai\].*HTTP 401"):
            emb.embed(["text"])


@pytest.mark.unit
class TestAzureEmbedding:
    """验证 Azure Embedding 使用 deployment URL 与 api-key 头。"""

    @patch("httpx.post")
    def test_embed_uses_azure_url_and_api_key(self, mock_post: MagicMock) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = ""
        mock_resp.json.return_value = _embedding_response(dim=3, count=1)
        mock_post.return_value = mock_resp

        emb = AzureEmbedding(
            EmbeddingSettings(
                provider="azure",
                model="text-embedding-ada-002",
                dimensions=3,
                api_key="azure-secret",
                azure_endpoint="https://my.openai.azure.com",
                deployment_name="embed-deploy",
                api_version="2024-02-15-preview",
            )
        )
        vectors = emb.embed(["azure text"])
        assert len(vectors) == 1
        url = mock_post.call_args.args[0]
        assert "deployments/embed-deploy/embeddings" in url
        assert "api-version=2024-02-15-preview" in url
        assert mock_post.call_args.kwargs["headers"]["api-key"] == "azure-secret"

    @patch("httpx.post", side_effect=httpx.TimeoutException("timeout"))
    def test_network_timeout_readable(self, mock_post: MagicMock) -> None:
        emb = AzureEmbedding(
            EmbeddingSettings(
                provider="azure",
                model="m",
                dimensions=4,
                api_key="k",
                azure_endpoint="https://x.azure.com",
                deployment_name="d",
            )
        )
        with pytest.raises(EmbeddingError, match=r"\[azure\].*TimeoutException"):
            emb.embed(["t"])


def _settings_with_embedding(base: Settings, **overrides: object) -> Settings:
    data = {
        "provider": base.embedding.provider,
        "model": base.embedding.model,
        "dimensions": base.embedding.dimensions,
        "api_key": base.embedding.api_key,
        "api_version": base.embedding.api_version,
        "azure_endpoint": base.embedding.azure_endpoint,
        "deployment_name": base.embedding.deployment_name,
        "base_url": base.embedding.base_url,
    }
    data.update(overrides)
    embedding = EmbeddingSettings(**data)
    return Settings(
        llm=base.llm,
        embedding=embedding,
        vector_store=base.vector_store,
        retrieval=base.retrieval,
        rerank=base.rerank,
        evaluation=base.evaluation,
        observability=base.observability,
        ingestion=base.ingestion,
        vision_llm=base.vision_llm,
    )
