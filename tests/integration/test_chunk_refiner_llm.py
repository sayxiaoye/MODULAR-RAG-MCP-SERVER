"""ChunkRefiner 集成测试：真实 LLM 调用与降级验收。"""

from __future__ import annotations

import httpx
import pytest

from core.settings import IngestionSettings, Settings, load_settings
from core.types import Chunk
from ingestion.transform.chunk_refiner import ChunkRefiner
from libs.llm.llm_factory import LLMFactory


def _chunk(text: str) -> Chunk:
    return Chunk(
        id="integration_chunk",
        text=text,
        metadata={"source_path": "tests/fixtures/sample.pdf", "chunk_index": 0},
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-integration",
    )


def _settings_with_ingestion(**overrides: object) -> Settings:
    base = load_settings()
    ingestion_data = {
        "chunk_size": base.ingestion.chunk_size,
        "chunk_overlap": base.ingestion.chunk_overlap,
        "splitter": base.ingestion.splitter,
        "batch_size": base.ingestion.batch_size,
        "chunk_refiner": {"use_llm": True},
    }
    ingestion_data.update(overrides)
    ingestion = IngestionSettings(**ingestion_data)
    return Settings(
        llm=base.llm,
        embedding=base.embedding,
        vector_store=base.vector_store,
        retrieval=base.retrieval,
        rerank=base.rerank,
        evaluation=base.evaluation,
        observability=base.observability,
        ingestion=ingestion,
        vision_llm=base.vision_llm,
    )


def _ollama_reachable(base_url: str) -> bool:
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/api/tags", timeout=3.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


@pytest.mark.integration
class TestChunkRefinerLLMIntegration:
    """真实 LLM 精炼与无效配置降级（需本地 Ollama 或可用 API）。"""

    def test_real_llm_refinement_when_ollama_available(self) -> None:
        settings = load_settings()
        if settings.llm.provider.lower() != "ollama":
            pytest.skip("当前 settings 非 ollama，跳过真实 LLM 集成测试")
        base_url = settings.llm.base_url or "http://localhost:11434"
        if not _ollama_reachable(base_url):
            pytest.skip(f"Ollama 不可达: {base_url}")

        noisy = "Page 1 of 5\n\n<!-- noise -->\nAzure  配置   说明\n\n---"
        refiner = ChunkRefiner(settings)
        result = refiner.transform([_chunk(noisy)])[0]

        assert result.metadata.get("refined_by") in {"llm", "rule"}
        assert "Azure" in result.text or "配置" in result.text
        assert "Page 1" not in result.text

    def test_invalid_llm_provider_falls_back_to_rule(self) -> None:
        base = load_settings()
        bad_llm = base.llm.__class__(
            provider="nonexistent-provider-xyz",
            model=base.llm.model,
            temperature=base.llm.temperature,
            max_tokens=base.llm.max_tokens,
            api_key=base.llm.api_key,
            api_version=base.llm.api_version,
            azure_endpoint=base.llm.azure_endpoint,
            deployment_name=base.llm.deployment_name,
            base_url=base.llm.base_url,
        )
        settings = Settings(
            llm=bad_llm,
            embedding=base.embedding,
            vector_store=base.vector_store,
            retrieval=base.retrieval,
            rerank=base.rerank,
            evaluation=base.evaluation,
            observability=base.observability,
            ingestion=IngestionSettings(
                chunk_size=base.ingestion.chunk_size,
                chunk_overlap=base.ingestion.chunk_overlap,
                splitter=base.ingestion.splitter,
                batch_size=base.ingestion.batch_size,
                chunk_refiner={"use_llm": True},
            ),
            vision_llm=base.vision_llm,
        )

        refiner = ChunkRefiner(settings)
        chunk = _chunk("Page 2 of 9\n正文保留")
        result = refiner.transform([chunk])[0]
        assert result.metadata["refined_by"] == "rule"
        assert "正文保留" in result.text

    def test_llm_factory_create_with_valid_settings(self) -> None:
        """验收前置：settings 中的 LLM 配置可被工厂解析（不强制联网）。"""
        settings = load_settings()
        if settings.llm.provider.lower() == "ollama":
            base_url = settings.llm.base_url or "http://localhost:11434"
            if not _ollama_reachable(base_url):
                pytest.skip("Ollama 不可达，跳过工厂创建校验")
        llm = LLMFactory.create(settings)
        assert llm is not None
