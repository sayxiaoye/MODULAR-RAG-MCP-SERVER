"""Dashboard 配置读取服务：把 Settings 格式化为总览页组件卡片。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.settings import Settings, load_settings


@dataclass(frozen=True)
class ComponentCard:
    """单个可插拔组件的展示卡片。"""

    title: str
    provider: str
    summary: str
    extras: dict[str, Any]


class ConfigService:
    """
    封装 Settings 读取与展示格式化，对应 G1 ConfigService。

    不直接访问向量库；集合统计由 Overview 调用 ChromaStore。
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or load_settings()

    @property
    def settings(self) -> Settings:
        return self._settings

    def component_cards(self) -> list[ComponentCard]:
        """按总览页约定输出 LLM / Embedding / Splitter / Reranker / Evaluator 卡片。"""
        settings = self._settings
        ingestion = settings.ingestion
        splitter_name = ingestion.splitter if ingestion is not None else "recursive"
        chunk_size = ingestion.chunk_size if ingestion is not None else 0
        chunk_overlap = ingestion.chunk_overlap if ingestion is not None else 0

        rerank_provider = settings.rerank.provider if settings.rerank.enabled else "none"
        rerank_model = settings.rerank.model if settings.rerank.enabled else "none"
        eval_backends = (
            [settings.evaluation.provider] if settings.evaluation.enabled else []
        )

        return [
            ComponentCard(
                title="LLM",
                provider=settings.llm.provider,
                summary=f"{settings.llm.provider} / {settings.llm.model}",
                extras={"model": settings.llm.model},
            ),
            ComponentCard(
                title="Embedding",
                provider=settings.embedding.provider,
                summary=(
                    f"{settings.embedding.provider} / {settings.embedding.model} / "
                    f"{settings.embedding.dimensions}d"
                ),
                extras={
                    "model": settings.embedding.model,
                    "dimensions": settings.embedding.dimensions,
                },
            ),
            ComponentCard(
                title="Splitter",
                provider=splitter_name,
                summary=f"{splitter_name} / size={chunk_size} / overlap={chunk_overlap}",
                extras={"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
            ),
            ComponentCard(
                title="Reranker",
                provider=rerank_provider,
                summary=f"{rerank_provider} / {rerank_model}",
                extras={"enabled": settings.rerank.enabled, "model": rerank_model},
            ),
            ComponentCard(
                title="Evaluator",
                provider=settings.evaluation.provider,
                summary=(
                    ", ".join(eval_backends)
                    if eval_backends
                    else "未启用"
                ),
                extras={
                    "enabled": settings.evaluation.enabled,
                    "metrics": list(settings.evaluation.metrics),
                },
            ),
        ]
