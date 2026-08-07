"""Embedding 工厂：按 settings.embedding.provider 路由到具体 BaseEmbedding 实现。"""

from __future__ import annotations

from typing import Callable, Dict, Type

from core.settings import EmbeddingSettings, Settings
from libs.embedding.base_embedding import BaseEmbedding, EmbeddingError


class EmbeddingFactoryError(EmbeddingError):
    """工厂无法解析或创建 Provider 时抛出。"""


# Provider 名称 -> 实现类，B7 阶段注册 OpenAI/Azure/Ollama 等真实后端
_EMBEDDING_REGISTRY: Dict[str, Type[BaseEmbedding]] = {}


def register_embedding_provider(name: str, implementation: Type[BaseEmbedding]) -> None:
    """注册 Embedding Provider 实现，供扩展与测试注入 Fake 后端。"""
    key = name.strip().lower()
    if not key:
        raise EmbeddingFactoryError("Provider 名称不能为空")
    _EMBEDDING_REGISTRY[key] = implementation


def _default_constructor(settings: EmbeddingSettings) -> BaseEmbedding:
    """根据 EmbeddingSettings 选择已注册实现并实例化。"""
    provider = settings.provider.strip().lower()
    if provider not in _EMBEDDING_REGISTRY:
        known = ", ".join(sorted(_EMBEDDING_REGISTRY)) or "（无）"
        raise EmbeddingFactoryError(
            f"未知的 Embedding provider: {settings.provider!r}，已注册: {known}"
        )
    return _EMBEDDING_REGISTRY[provider](settings)


class EmbeddingFactory:
    """按配置创建 BaseEmbedding 实例的工厂入口。"""

    _constructor: Callable[[EmbeddingSettings], BaseEmbedding] = _default_constructor

    @classmethod
    def create(cls, settings: Settings) -> BaseEmbedding:
        """
        从 Settings 读取 embedding 配置并创建对应 Provider 实例。

        Args:
            settings: 项目全局配置，使用其中的 embedding 段。

        Returns:
            已配置的 BaseEmbedding 实现。
        """
        return cls._constructor(settings.embedding)

    @classmethod
    def set_constructor(cls, constructor: Callable[[EmbeddingSettings], BaseEmbedding]) -> None:
        """测试专用：替换默认构造逻辑。"""
        cls._constructor = constructor

    @classmethod
    def reset_constructor(cls) -> None:
        """恢复默认构造逻辑。"""
        cls._constructor = _default_constructor
