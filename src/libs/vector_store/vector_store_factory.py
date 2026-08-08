"""VectorStore 工厂：按 settings.vector_store.provider 路由到具体 BaseVectorStore 实现。"""

from __future__ import annotations

from typing import Callable, Dict, Type

from core.settings import Settings, VectorStoreSettings
from libs.vector_store.base_vector_store import BaseVectorStore, VectorStoreError


class VectorStoreFactoryError(VectorStoreError):
    """工厂无法解析或创建 Provider 时抛出。"""


# Provider 名称 -> 实现类，B7.6 阶段注册 ChromaStore 等真实后端
_VECTOR_STORE_REGISTRY: Dict[str, Type[BaseVectorStore]] = {}


def register_vector_store(name: str, implementation: Type[BaseVectorStore]) -> None:
    """注册 VectorStore Provider 实现，供扩展与测试注入 Fake 后端。"""
    key = name.strip().lower()
    if not key:
        raise VectorStoreFactoryError("Provider 名称不能为空")
    _VECTOR_STORE_REGISTRY[key] = implementation


def _default_constructor(settings: VectorStoreSettings) -> BaseVectorStore:
    """根据 VectorStoreSettings 选择已注册实现并实例化。"""
    provider = settings.provider.strip().lower()
    if provider not in _VECTOR_STORE_REGISTRY:
        known = ", ".join(sorted(_VECTOR_STORE_REGISTRY)) or "（无）"
        raise VectorStoreFactoryError(
            f"未知的 VectorStore provider: {settings.provider!r}，已注册: {known}"
        )
    return _VECTOR_STORE_REGISTRY[provider](settings)


class VectorStoreFactory:
    """按配置创建 BaseVectorStore 实例的工厂入口。"""

    _constructor: Callable[[VectorStoreSettings], BaseVectorStore] = _default_constructor

    @classmethod
    def create(cls, settings: Settings) -> BaseVectorStore:
        """
        从 Settings 读取 vector_store 配置并创建对应 Provider 实例。

        Args:
            settings: 项目全局配置，使用其中的 vector_store 段。

        Returns:
            已配置的 BaseVectorStore 实现。
        """
        return cls._constructor(settings.vector_store)

    @classmethod
    def set_constructor(cls, constructor: Callable[[VectorStoreSettings], BaseVectorStore]) -> None:
        """测试专用：替换默认构造逻辑。"""
        cls._constructor = constructor

    @classmethod
    def reset_constructor(cls) -> None:
        """恢复默认构造逻辑。"""
        cls._constructor = _default_constructor


def _register_builtin_providers() -> None:
    """注册 B7.6 阶段内置 VectorStore Provider。"""
    from libs.vector_store.chroma_store import ChromaStore

    register_vector_store("chroma", ChromaStore)


_register_builtin_providers()
