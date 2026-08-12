"""Reranker 工厂：按 settings.rerank 路由，未启用时回退到 NoneReranker。"""

from __future__ import annotations

from typing import Callable, Dict, Type

from core.settings import RerankSettings, Settings
from libs.reranker.base_reranker import BaseReranker, NoneReranker, RerankerError


class RerankerFactoryError(RerankerError):
    """工厂无法解析或创建 Provider 时抛出。"""


# Provider 名称 -> 实现类，B7.7/B7.8 阶段注册 LLM/CrossEncoder 后端
_RERANKER_REGISTRY: Dict[str, Type[BaseReranker]] = {
    "none": NoneReranker,
}


def register_reranker(name: str, implementation: Type[BaseReranker]) -> None:
    """注册 Reranker Provider 实现（如 cross_encoder、llm）。"""
    key = name.strip().lower()
    if not key:
        raise RerankerFactoryError("Provider 名称不能为空")
    _RERANKER_REGISTRY[key] = implementation


def _should_use_none_reranker(settings: RerankSettings) -> bool:
    """未启用重排或显式配置 none 时走空实现，避免调用外部模型。"""
    return not settings.enabled or settings.provider.strip().lower() == "none"


def _default_constructor(settings: RerankSettings) -> BaseReranker:
    """根据 RerankSettings 选择实现；默认回退 NoneReranker。"""
    if _should_use_none_reranker(settings):
        return NoneReranker()

    provider = settings.provider.strip().lower()
    if provider not in _RERANKER_REGISTRY:
        known = ", ".join(sorted(_RERANKER_REGISTRY)) or "（无）"
        raise RerankerFactoryError(
            f"未知的 Reranker provider: {settings.provider!r}，已注册: {known}"
        )
    return _RERANKER_REGISTRY[provider](settings)


class RerankerFactory:
    """按配置创建 BaseReranker 实例的工厂入口。"""

    _constructor: Callable[[RerankSettings], BaseReranker] = _default_constructor

    @classmethod
    def create(cls, settings: Settings) -> BaseReranker:
        """
        从 Settings 读取 rerank 配置并创建 Reranker。

        enabled=false 或 provider=none 时始终返回 NoneReranker。
        """
        return cls._constructor(settings.rerank)

    @classmethod
    def set_constructor(cls, constructor: Callable[[RerankSettings], BaseReranker]) -> None:
        """测试专用：替换默认构造逻辑。"""
        cls._constructor = constructor

    @classmethod
    def reset_constructor(cls) -> None:
        """恢复默认构造逻辑。"""
        cls._constructor = _default_constructor


def _register_builtin_rerankers() -> None:
    """注册 B7.7/B7.8 阶段内置 Reranker Provider。"""
    from libs.reranker.cross_encoder_reranker import CrossEncoderReranker
    from libs.reranker.llm_reranker import LLMReranker

    register_reranker("llm", LLMReranker)
    register_reranker("cross_encoder", CrossEncoderReranker)


_register_builtin_rerankers()
