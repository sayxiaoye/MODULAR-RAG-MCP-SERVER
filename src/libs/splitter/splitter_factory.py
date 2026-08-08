"""Splitter 工厂：按 settings.ingestion.splitter 路由到具体 BaseSplitter 实现。"""

from __future__ import annotations

from typing import Callable, Dict, Type

from core.settings import IngestionSettings, Settings
from libs.splitter.base_splitter import BaseSplitter, SplitterError


class SplitterFactoryError(SplitterError):
    """工厂无法解析或创建切分策略时抛出。"""


# 切分策略名称 -> 实现类，B7.5 阶段注册 Recursive 等真实实现
_SPLITTER_REGISTRY: Dict[str, Type[BaseSplitter]] = {}


def register_splitter(name: str, implementation: Type[BaseSplitter]) -> None:
    """注册 Splitter 实现，策略名与 ingestion.splitter 配置值对应。"""
    key = name.strip().lower()
    if not key:
        raise SplitterFactoryError("Splitter 名称不能为空")
    _SPLITTER_REGISTRY[key] = implementation


def _resolve_splitter_key(settings: Settings) -> str:
    """从 Settings 解析切分策略名（ingestion.splitter）。"""
    ingestion = settings.ingestion
    if ingestion is None:
        raise SplitterFactoryError("缺少 ingestion 配置，无法确定 splitter 策略")
    return ingestion.splitter.strip().lower()


def _default_constructor(settings: Settings) -> BaseSplitter:
    """根据 ingestion.splitter 选择已注册实现并实例化。"""
    key = _resolve_splitter_key(settings)
    if key not in _SPLITTER_REGISTRY:
        known = ", ".join(sorted(_SPLITTER_REGISTRY)) or "（无）"
        raise SplitterFactoryError(
            f"未知的 splitter 策略: {settings.ingestion.splitter!r}，已注册: {known}"
        )
    # 传入 ingestion 段供实现读取 chunk_size 等参数（B7.5 RecursiveSplitter 使用）
    return _SPLITTER_REGISTRY[key](settings.ingestion)


class SplitterFactory:
    """按配置创建 BaseSplitter 实例的工厂入口。"""

    _constructor: Callable[[Settings], BaseSplitter] = _default_constructor

    @classmethod
    def create(cls, settings: Settings) -> BaseSplitter:
        """
        从 Settings 读取 ingestion.splitter 并创建对应切分器。

        Args:
            settings: 项目全局配置，使用其中的 ingestion.splitter 字段。

        Returns:
            已配置的 BaseSplitter 实现。
        """
        return cls._constructor(settings)

    @classmethod
    def set_constructor(cls, constructor: Callable[[Settings], BaseSplitter]) -> None:
        """测试专用：替换默认构造逻辑。"""
        cls._constructor = constructor

    @classmethod
    def reset_constructor(cls) -> None:
        """恢复默认构造逻辑。"""
        cls._constructor = _default_constructor


def _register_builtin_splitters() -> None:
    """注册 B7.5 阶段内置 Splitter 实现。"""
    from libs.splitter.recursive_splitter import RecursiveSplitter

    register_splitter("recursive", RecursiveSplitter)


_register_builtin_splitters()
