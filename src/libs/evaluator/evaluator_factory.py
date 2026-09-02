"""Evaluator 工厂：按 settings.evaluation.provider 路由到具体评估实现。"""

from __future__ import annotations

from typing import Callable, Dict, Type

from core.settings import EvaluationSettings, Settings
from libs.evaluator.base_evaluator import BaseEvaluator, EvaluatorError
from libs.evaluator.custom_evaluator import CustomEvaluator


class EvaluatorFactoryError(EvaluatorError):
    """工厂无法解析或创建 Provider 时抛出。"""


_EVALUATOR_REGISTRY: Dict[str, Type[BaseEvaluator]] = {
    "custom": CustomEvaluator,
}


def register_evaluator(name: str, implementation: Type[BaseEvaluator]) -> None:
    """注册 Evaluator Provider（如 ragas），供 H1 等阶段扩展。"""
    key = name.strip().lower()
    if not key:
        raise EvaluatorFactoryError("Provider 名称不能为空")
    _EVALUATOR_REGISTRY[key] = implementation


def _register_builtin_providers() -> None:
    """注册 H1 内置 ragas 实现；延迟导入以免循环依赖。"""
    from observability.evaluation.ragas_evaluator import RagasEvaluator

    register_evaluator("ragas", RagasEvaluator)


def _default_constructor(settings: EvaluationSettings) -> BaseEvaluator:
    """根据 evaluation.provider 选择已注册实现。"""
    provider = settings.provider.strip().lower()
    if provider not in _EVALUATOR_REGISTRY:
        known = ", ".join(sorted(_EVALUATOR_REGISTRY)) or "（无）"
        raise EvaluatorFactoryError(
            f"未知的 Evaluator provider: {settings.provider!r}，已注册: {known}"
        )
    return _EVALUATOR_REGISTRY[provider](settings)


class EvaluatorFactory:
    """按配置创建 BaseEvaluator 实例的工厂入口。"""

    _constructor: Callable[[EvaluationSettings], BaseEvaluator] = _default_constructor

    @classmethod
    def create(cls, settings: Settings) -> BaseEvaluator:
        """从 Settings 读取 evaluation 配置并创建评估器。"""
        return cls._constructor(settings.evaluation)

    @classmethod
    def set_constructor(cls, constructor: Callable[[EvaluationSettings], BaseEvaluator]) -> None:
        """测试专用：替换默认构造逻辑。"""
        cls._constructor = constructor

    @classmethod
    def reset_constructor(cls) -> None:
        """恢复默认构造逻辑。"""
        cls._constructor = _default_constructor


_register_builtin_providers()
