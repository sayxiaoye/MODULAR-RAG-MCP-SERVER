"""Evaluator 工厂：按 provider 或 backends 列表路由到具体评估实现。"""

from __future__ import annotations

from typing import Callable, Dict, Type

from core.settings import EvaluationSettings, Settings
from libs.evaluator.base_evaluator import BaseEvaluator, EvaluatorError
from libs.evaluator.custom_evaluator import CustomEvaluator

# spec 技术栈写 custom_metrics，注册表键为 custom
_BACKEND_ALIASES: Dict[str, str] = {
    "custom_metrics": "custom",
}


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


def _normalize_backend(name: str) -> str:
    """统一大小写，并把 spec 别名映射到注册表键。"""
    key = name.strip().lower()
    return _BACKEND_ALIASES.get(key, key)


def _resolve_backend_names(settings: EvaluationSettings) -> list[str]:
    """从 backends 取出去重后的名称；未配置则空列表（改走 provider）。"""
    raw_backends = settings.backends or []
    names: list[str] = []
    seen: set[str] = set()
    for item in raw_backends:
        key = _normalize_backend(str(item))
        if not key or key in seen:
            continue
        seen.add(key)
        names.append(key)
    return names


def _construct_evaluator(settings: EvaluationSettings, provider: str) -> BaseEvaluator:
    """按已归一化的 provider 名实例化注册表中的实现。"""
    if provider not in _EVALUATOR_REGISTRY:
        known = ", ".join(sorted(_EVALUATOR_REGISTRY)) or "（无）"
        raise EvaluatorFactoryError(
            f"未知的 Evaluator provider: {provider!r}，已注册: {known}"
        )
    return _EVALUATOR_REGISTRY[provider](settings)


def _default_constructor(settings: EvaluationSettings) -> BaseEvaluator:
    """backends 两项及以上则组合；一项用该 backend；否则回退 provider。"""
    names = _resolve_backend_names(settings)
    if len(names) >= 2:
        # 延迟导入，避免 factory ↔ evaluation 包循环
        from observability.evaluation.composite_evaluator import CompositeEvaluator

        evaluators = [_construct_evaluator(settings, name) for name in names]
        return CompositeEvaluator(evaluators)
    if len(names) == 1:
        return _construct_evaluator(settings, names[0])
    return _construct_evaluator(settings, _normalize_backend(settings.provider))


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
