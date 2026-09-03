"""Evaluator 工厂：按 provider 或 backends 列表路由到具体评估实现。"""

from __future__ import annotations

from typing import Any, Callable, Dict, Type

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


def _construct_evaluator(
    eval_settings: EvaluationSettings,
    provider: str,
    app_settings: Settings,
) -> BaseEvaluator:
    """按已归一化的 provider 名实例化注册表中的实现。"""
    if provider not in _EVALUATOR_REGISTRY:
        known = ", ".join(sorted(_EVALUATOR_REGISTRY)) or "（无）"
        raise EvaluatorFactoryError(
            f"未知的 Evaluator provider: {provider!r}，已注册: {known}"
        )
    cls = _EVALUATOR_REGISTRY[provider]
    if provider == "ragas":
        return cls(
            eval_settings,
            llm=_create_judge_llm(app_settings),
            embeddings=_create_judge_embeddings(app_settings),
        )
    return cls(eval_settings)


def _create_judge_llm(settings: Settings) -> Any:
    """用项目 LLM 工厂创建 Ragas Judge，保证走 llamacpp 按需启停或 OpenAI key。"""
    from libs.llm.llm_factory import LLMFactory

    try:
        return LLMFactory.create(settings)
    except Exception as exc:
        raise EvaluatorFactoryError(
            f"无法为 Ragas Judge 创建 LLM（请检查 settings.llm 或 OPENAI_API_KEY）: {exc}"
        ) from exc


def _create_judge_embeddings(settings: Settings) -> Any:
    """用项目 Embedding 工厂创建 Ragas 向量后端，避免回退 OpenAI embedding。"""
    from libs.embedding.embedding_factory import EmbeddingFactory

    try:
        return EmbeddingFactory.create(settings)
    except Exception as exc:
        raise EvaluatorFactoryError(
            f"无法为 Ragas 创建 Embedding（请检查 settings.embedding）: {exc}"
        ) from exc


def _default_constructor(settings: Settings) -> BaseEvaluator:
    """backends 两项及以上则组合；一项用该 backend；否则回退 provider。"""
    eval_settings = settings.evaluation
    names = _resolve_backend_names(eval_settings)
    if len(names) >= 2:
        # 延迟导入，避免 factory ↔ evaluation 包循环
        from observability.evaluation.composite_evaluator import CompositeEvaluator

        evaluators = [_construct_evaluator(eval_settings, name, settings) for name in names]
        return CompositeEvaluator(evaluators)
    if len(names) == 1:
        return _construct_evaluator(eval_settings, names[0], settings)
    return _construct_evaluator(
        eval_settings,
        _normalize_backend(eval_settings.provider),
        settings,
    )


class EvaluatorFactory:
    """按配置创建 BaseEvaluator 实例的工厂入口。"""

    _constructor: Callable[[Settings], BaseEvaluator] = _default_constructor

    @classmethod
    def create(cls, settings: Settings) -> BaseEvaluator:
        """从 Settings 读取 evaluation 配置并创建评估器。"""
        return cls._constructor(settings)

    @classmethod
    def set_constructor(cls, constructor: Callable[[Settings], BaseEvaluator]) -> None:
        """测试专用：替换默认构造逻辑。"""
        cls._constructor = constructor

    @classmethod
    def reset_constructor(cls) -> None:
        """恢复默认构造逻辑。"""
        cls._constructor = _default_constructor


_register_builtin_providers()
