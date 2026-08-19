"""LLM 工厂：按 settings.llm.provider 路由到具体 BaseLLM 实现。"""

from __future__ import annotations

from typing import Callable, Dict, Type

from core.settings import LLMSettings, Settings, VisionLLMSettings
from libs.llm.base_llm import BaseLLM, LLMError
from libs.llm.base_vision_llm import BaseVisionLLM


class LLMFactoryError(LLMError):
    """工厂无法解析或创建 Provider 时抛出。"""


# Provider 名称 -> 实现类的注册表，B7 阶段注册真实后端
_LLM_REGISTRY: Dict[str, Type[BaseLLM]] = {}

# Vision LLM Provider 注册表，B8/B9 阶段注册多模态后端
_VISION_LLM_REGISTRY: Dict[str, Type[BaseVisionLLM]] = {}


def register_vision_llm_provider(name: str, implementation: Type[BaseVisionLLM]) -> None:
    """注册 Vision LLM Provider 实现，供 B9 真实后端与测试 Fake 注入。"""
    key = name.strip().lower()
    if not key:
        raise LLMFactoryError("Vision LLM Provider 名称不能为空")
    _VISION_LLM_REGISTRY[key] = implementation


def _default_vision_constructor(settings: VisionLLMSettings) -> BaseVisionLLM:
    """根据 VisionLLMSettings 选择已注册的多模态实现并实例化。"""
    provider = settings.provider.strip().lower()
    if provider not in _VISION_LLM_REGISTRY:
        known = ", ".join(sorted(_VISION_LLM_REGISTRY)) or "（无）"
        raise LLMFactoryError(
            f"未知的 Vision LLM provider: {settings.provider!r}，已注册: {known}"
        )
    return _VISION_LLM_REGISTRY[provider](settings)


def register_llm_provider(name: str, implementation: Type[BaseLLM]) -> None:
    """注册 LLM Provider 实现，供扩展与测试注入 Fake 后端。"""
    key = name.strip().lower()
    if not key:
        raise LLMFactoryError("Provider 名称不能为空")
    _LLM_REGISTRY[key] = implementation


def _default_constructor(settings: LLMSettings) -> BaseLLM:
    """根据 LLMSettings 选择已注册的实现并实例化。"""
    provider = settings.provider.strip().lower()
    if provider not in _LLM_REGISTRY:
        known = ", ".join(sorted(_LLM_REGISTRY)) or "（无）"
        raise LLMFactoryError(
            f"未知的 LLM provider: {settings.provider!r}，已注册: {known}"
        )
    return _LLM_REGISTRY[provider](settings)


class LLMFactory:
    """按配置创建 BaseLLM / BaseVisionLLM 实例的工厂入口。"""

    _constructor: Callable[[LLMSettings], BaseLLM] = _default_constructor
    _vision_constructor: Callable[[VisionLLMSettings], BaseVisionLLM] = _default_vision_constructor

    @classmethod
    def create(cls, settings: Settings) -> BaseLLM:
        """
        从 Settings 读取 llm 配置并创建对应 Provider 实例。

        Args:
            settings: 项目全局配置，使用其中的 llm 段。

        Returns:
            已配置的 BaseLLM 实现。
        """
        return cls._constructor(settings.llm)

    @classmethod
    def set_constructor(cls, constructor: Callable[[LLMSettings], BaseLLM]) -> None:
        """测试专用：替换默认构造逻辑（例如注入 Fake 路由）。"""
        cls._constructor = constructor

    @classmethod
    def reset_constructor(cls) -> None:
        """恢复默认构造逻辑。"""
        cls._constructor = _default_constructor

    @classmethod
    def create_vision_llm(cls, settings: Settings) -> BaseVisionLLM:
        """
        从 Settings 读取 vision_llm 配置并创建多模态 Provider 实例。

        Args:
            settings: 项目全局配置，使用其中的 vision_llm 段。

        Returns:
            已配置的 BaseVisionLLM 实现。

        Raises:
            LLMFactoryError: 缺少 vision_llm 配置、未启用或 provider 未注册。
        """
        vision = settings.vision_llm
        if vision is None:
            raise LLMFactoryError("缺少 vision_llm 配置，无法创建 Vision LLM")
        if not vision.enabled:
            raise LLMFactoryError("vision_llm.enabled=false，无法创建 Vision LLM")
        return cls._vision_constructor(vision)

    @classmethod
    def set_vision_constructor(
        cls,
        constructor: Callable[[VisionLLMSettings], BaseVisionLLM],
    ) -> None:
        """测试专用：替换 Vision LLM 默认构造逻辑。"""
        cls._vision_constructor = constructor

    @classmethod
    def reset_vision_constructor(cls) -> None:
        """恢复 Vision LLM 默认构造逻辑。"""
        cls._vision_constructor = _default_vision_constructor

    @classmethod
    def reset_all_constructors(cls) -> None:
        """同时恢复文本 LLM 与 Vision LLM 的默认构造逻辑。"""
        cls.reset_constructor()
        cls.reset_vision_constructor()


def _register_builtin_providers() -> None:
    """注册 B7 阶段内置 LLM Provider（import 时执行一次）。"""
    from libs.llm.azure_llm import AzureLLM
    from libs.llm.deepseek_llm import DeepSeekLLM
    from libs.llm.llamacpp_llm import LlamaCppLLM
    from libs.llm.ollama_llm import OllamaLLM
    from libs.llm.openai_llm import OpenAILLM

    register_llm_provider("openai", OpenAILLM)
    register_llm_provider("azure", AzureLLM)
    register_llm_provider("deepseek", DeepSeekLLM)
    register_llm_provider("ollama", OllamaLLM)
    register_llm_provider("llamacpp", LlamaCppLLM)


_register_builtin_providers()


def _register_builtin_vision_providers() -> None:
    """注册 B9 阶段内置 Vision LLM Provider。"""
    from libs.llm.azure_vision_llm import AzureVisionLLM

    register_vision_llm_provider("azure", AzureVisionLLM)


_register_builtin_vision_providers()
