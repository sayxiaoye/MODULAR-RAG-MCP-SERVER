"""Vision LLM 工厂与 BaseVisionLLM 契约的单元测试。"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from core.settings import Settings, VisionLLMSettings, load_settings
from libs.llm.base_llm import ChatResponse
from libs.llm.base_vision_llm import BaseVisionLLM, VisionLLMError
from libs.llm.llm_factory import LLMFactory, LLMFactoryError, register_vision_llm_provider


class FakeVisionLLM(BaseVisionLLM):
    """测试用 Fake Vision LLM：根据图片字节长度生成确定性描述。"""

    def chat_with_image(self, text: str, image_path: str | bytes, trace=None) -> ChatResponse:
        validated_text = self._validate_text(text)
        image_bytes = self.prepare_image_bytes(image_path)
        return ChatResponse(
            content=f"fake-vision:{validated_text}:{len(image_bytes)}",
            model=self.settings.model,
        )


def _settings_with_vision(
    base: Settings,
    *,
    enabled: bool = True,
    provider: str = "fake",
    model: str = "vision-fake",
) -> Settings:
    return Settings(
        llm=base.llm,
        embedding=base.embedding,
        vector_store=base.vector_store,
        retrieval=base.retrieval,
        rerank=base.rerank,
        evaluation=base.evaluation,
        observability=base.observability,
        ingestion=base.ingestion,
        vision_llm=VisionLLMSettings(
            enabled=enabled,
            provider=provider,
            model=model,
            max_image_size=2048,
        ),
    )


@pytest.fixture(autouse=True)
def _reset_llm_factory() -> None:
    """每个用例前后恢复工厂默认构造器。"""
    LLMFactory.reset_all_constructors()
    yield
    LLMFactory.reset_all_constructors()


@pytest.mark.unit
class TestBaseVisionLLMImageInput:
    """验证多模态输入校验与图片字节准备逻辑。"""

    def test_prepare_image_bytes_from_path(self) -> None:
        """应从本地路径读取图片字节。"""
        vision = FakeVisionLLM(
            VisionLLMSettings(
                enabled=True,
                provider="fake",
                model="m",
                max_image_size=1024,
            )
        )
        with tempfile.NamedTemporaryFile("wb", suffix=".png", delete=False) as tmp:
            tmp.write(b"png-bytes")
            tmp_path = tmp.name
        try:
            assert vision.prepare_image_bytes(tmp_path) == b"png-bytes"
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_prepare_image_bytes_from_raw_bytes(self) -> None:
        """应直接接受 bytes 输入。"""
        vision = FakeVisionLLM(
            VisionLLMSettings(
                enabled=True,
                provider="fake",
                model="m",
                max_image_size=1024,
            )
        )
        assert vision.prepare_image_bytes(b"raw-image") == b"raw-image"

    def test_missing_image_path_raises(self) -> None:
        """不存在的路径应抛出可读错误。"""
        vision = FakeVisionLLM(
            VisionLLMSettings(
                enabled=True,
                provider="fake",
                model="m",
                max_image_size=1024,
            )
        )
        with pytest.raises(VisionLLMError, match="不存在"):
            vision.prepare_image_bytes("not-exists.png")


@pytest.mark.unit
class TestVisionLLMFactoryRouting:
    """验证 create_vision_llm 按 provider 路由。"""

    def test_fake_provider_routing(self) -> None:
        """注册 fake provider 后应能创建并调用 chat_with_image。"""
        register_vision_llm_provider("fake", FakeVisionLLM)
        settings = _settings_with_vision(load_settings(), provider="fake")
        vision_llm = LLMFactory.create_vision_llm(settings)
        assert isinstance(vision_llm, FakeVisionLLM)
        response = vision_llm.chat_with_image("describe", b"img")
        assert response.content == "fake-vision:describe:3"

    def test_missing_vision_config_raises(self) -> None:
        """无 vision_llm 配置时不应创建实例。"""
        base = load_settings()
        no_vision = Settings(
            llm=base.llm,
            embedding=base.embedding,
            vector_store=base.vector_store,
            retrieval=base.retrieval,
            rerank=base.rerank,
            evaluation=base.evaluation,
            observability=base.observability,
            ingestion=base.ingestion,
            vision_llm=None,
        )
        with pytest.raises(LLMFactoryError, match="vision_llm"):
            LLMFactory.create_vision_llm(no_vision)

    def test_disabled_vision_raises(self) -> None:
        """enabled=false 时不应创建 Vision LLM。"""
        settings = _settings_with_vision(load_settings(), enabled=False)
        with pytest.raises(LLMFactoryError, match="enabled=false"):
            LLMFactory.create_vision_llm(settings)

    def test_unknown_provider_raises(self) -> None:
        """未注册的 provider 应抛出可读错误。"""
        settings = _settings_with_vision(load_settings(), provider="unknown_vision_xyz")
        with pytest.raises(LLMFactoryError, match="unknown_vision_xyz"):
            LLMFactory.create_vision_llm(settings)
