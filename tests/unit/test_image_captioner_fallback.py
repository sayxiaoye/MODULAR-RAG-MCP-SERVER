"""ImageCaptioner 降级与 mock Vision LLM 契约测试。"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from core.settings import Settings, VisionLLMSettings, load_settings
from core.types import Chunk, format_image_placeholder
from ingestion.transform.image_captioner import ImageCaptioner, load_image_captioning_prompt
from libs.llm.base_llm import ChatResponse
from libs.llm.base_vision_llm import BaseVisionLLM, VisionLLMError


def _vision_settings(**overrides: object) -> VisionLLMSettings:
    base = {
        "enabled": True,
        "provider": "azure",
        "model": "gpt-4o",
        "max_image_size": 256,
        "api_key": "test-key",
        "api_version": "2024-02-15-preview",
        "azure_endpoint": "https://example.openai.azure.com",
        "deployment_name": "gpt-4o",
    }
    base.update(overrides)
    return VisionLLMSettings(**base)


class FakeVisionLLM(BaseVisionLLM):
    """测试用 Vision LLM：返回固定 caption 并记录调用。"""

    def __init__(
        self,
        caption: str = "系统三层架构示意图",
        should_fail: bool = False,
    ) -> None:
        super().__init__(_vision_settings())
        self.caption = caption
        self.should_fail = should_fail
        self.calls: list[tuple[str, str | bytes]] = []

    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace=None,
    ) -> ChatResponse:
        self.calls.append((text, image_path))
        if self.should_fail:
            raise VisionLLMError("mock vision failure")
        return ChatResponse(content=self.caption, model="fake-vision")


def _settings_with_vision(enabled: bool = True) -> Settings:
    base = load_settings()
    vision = _vision_settings(enabled=enabled)
    return Settings(
        llm=base.llm,
        embedding=base.embedding,
        vector_store=base.vector_store,
        retrieval=base.retrieval,
        rerank=base.rerank,
        evaluation=base.evaluation,
        observability=base.observability,
        ingestion=base.ingestion,
        vision_llm=vision,
    )


def _write_png(path: Path) -> Path:
    buffer = BytesIO()
    Image.new("RGB", (32, 32), color=(40, 80, 120)).save(buffer, format="PNG")
    path.write_bytes(buffer.getvalue())
    return path


def _chunk_with_image(image_id: str, image_path: str) -> Chunk:
    placeholder = format_image_placeholder(image_id)
    text = f"正文前缀 {placeholder} 正文后缀"
    return Chunk(
        id="chunk_img_001",
        text=text,
        metadata={
            "source_path": "docs/with_images.pdf",
            "chunk_index": 0,
            "image_refs": [image_id],
            "images": [{"id": image_id, "path": image_path, "text_offset": 0, "text_length": 10}],
        },
        start_offset=0,
        end_offset=len(text),
        source_ref="doc-img",
    )


@pytest.mark.unit
class TestImageCaptionerEnabled:
    """启用 Vision LLM 时应生成 caption 并注入正文。"""

    def test_generates_caption_in_metadata_and_text(self, tmp_path: Path) -> None:
        image_id = "img_arch_1"
        image_file = _write_png(tmp_path / f"{image_id}.png")
        fake_vision = FakeVisionLLM("架构图包含 API 与数据库层")
        captioner = ImageCaptioner(_settings_with_vision(), vision_llm=fake_vision)

        result = captioner.transform([_chunk_with_image(image_id, str(image_file))])[0]

        assert result.metadata["image_captions"][image_id] == "架构图包含 API 与数据库层"
        assert result.metadata["captioned_by"] == "vision_llm"
        assert "has_unprocessed_images" not in result.metadata
        assert "[图片描述: 架构图包含 API 与数据库层]" in result.text
        assert format_image_placeholder(image_id) not in result.text
        assert len(fake_vision.calls) == 1

    def test_load_prompt_contains_context_placeholder(self) -> None:
        template = load_image_captioning_prompt()
        assert "{context}" in template


@pytest.mark.unit
class TestImageCaptionerFallback:
    """禁用或异常时应保留 image_refs 并标记未处理。"""

    def test_disabled_marks_unprocessed_images(self, tmp_path: Path) -> None:
        image_id = "img_off"
        image_file = _write_png(tmp_path / "off.png")
        captioner = ImageCaptioner(_settings_with_vision(enabled=False))
        chunk = _chunk_with_image(image_id, str(image_file))

        result = captioner.transform([chunk])[0]

        assert result.metadata["image_refs"] == [image_id]
        assert result.metadata.get("has_unprocessed_images") is True
        assert "image_captions" not in result.metadata
        assert result.text == chunk.text

    def test_vision_failure_marks_unprocessed(self, tmp_path: Path) -> None:
        image_id = "img_fail"
        image_file = _write_png(tmp_path / "fail.png")
        captioner = ImageCaptioner(
            _settings_with_vision(),
            vision_llm=FakeVisionLLM(should_fail=True),
        )
        chunk = _chunk_with_image(image_id, str(image_file))
        result = captioner.transform([chunk])[0]

        assert result.metadata.get("has_unprocessed_images") is True
        assert "image_captions" not in result.metadata
        assert result.metadata["image_refs"] == [image_id]

    def test_chunk_without_image_refs_is_unchanged(self) -> None:
        chunk = Chunk(
            id="plain",
            text="纯文本",
            metadata={"source_path": "plain.pdf", "chunk_index": 0},
            start_offset=0,
            end_offset=3,
            source_ref="doc-plain",
        )
        captioner = ImageCaptioner(_settings_with_vision(), vision_llm=FakeVisionLLM())
        result = captioner.transform([chunk])[0]
        assert result.text == chunk.text
        assert "has_unprocessed_images" not in result.metadata

    def test_missing_image_path_marks_unprocessed(self, tmp_path: Path) -> None:
        image_id = "img_missing"
        chunk = _chunk_with_image(image_id, str(tmp_path / "not_exists.png"))
        captioner = ImageCaptioner(_settings_with_vision(), vision_llm=FakeVisionLLM())
        result = captioner.transform([chunk])[0]
        assert result.metadata.get("has_unprocessed_images") is True
