"""ImageCaptioner：为含图片引用的 Chunk 生成 Vision LLM 描述，失败时降级。"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.settings import REPO_ROOT, Settings, resolve_path
from core.trace.trace_context import TraceContext
from core.types import Chunk, format_image_placeholder
from ingestion.transform.base_transform import BaseTransform, TransformError
from libs.llm.base_llm import ChatResponse
from libs.llm.base_vision_llm import BaseVisionLLM, VisionLLMError
from libs.llm.llm_factory import LLMFactory

logger = logging.getLogger(__name__)

DEFAULT_IMAGE_CAPTION_PROMPT_PATH = REPO_ROOT / "config" / "prompts" / "image_captioning.txt"
_CAPTION_TEMPLATE = "[图片描述: {caption}]"
_FALLBACK_PROMPT = (
    "Describe the image in detail for retrieval. Focus on visible text, objects, and context.\n\n"
    "Image context: {context}\n"
)


def load_image_captioning_prompt(path: str | Path | None = None) -> str:
    """加载图片描述 prompt 模板，必须含 ``{context}`` 占位符。"""
    prompt_path = Path(path) if path is not None else DEFAULT_IMAGE_CAPTION_PROMPT_PATH
    if not prompt_path.is_absolute():
        prompt_path = resolve_path(prompt_path)
    if not prompt_path.is_file():
        raise TransformError(f"image captioning prompt 文件不存在: {prompt_path}")
    template = prompt_path.read_text(encoding="utf-8")
    if "{context}" not in template:
        raise TransformError("image captioning prompt 必须包含 {context} 占位符")
    return template


class ImageCaptioner(BaseTransform):
    """当 Vision LLM 可用且 Chunk 含 image_refs 时生成 caption 并写回 metadata/text。"""

    def __init__(
        self,
        settings: Settings,
        vision_llm: BaseVisionLLM | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        self._settings = settings
        self._vision_llm = vision_llm
        self._prompt_path = prompt_path
        self._prompt_template: str | None = None

    def transform(
        self,
        chunks: Sequence[Chunk],
        trace: Any | None = None,
    ) -> list[Chunk]:
        """逐 Chunk 生成图片描述；失败时标记 has_unprocessed_images 且不阻塞。"""
        captioned: list[Chunk] = []
        for chunk in chunks:
            try:
                captioned.append(self._caption_single_chunk(chunk, trace))
            except Exception as exc:
                logger.warning("图片描述失败，标记未处理图片: %s", exc)
                captioned.append(self._mark_unprocessed(chunk))
        return captioned

    def _caption_single_chunk(self, chunk: Chunk, trace: Any | None) -> Chunk:
        image_refs = chunk.metadata.get("image_refs")
        if not isinstance(image_refs, list) or not image_refs:
            return chunk

        if not self._is_enabled():
            return self._mark_unprocessed(chunk)

        image_lookup = self._build_image_path_lookup(chunk.metadata.get("images"))
        captions: dict[str, str] = {}
        failures = 0

        vision = self._vision_llm
        if vision is None:
            try:
                vision = LLMFactory.create_vision_llm(self._settings)
            except Exception as exc:
                logger.warning("无法创建 Vision LLM，降级为未处理图片: %s", exc)
                return self._mark_unprocessed(chunk)

        prompt_template = self._load_prompt()
        context = chunk.text[:500]

        for image_id in image_refs:
            image_id_str = str(image_id).strip()
            if not image_id_str:
                failures += 1
                continue
            image_path = image_lookup.get(image_id_str)
            if not image_path or not Path(image_path).is_file():
                failures += 1
                continue

            start = time.perf_counter()
            try:
                prompt = prompt_template.replace("{context}", context)
                response = vision.chat_with_image(prompt, image_path, trace=trace)
                caption = response.content.strip()
                if not caption:
                    failures += 1
                    continue
                captions[image_id_str] = caption
                self._record_trace(
                    trace,
                    "image_captioner",
                    elapsed_ms=(time.perf_counter() - start) * 1000,
                    method="vision_llm",
                    image_id=image_id_str,
                    chunk_id=chunk.id,
                )
            except VisionLLMError as exc:
                failures += 1
                logger.warning("图片 %s caption 失败: %s", image_id_str, exc)

        if not captions:
            return self._mark_unprocessed(chunk)

        metadata = dict(chunk.metadata)
        metadata["image_captions"] = captions
        metadata["captioned_by"] = "vision_llm"
        if failures > 0:
            metadata["has_unprocessed_images"] = True
        else:
            metadata.pop("has_unprocessed_images", None)

        new_text = self._inject_captions_into_text(chunk.text, captions)
        end_offset = chunk.start_offset + len(new_text)
        captioned = Chunk(
            id=chunk.id,
            text=new_text,
            metadata=metadata,
            start_offset=chunk.start_offset,
            end_offset=end_offset,
            source_ref=chunk.source_ref,
        )
        captioned.validate()
        return captioned

    def _is_enabled(self) -> bool:
        vision = self._settings.vision_llm
        return vision is not None and vision.enabled

    def _load_prompt(self) -> str:
        if self._prompt_template is not None:
            return self._prompt_template
        try:
            self._prompt_template = load_image_captioning_prompt(self._prompt_path)
        except TransformError:
            self._prompt_template = _FALLBACK_PROMPT
        return self._prompt_template

    @staticmethod
    def _build_image_path_lookup(images: Any) -> dict[str, str]:
        """从 chunk.metadata.images 构建 image_id → path 映射。"""
        lookup: dict[str, str] = {}
        if not isinstance(images, list):
            return lookup
        for item in images:
            if isinstance(item, Mapping):
                image_id = str(item.get("id", "")).strip()
                path = str(item.get("path", "")).strip()
                if image_id and path:
                    lookup[image_id] = path
        return lookup

    @staticmethod
    def _inject_captions_into_text(text: str, captions: Mapping[str, str]) -> str:
        """将 ``[IMAGE: id]`` 占位符替换为 ``[图片描述: caption]`` 以参与检索。"""
        updated = text
        for image_id, caption in captions.items():
            placeholder = format_image_placeholder(image_id)
            caption_text = _CAPTION_TEMPLATE.format(caption=caption.strip())
            updated = updated.replace(placeholder, caption_text)
        return updated

    @staticmethod
    def _mark_unprocessed(chunk: Chunk) -> Chunk:
        """降级：保留 image_refs，不写 caption，标记未处理图片。"""
        if not chunk.metadata.get("image_refs"):
            return chunk
        metadata = dict(chunk.metadata)
        metadata["has_unprocessed_images"] = True
        metadata.pop("image_captions", None)
        metadata.pop("captioned_by", None)
        marked = Chunk(
            id=chunk.id,
            text=chunk.text,
            metadata=metadata,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref,
        )
        marked.validate()
        return marked

    @staticmethod
    def _record_trace(
        trace: Any | None,
        name: str,
        *,
        elapsed_ms: float,
        **details: Any,
    ) -> None:
        if isinstance(trace, TraceContext):
            trace.record_stage(name, elapsed_ms=elapsed_ms, **details)
