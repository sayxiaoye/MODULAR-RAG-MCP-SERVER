"""Document → Chunk 适配层：在 libs.splitter 之上封装业务对象转换。"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Mapping

from core.settings import Settings
from core.types import Chunk, Document, ImageMetadata
from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.splitter_factory import SplitterFactory

# 与 C1 format_image_placeholder 保持一致，用于扫描 chunk 内图片引用
_IMAGE_PLACEHOLDER_PATTERN = re.compile(r"\[IMAGE:\s*([^\]]+?)\s*\]")


class DocumentChunker:
    """将 Document 切分为带溯源与元数据的 Chunk 列表。"""

    def __init__(
        self,
        settings: Settings,
        splitter: BaseSplitter | None = None,
    ) -> None:
        self._settings = settings
        # 测试可注入 FakeSplitter，生产环境走 SplitterFactory
        self._splitter = splitter or SplitterFactory.create(settings)

    def split_document(self, document: Document) -> list[Chunk]:
        """
        完整 Document → Chunks 转换流程。

        Args:
            document: Loader 产出的标准文档对象。

        Returns:
            符合 core.types 契约的 Chunk 列表。
        """
        document.validate()
        chunk_texts = self._splitter.split_text(document.text)
        offset_pairs = self._compute_offsets(document.text, chunk_texts)

        chunks: list[Chunk] = []
        for index, (chunk_text, (start_offset, end_offset)) in enumerate(
            zip(chunk_texts, offset_pairs, strict=True)
        ):
            chunk_id = self._generate_chunk_id(document.id, index, chunk_text)
            metadata = self._inherit_metadata(document, index, chunk_text)
            chunk = Chunk(
                id=chunk_id,
                text=chunk_text,
                metadata=metadata,
                start_offset=start_offset,
                end_offset=end_offset,
                source_ref=document.id,
            )
            chunk.validate()
            chunks.append(chunk)
        return chunks

    def _generate_chunk_id(self, doc_id: str, index: int, text: str) -> str:
        """生成确定性 Chunk ID：``{doc_id}_{index:04d}_{hash_8chars}``。"""
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
        return f"{doc_id}_{index:04d}_{content_hash}"

    def _inherit_metadata(
        self,
        document: Document,
        chunk_index: int,
        chunk_text: str,
    ) -> dict[str, Any]:
        """
        继承 Document 元数据并追加 chunk_index；按占位符分发图片子集。

        无 ``[IMAGE: id]`` 占位符的 chunk 不包含 ``images`` 字段。
        """
        # 复制文档级元数据，但不在 chunk 层整体继承 images
        metadata: dict[str, Any] = {
            key: value for key, value in document.metadata.items() if key != "images"
        }
        metadata["chunk_index"] = chunk_index

        image_refs = self._extract_image_refs(chunk_text)
        if not image_refs:
            return metadata

        metadata["image_refs"] = image_refs
        doc_images = document.metadata.get("images")
        if not doc_images:
            return metadata

        image_lookup = self._build_image_lookup(doc_images)
        filtered_images = [image_lookup[ref] for ref in image_refs if ref in image_lookup]
        if filtered_images:
            metadata["images"] = filtered_images
        return metadata

    @staticmethod
    def _extract_image_refs(chunk_text: str) -> list[str]:
        """按出现顺序提取 chunk 文本中的 image_id 列表。"""
        return [
            match.group(1).strip()
            for match in _IMAGE_PLACEHOLDER_PATTERN.finditer(chunk_text)
        ]

    @staticmethod
    def _build_image_lookup(
        images: list[Any],
    ) -> dict[str, dict[str, Any]]:
        """将 Document.metadata.images 规范化为 id → dict 映射。"""
        lookup: dict[str, dict[str, Any]] = {}
        for item in images:
            if isinstance(item, ImageMetadata):
                lookup[item.id] = item.to_dict()
            elif isinstance(item, Mapping):
                image_id = str(item.get("id", "")).strip()
                if image_id:
                    lookup[image_id] = dict(item)
        return lookup

    @staticmethod
    def _compute_offsets(
        document_text: str,
        chunk_texts: list[str],
    ) -> list[tuple[int, int]]:
        """在原文中顺序定位各 chunk 的起止偏移（兼容 overlap 切分）。"""
        offsets: list[tuple[int, int]] = []
        search_from = 0
        for chunk_text in chunk_texts:
            if not chunk_text:
                offsets.append((search_from, search_from))
                continue

            start = document_text.find(chunk_text, search_from)
            if start == -1:
                # overlap 场景下回退全局搜索，避免定位失败
                start = document_text.find(chunk_text)
            if start == -1:
                start = search_from

            end = start + len(chunk_text)
            offsets.append((start, end))
            # 下一 chunk 可能因 overlap 在当前片段之前开始，故仅推进 1 而非 end
            search_from = max(search_from, start + 1)
        return offsets
