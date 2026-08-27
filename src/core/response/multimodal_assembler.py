"""多模态内容组装：命中 chunk 含 image_refs 时读取本地图片并编码为 MCP ImageContent。"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.types import RetrievalResult
from observability.logger import get_logger

logger = get_logger("core.response.multimodal")

_MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


class MultimodalAssembler:
    """
    从 RetrievalResult.metadata.image_refs 组装 ImageContent。

    路径解析顺序：chunk metadata.images[].path → ImageStorage 索引。
    找不到文件时跳过该图，不阻断文本结果返回。
    """

    def __init__(self, image_storage: Any | None = None) -> None:
        self._image_storage = image_storage

    def assemble_image_contents(
        self,
        retrieval_results: Sequence[RetrievalResult],
    ) -> list[dict[str, Any]]:
        """
        按命中顺序去重后生成 MCP image content 列表。

        Returns:
            ``{"type": "image", "data": "<base64>", "mimeType": "..."}`` 列表。
        """
        contents: list[dict[str, Any]] = []
        seen: set[str] = set()

        for result in retrieval_results:
            for image_id in self._extract_image_refs(result.metadata):
                if image_id in seen:
                    continue
                seen.add(image_id)
                path = self._resolve_image_path(image_id, result.metadata)
                if path is None:
                    logger.warning("图片文件缺失，跳过 image_id=%s", image_id)
                    continue
                encoded = self._encode_file(path)
                if encoded is None:
                    continue
                contents.append(encoded)
        return contents

    def _extract_image_refs(self, metadata: Mapping[str, Any]) -> list[str]:
        """解析 image_refs，兼容 list 与 Chroma 序列化后的 JSON 字符串。"""
        raw = metadata.get("image_refs")
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                return [raw.strip()] if raw.strip() else []
        if not isinstance(raw, list):
            return []
        refs: list[str] = []
        for item in raw:
            if isinstance(item, str) and item.strip():
                refs.append(item.strip())
        return refs

    def _resolve_image_path(self, image_id: str, metadata: Mapping[str, Any]) -> Path | None:
        """优先使用 metadata.images 中的 path，再查 ImageStorage 索引。"""
        from_meta = self._path_from_metadata_images(image_id, metadata)
        if from_meta is not None:
            return from_meta

        storage = self._ensure_storage()
        if storage is None:
            return None
        try:
            return storage.get_path(image_id)
        except Exception as exc:  # noqa: BLE001 — 单图失败不影响整次检索返回
            logger.warning("ImageStorage 查询失败 image_id=%s: %s", image_id, exc)
            return None

    def _path_from_metadata_images(self, image_id: str, metadata: Mapping[str, Any]) -> Path | None:
        """从 chunk metadata.images 提取对应图片路径。"""
        images = metadata.get("images")
        if isinstance(images, str):
            try:
                images = json.loads(images)
            except json.JSONDecodeError:
                return None
        if not isinstance(images, list):
            return None
        for item in images:
            if not isinstance(item, Mapping):
                continue
            if str(item.get("id", "")).strip() != image_id:
                continue
            raw_path = item.get("path")
            if not isinstance(raw_path, str) or not raw_path.strip():
                return None
            path = Path(raw_path.strip())
            return path if path.is_file() else None
        return None

    def _ensure_storage(self) -> Any | None:
        """仅在需要按 image_id 查索引时才初始化 ImageStorage，避免测试误写默认库。"""
        if self._image_storage is not None:
            return self._image_storage
        try:
            from ingestion.storage.image_storage import ImageStorage

            self._image_storage = ImageStorage()
        except Exception as exc:  # noqa: BLE001
            logger.warning("无法初始化 ImageStorage: %s", exc)
            return None
        return self._image_storage

    @staticmethod
    def _encode_file(path: Path) -> dict[str, Any] | None:
        """读取文件并编码为 MCP ImageContent。"""
        try:
            data = path.read_bytes()
        except OSError as exc:
            logger.warning("读取图片失败 path=%s: %s", path, exc)
            return None
        if not data:
            return None
        mime = _MIME_BY_SUFFIX.get(path.suffix.lower(), "image/png")
        return {
            "type": "image",
            "data": base64.b64encode(data).decode("ascii"),
            "mimeType": mime,
        }
