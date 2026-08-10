"""Core 层公共导出：配置与全链路数据契约。"""

from core.settings import Settings, load_settings
from core.types import (
    Chunk,
    ChunkRecord,
    Document,
    ImageMetadata,
    TypesError,
    format_image_placeholder,
)

__all__ = [
    "Chunk",
    "ChunkRecord",
    "Document",
    "ImageMetadata",
    "Settings",
    "TypesError",
    "format_image_placeholder",
    "load_settings",
]
