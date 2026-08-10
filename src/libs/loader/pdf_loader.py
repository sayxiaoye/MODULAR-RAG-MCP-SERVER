"""PDF Loader：将 PDF 转为 Markdown Document，并支持图片占位符契约。"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from core.settings import REPO_ROOT, resolve_path
from core.types import Document, ImageMetadata, format_image_placeholder
from libs.loader.base_loader import BaseLoader, LoaderError

logger = logging.getLogger(__name__)

DEFAULT_IMAGES_ROOT = REPO_ROOT / "data" / "images"


@dataclass(frozen=True)
class ExtractedImage:
    """PDF 内提取的单张图片，供占位符与落盘使用。"""

    id: str
    data: bytes
    page: int | None = None
    position: dict[str, object] | None = None


MarkdownConverter = Callable[[Path], str]
ImageExtractor = Callable[[Path, str], Sequence[ExtractedImage]]


def _compute_sha256(path: Path) -> str:
    """计算文件 SHA256，用作 doc_hash / Document.id 的稳定来源。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _default_markdown_converter(path: Path) -> str:
    """默认使用 MarkItDown 将 PDF 转为 Markdown 文本。"""
    try:
        from markitdown import MarkItDown
    except ImportError as exc:
        raise LoaderError(
            "PdfLoader 默认转换需要 markitdown，请安装：pip install -e \".[ingestion]\""
        ) from exc
    result = MarkItDown().convert(str(path))
    text = getattr(result, "text_content", None)
    if not isinstance(text, str):
        raise LoaderError("[pdf] MarkItDown 未返回 text_content")
    return text


def _default_image_extractor(path: Path, doc_hash: str) -> list[ExtractedImage]:
    """
    默认图片提取占位：壳子版本不绑定具体 PDF 图像库，返回空列表。

    后续可替换为 PyMuPDF / pdfimages 等实现，而不影响 BaseLoader 契约。
    """
    return []


class PdfLoader(BaseLoader):
    """PDF → Document 加载器，遵循 C1 图片占位符与 metadata.images 契约。"""

    def __init__(
        self,
        images_root: str | Path | None = None,
        markdown_converter: MarkdownConverter | None = None,
        image_extractor: ImageExtractor | None = None,
    ) -> None:
        root = Path(images_root) if images_root is not None else DEFAULT_IMAGES_ROOT
        if not root.is_absolute():
            root = resolve_path(root)
        self.images_root = root
        self._markdown_converter = markdown_converter or _default_markdown_converter
        self._image_extractor = image_extractor or _default_image_extractor

    def load(self, path: str) -> Document:
        pdf_path = Path(path)
        if not pdf_path.is_file():
            raise LoaderError(f"[pdf] 文件不存在: {path}")
        if pdf_path.suffix.lower() != ".pdf":
            raise LoaderError(f"[pdf] 仅支持 PDF 文件: {path}")

        doc_hash = _compute_sha256(pdf_path)
        doc_id = doc_hash

        try:
            markdown_text = self._markdown_converter(pdf_path).strip()
        except LoaderError:
            raise
        except Exception as exc:
            raise LoaderError(f"[pdf] 文本解析失败: {exc}") from exc

        images_meta: list[ImageMetadata] = []
        try:
            extracted = list(self._image_extractor(pdf_path, doc_hash))
            markdown_text, images_meta = self._merge_images_into_text(
                markdown_text,
                extracted,
                doc_hash,
            )
        except Exception as exc:
            # 图片提取失败不阻塞文本摄取，仅记录警告
            logger.warning("[pdf] 图片提取失败，将继续仅文本解析: %s", exc)

        metadata: dict[str, object] = {
            "source_path": str(pdf_path.resolve()),
            "doc_type": "pdf",
            "doc_hash": doc_hash,
        }
        if images_meta:
            metadata["images"] = images_meta

        document = Document(id=doc_id, text=markdown_text, metadata=metadata)
        document.validate()
        return document

    def _merge_images_into_text(
        self,
        markdown_text: str,
        extracted: Sequence[ExtractedImage],
        doc_hash: str,
    ) -> tuple[str, list[ImageMetadata]]:
        """保存图片文件并在文本末尾追加占位符（壳子阶段采用稳定追加策略）。"""
        if not extracted:
            return markdown_text, []

        output_dir = self.images_root / doc_hash
        output_dir.mkdir(parents=True, exist_ok=True)

        merged_text = markdown_text
        if merged_text and not merged_text.endswith("\n"):
            merged_text += "\n"

        images_meta: list[ImageMetadata] = []
        for index, image in enumerate(extracted):
            image_id = image.id or f"{doc_hash}_{index}"
            placeholder = format_image_placeholder(image_id)
            offset = len(merged_text)

            image_path = output_dir / f"{image_id}.png"
            image_path.write_bytes(image.data)

            merged_text += placeholder + "\n"
            images_meta.append(
                ImageMetadata(
                    id=image_id,
                    path=str(image_path),
                    page=image.page,
                    text_offset=offset,
                    text_length=len(placeholder),
                    position=image.position,
                )
            )

        return merged_text.strip(), images_meta
