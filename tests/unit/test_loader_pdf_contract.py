"""PdfLoader 契约测试：验证 Document 产出、图片占位符与降级行为。"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.types import format_image_placeholder
from libs.loader.base_loader import BaseLoader, LoaderError
from libs.loader.pdf_loader import ExtractedImage, PdfLoader


def _write_dummy_pdf(path: Path, content: bytes = b"%PDF-1.4 dummy") -> Path:
    """写入最小 PDF 占位文件，实际解析由注入的 converter 完成。"""
    path.write_bytes(content)
    return path


@pytest.mark.unit
class TestPdfLoaderContract:
    """验证 PdfLoader 对 C1 Document 契约的满足情况。"""

    def test_load_produces_document_with_source_path(self, tmp_path: Path) -> None:
        """纯文本 PDF（无图）应产出含 source_path 的 Document。"""
        pdf_path = _write_dummy_pdf(tmp_path / "simple.pdf")
        loader = PdfLoader(
            images_root=tmp_path / "images",
            markdown_converter=lambda p: "# Sample\n\nHello from PDF.",
        )
        document = loader.load(str(pdf_path))

        assert document.id
        assert "Hello from PDF" in document.text
        assert document.metadata["source_path"] == str(pdf_path.resolve())
        assert document.metadata.get("images") is None

    def test_load_with_images_inserts_placeholders_and_saves_files(self, tmp_path: Path) -> None:
        """带图 PDF 应在 text 中插入占位符并落盘到 data/images/{doc_hash}/。"""
        pdf_path = _write_dummy_pdf(tmp_path / "with_images.pdf")
        image_id = "img_page1_0"
        images_root = tmp_path / "images"

        def image_extractor(path: Path, doc_hash: str) -> list[ExtractedImage]:
            return [
                ExtractedImage(
                    id=image_id,
                    data=b"fake-png-bytes",
                    page=1,
                    position={"x": 0, "y": 0},
                )
            ]

        loader = PdfLoader(
            images_root=images_root,
            markdown_converter=lambda p: "正文段落",
            image_extractor=image_extractor,
        )
        document = loader.load(str(pdf_path))

        placeholder = format_image_placeholder(image_id)
        assert placeholder in document.text
        images = document.metadata["images"]
        assert len(images) == 1
        assert images[0].id == image_id

        doc_hash = document.metadata["doc_hash"]
        saved = images_root / doc_hash / f"{image_id}.png"
        assert saved.is_file()
        assert saved.read_bytes() == b"fake-png-bytes"

    def test_image_extractor_failure_does_not_block_text(self, tmp_path: Path) -> None:
        """图片提取失败时仍应返回可解析的文本 Document。"""
        pdf_path = _write_dummy_pdf(tmp_path / "broken_images.pdf")

        def failing_extractor(path: Path, doc_hash: str) -> list[ExtractedImage]:
            raise RuntimeError("mock image extraction failure")

        loader = PdfLoader(
            images_root=tmp_path / "images",
            markdown_converter=lambda p: "仅文本内容",
            image_extractor=failing_extractor,
        )
        document = loader.load(str(pdf_path))

        assert document.text == "仅文本内容"
        assert document.metadata.get("images") is None

    def test_missing_file_raises_loader_error(self, tmp_path: Path) -> None:
        """文件不存在时应抛出 LoaderError。"""
        loader = PdfLoader(markdown_converter=lambda p: "x")
        with pytest.raises(LoaderError, match="文件不存在"):
            loader.load(str(tmp_path / "missing.pdf"))

    def test_base_loader_is_abstract(self) -> None:
        """BaseLoader 子类必须实现 load。"""
        assert issubclass(PdfLoader, BaseLoader)

        class IncompleteLoader(BaseLoader):
            pass

        with pytest.raises(TypeError):
            IncompleteLoader()
