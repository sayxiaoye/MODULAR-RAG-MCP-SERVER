"""Ingestion 管理页纯函数测试：进度映射、路径收集、上传落盘。"""

from __future__ import annotations

from pathlib import Path

import pytest

from observability.dashboard.pages.ingestion_manager import (
    collect_pdf_paths,
    progress_fraction,
    save_uploaded_files,
)


class _FakeUpload:
    """模拟 Streamlit UploadedFile：仅提供 name 与 getvalue。"""

    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


@pytest.mark.unit
class TestIngestionManagerHelpers:
    """验证进度条映射与文件收集，不依赖 Streamlit 运行时。"""

    def test_progress_fraction_covers_canonical_stages(self) -> None:
        """规范阶段应按顺序推进到 1.0；未知阶段为 0。"""
        assert progress_fraction("integrity", 1, 1) == pytest.approx(1 / 6)
        assert progress_fraction("load", 1, 1) == pytest.approx(2 / 6)
        assert progress_fraction("upsert", 1, 1) == pytest.approx(1.0)
        assert progress_fraction("unknown", 1, 1) == 0.0
        assert progress_fraction("transform", 0, 3) == pytest.approx(3 / 6)
        assert progress_fraction("transform", 3, 3) == pytest.approx(4 / 6)

    def test_collect_pdf_paths_file_and_directory(self, tmp_path: Path) -> None:
        """单文件返回其自身；目录只收集顶层 PDF。"""
        pdf = tmp_path / "a.pdf"
        pdf.write_bytes(b"%PDF")
        (tmp_path / "note.txt").write_text("skip")
        nested = tmp_path / "nested"
        nested.mkdir()
        (nested / "b.pdf").write_bytes(b"%PDF")

        assert collect_pdf_paths(str(pdf)) == [pdf.resolve()]
        listed = collect_pdf_paths(str(tmp_path))
        assert listed == [pdf.resolve()]

    def test_collect_pdf_paths_rejects_missing_and_non_pdf(self, tmp_path: Path) -> None:
        """缺失路径与非 PDF 应抛出明确错误。"""
        with pytest.raises(FileNotFoundError):
            collect_pdf_paths(str(tmp_path / "missing.pdf"))
        txt = tmp_path / "a.txt"
        txt.write_text("no")
        with pytest.raises(ValueError, match="仅支持 PDF"):
            collect_pdf_paths(str(txt))

    def test_save_uploaded_files_writes_bytes(self, tmp_path: Path) -> None:
        """上传对象应写入 staging 目录并保留文件名。"""
        uploaded = [_FakeUpload("demo.pdf", b"%PDF-fake")]
        saved = save_uploaded_files(uploaded, staging_dir=tmp_path)
        assert len(saved) == 1
        assert saved[0].name == "demo.pdf"
        assert saved[0].read_bytes() == b"%PDF-fake"
