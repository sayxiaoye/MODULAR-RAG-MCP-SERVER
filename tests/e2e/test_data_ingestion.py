"""ingest.py E2E 测试：CLI 摄取与 data/db 产物验证（使用临时目录）。"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

from ingestion.pipeline import IngestionPipeline
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.loader.pdf_loader import PdfLoader
from libs.vector_store.vector_store_factory import VectorStoreFactory

REPO_ROOT = Path(__file__).resolve().parents[2]
INGEST_SCRIPT = REPO_ROOT / "scripts" / "ingest.py"


def _load_ingest_module() -> ModuleType:
    """动态加载 scripts/ingest.py，避免 scripts 包路径问题。"""
    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("ingest_cli", INGEST_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 ingest 脚本: {INGEST_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeEmbedding(BaseEmbedding):
    """固定维度向量，避免 E2E 依赖外部 Embedding 服务。"""

    def __init__(self, settings) -> None:
        self.settings = settings

    def embed(self, texts, trace=None) -> list[list[float]]:
        dim = self.settings.dimensions
        return [[0.1] * dim for _ in texts]


def _write_dummy_pdf(path: Path) -> Path:
    path.write_bytes(b"%PDF-1.4 ingest-e2e")
    return path


def _fake_resolve_loader(self, source_path: str, settings) -> PdfLoader:
    """注入 Markdown 转换器，避免 dummy PDF 依赖 MarkItDown。"""
    images_root = self._images_root or self._default_images_root(settings)
    return PdfLoader(
        images_root=images_root,
        markdown_converter=lambda p: "E2E ingest 测试文档内容 Azure BM25。",
    )


@pytest.fixture(autouse=True)
def _reset_factories() -> None:
    EmbeddingFactory.reset_constructor()
    VectorStoreFactory.reset_constructor()
    yield
    EmbeddingFactory.reset_constructor()
    VectorStoreFactory.reset_constructor()


@pytest.mark.e2e
class TestDataIngestion:
    """验证 ingest CLI 写入产物与跳过逻辑。"""

    def test_run_ingest_creates_db_artifacts(self, tmp_path: Path) -> None:
        """首次摄取应在临时 data 根下生成 Chroma/BM25/integrity 产物。"""
        pdf_dir = tmp_path / "pdfs"
        pdf_dir.mkdir()
        pdf_path = _write_dummy_pdf(pdf_dir / "sample_e2e.pdf")
        data_root = tmp_path / "data"
        collection = "e2e_col"

        EmbeddingFactory.set_constructor(lambda settings: FakeEmbedding(settings))

        with patch.object(IngestionPipeline, "_resolve_loader", _fake_resolve_loader):
            ingest = _load_ingest_module()
            code = ingest.run_ingest(
                path=str(pdf_path),
                collection=collection,
                force=True,
                data_root=str(data_root),
            )

        assert code == 0
        assert (data_root / "db" / "chroma").exists()
        assert (data_root / "db" / "bm25" / f"{collection}.json").is_file()
        assert (data_root / "db" / "ingestion_history.db").is_file()

    def test_run_ingest_skips_unchanged_file(self, tmp_path: Path) -> None:
        """未变更文件再次摄取应成功退出且不重复处理。"""
        pdf_path = _write_dummy_pdf(tmp_path / "repeat.pdf")
        data_root = tmp_path / "data"
        collection = "skip_col"

        EmbeddingFactory.set_constructor(lambda settings: FakeEmbedding(settings))

        with patch.object(IngestionPipeline, "_resolve_loader", _fake_resolve_loader):
            ingest = _load_ingest_module()
            first = ingest.run_ingest(
                path=str(pdf_path),
                collection=collection,
                force=True,
                data_root=str(data_root),
            )
            second = ingest.run_ingest(
                path=str(pdf_path),
                collection=collection,
                force=False,
                data_root=str(data_root),
            )

        assert first == 0
        assert second == 0

    def test_main_cli_parses_arguments(self, tmp_path: Path) -> None:
        """main() 应正确解析 --collection / --path / --force。"""
        pdf_path = _write_dummy_pdf(tmp_path / "cli.pdf")
        data_root = tmp_path / "data"

        EmbeddingFactory.set_constructor(lambda settings: FakeEmbedding(settings))

        with patch.object(IngestionPipeline, "_resolve_loader", _fake_resolve_loader):
            ingest = _load_ingest_module()
            code = ingest.main(
                [
                    "--path",
                    str(pdf_path),
                    "--collection",
                    "cli_col",
                    "--force",
                    "--data-root",
                    str(data_root),
                ]
            )

        assert code == 0
        assert (data_root / "db" / "bm25" / "cli_col.json").is_file()

    def test_collect_pdf_paths_from_directory(self, tmp_path: Path) -> None:
        """目录模式应收集顶层 PDF 列表。"""
        pdf_dir = tmp_path / "batch"
        pdf_dir.mkdir()
        _write_dummy_pdf(pdf_dir / "a.pdf")
        _write_dummy_pdf(pdf_dir / "b.pdf")

        ingest = _load_ingest_module()
        paths = ingest.collect_pdf_paths(str(pdf_dir))
        assert len(paths) == 2
        assert all(path.endswith(".pdf") for path in paths)

    def test_subprocess_cli_exit_success(self, tmp_path: Path) -> None:
        """子进程执行 scripts/ingest.py 应能正常退出（需 patch 在进程内不可用，用最小路径测试）。"""
        pdf_path = _write_dummy_pdf(tmp_path / "subprocess.pdf")
        data_root = tmp_path / "data"

        # 子进程无法注入 FakeEmbedding，仅验证参数解析与配置加载前的脚本可执行性
        empty_dir = tmp_path / "empty_dir"
        empty_dir.mkdir()
        result = subprocess.run(
            [
                sys.executable,
                str(INGEST_SCRIPT),
                "--path",
                str(empty_dir),
                "--collection",
                "none",
                "--data-root",
                str(data_root),
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        # 空目录无 PDF，应返回 0 并警告
        assert result.returncode == 0

        # 非 PDF 文件应返回 1
        bad = tmp_path / "not_pdf.txt"
        bad.write_text("text", encoding="utf-8")
        fail = subprocess.run(
            [
                sys.executable,
                str(INGEST_SCRIPT),
                "--path",
                str(bad),
                "--collection",
                "bad",
                "--data-root",
                str(data_root),
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
        )
        assert fail.returncode == 1
