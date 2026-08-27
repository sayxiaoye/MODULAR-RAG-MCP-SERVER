"""Ingestion Pipeline 集成测试：完整链路写入 Chroma、BM25 与图片索引。"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.settings import Settings, VectorStoreSettings, load_settings
from core.trace.trace_context import TraceContext
from ingestion.embedding.batch_processor import BatchProcessor
from ingestion.embedding.dense_encoder import DenseEncoder
from ingestion.pipeline import IngestionPipeline, IngestionPipelineError
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.loader.file_integrity import SQLiteIntegrityChecker
from libs.loader.pdf_loader import ExtractedImage, PdfLoader
from libs.vector_store.chroma_store import ChromaStore
from libs.vector_store.vector_store_factory import VectorStoreFactory


class FakeEmbedding(BaseEmbedding):
    """固定维度向量，避免集成测试依赖外部 Embedding 服务。"""

    def __init__(self, settings) -> None:
        self.settings = settings

    def embed(self, texts, trace=None) -> list[list[float]]:
        dim = self.settings.dimensions
        return [[0.1] * dim for _ in texts]


def _write_dummy_pdf(path: Path) -> Path:
    path.write_bytes(b"%PDF-1.4 pipeline-test")
    return path


def _settings_with_paths(
    tmp_path: Path,
    collection: str,
    batch_size: int = 100,
) -> Settings:
    base = load_settings()
    ingestion = base.ingestion
    assert ingestion is not None
    from core.settings import IngestionSettings

    return Settings(
        llm=base.llm,
        embedding=base.embedding,
        vector_store=VectorStoreSettings(
            provider="chroma",
            persist_directory=str(tmp_path / "chroma"),
            collection_name=collection,
        ),
        retrieval=base.retrieval,
        rerank=base.rerank,
        evaluation=base.evaluation,
        observability=base.observability,
        ingestion=IngestionSettings(
            chunk_size=ingestion.chunk_size,
            chunk_overlap=ingestion.chunk_overlap,
            splitter=ingestion.splitter,
            batch_size=batch_size,
            chunk_refiner=ingestion.chunk_refiner,
            metadata_enricher=ingestion.metadata_enricher,
        ),
        vision_llm=base.vision_llm,
    )


@pytest.fixture(autouse=True)
def _reset_embedding_factory() -> None:
    EmbeddingFactory.reset_constructor()
    VectorStoreFactory.reset_constructor()
    yield
    EmbeddingFactory.reset_constructor()
    VectorStoreFactory.reset_constructor()


@pytest.fixture
def pipeline_bundle(tmp_path: Path) -> dict:
    """构造可注入 Fake 后端的 Pipeline 与临时存储目录。"""
    collection = "pipeline_test"
    settings = _settings_with_paths(tmp_path, collection)
    chroma_dir = tmp_path / "chroma"
    bm25_dir = tmp_path / "bm25"
    images_root = tmp_path / "images"
    integrity_db = tmp_path / "ingestion_history.db"
    image_db = tmp_path / "image_index.db"

    EmbeddingFactory.set_constructor(lambda embedding_settings: FakeEmbedding(embedding_settings))

    image_id = "img_pipeline_1"
    markdown = "Azure OpenAI 配置指南与 BM25 检索实践。"

    def markdown_converter(path: Path) -> str:
        return markdown

    def image_extractor(path: Path, doc_hash: str) -> list[ExtractedImage]:
        return [
            ExtractedImage(
                id=image_id,
                data=b"fake-image-bytes",
                page=1,
                position={"x": 0, "y": 0},
            )
        ]

    pdf_path = _write_dummy_pdf(tmp_path / "sample_pipeline.pdf")
    loader = PdfLoader(
        images_root=images_root,
        markdown_converter=markdown_converter,
        image_extractor=image_extractor,
    )

    pipeline = IngestionPipeline(
        settings=settings,
        integrity_checker=SQLiteIntegrityChecker(db_path=integrity_db),
        loader=loader,
        batch_processor=BatchProcessor(
            settings,
            dense_encoder=DenseEncoder(settings, embedding=FakeEmbedding(settings.embedding)),
            batch_size=settings.ingestion.batch_size,
        ),
        bm25_indexer=BM25Indexer(collection=collection, index_root=bm25_dir),
        image_storage=ImageStorage(images_root=images_root, db_path=image_db),
        images_root=images_root,
        bm25_root=bm25_dir,
        chroma_persist_directory=chroma_dir,
    )

    return {
        "pipeline": pipeline,
        "pdf_path": str(pdf_path),
        "collection": collection,
        "chroma_dir": chroma_dir,
        "bm25_dir": bm25_dir,
        "images_root": images_root,
        "image_db": image_db,
        "image_id": image_id,
        "settings": settings,
        "markdown": markdown,
    }


@pytest.mark.integration
class TestIngestionPipeline:
    """验证 MVP Pipeline 产物与跳过/失败行为。"""

    def test_full_pipeline_writes_chroma_bm25_and_images(self, pipeline_bundle: dict) -> None:
        """完整链路应写入向量库、BM25 索引与图片存储。"""
        pipeline = pipeline_bundle["pipeline"]
        collection = pipeline_bundle["collection"]
        result = pipeline.run(pipeline_bundle["pdf_path"], collection=collection, force=True)

        assert result.skipped is False
        assert result.chunk_count > 0
        assert result.image_count == 1
        assert len(result.chunk_ids) == result.chunk_count

        chroma = ChromaStore(
            VectorStoreSettings(
                provider="chroma",
                persist_directory=str(pipeline_bundle["chroma_dir"]),
                collection_name=collection,
            )
        )
        query_vector = [0.1] * pipeline_bundle["settings"].embedding.dimensions
        hits = chroma.query(query_vector, top_k=3)
        assert len(hits) >= 1
        assert hits[0]["metadata"]["collection"] == collection

        bm25 = BM25Indexer(collection=collection, index_root=pipeline_bundle["bm25_dir"])
        bm25.load()
        sparse_hits = bm25.query("Azure BM25", top_k=3)
        assert sparse_hits

        image_storage = ImageStorage(
            images_root=pipeline_bundle["images_root"],
            db_path=pipeline_bundle["image_db"],
        )
        image_path = image_storage.get_path(pipeline_bundle["image_id"])
        assert image_path is not None
        assert image_path.read_bytes() == b"fake-image-bytes"
        assert (pipeline_bundle["bm25_dir"] / f"{collection}.json").is_file()

    def test_second_run_skips_without_force(self, pipeline_bundle: dict) -> None:
        """未变更文件再次摄取应被 integrity 跳过。"""
        pipeline = pipeline_bundle["pipeline"]
        path = pipeline_bundle["pdf_path"]
        collection = pipeline_bundle["collection"]

        first = pipeline.run(path, collection=collection, force=True)
        second = pipeline.run(path, collection=collection, force=False)

        assert first.skipped is False
        assert second.skipped is True
        assert second.chunk_count == 0

    def test_on_progress_receives_stage_updates(self, pipeline_bundle: dict) -> None:
        """进度回调应收到各阶段事件。"""
        stages: list[str] = []
        pipeline = pipeline_bundle["pipeline"]

        pipeline.run(
            pipeline_bundle["pdf_path"],
            collection=pipeline_bundle["collection"],
            force=True,
            on_progress=lambda stage, current, total: stages.append(stage),
        )

        assert "integrity" in stages
        assert "load" in stages
        assert "split" in stages
        assert "transform" in stages
        assert "encode" in stages
        assert "store" in stages

    def test_ingestion_trace_records_canonical_stages(self, pipeline_bundle: dict) -> None:
        """F4：一次摄取应包含 load/split/transform/embed/upsert，且带 elapsed_ms 与 method。"""
        trace = TraceContext(trace_type="ingestion")
        pipeline = pipeline_bundle["pipeline"]
        settings = pipeline_bundle["settings"]

        pipeline.run(
            pipeline_bundle["pdf_path"],
            collection=pipeline_bundle["collection"],
            force=True,
            trace=trace,
        )

        payload = trace.to_dict()
        assert payload["trace_type"] == "ingestion"
        assert trace.is_finished is True

        by_name = {stage["name"]: stage for stage in payload["stages"]}
        expected = {
            "load": "markitdown",
            "split": settings.ingestion.splitter,
            "transform": "sequential",
            "embed": settings.embedding.provider,
            "upsert": settings.vector_store.provider,
        }
        for name, method in expected.items():
            stage = by_name[name]
            assert "elapsed_ms" in stage
            assert isinstance(stage["elapsed_ms"], (int, float))
            assert stage["elapsed_ms"] >= 0
            assert stage["method"] == method

    def test_load_failure_raises_clear_pipeline_error(self, tmp_path: Path) -> None:
        """Loader 失败应抛出带阶段名的 IngestionPipelineError。"""
        settings = _settings_with_paths(tmp_path, "err_col")
        loader = PdfLoader(
            images_root=tmp_path / "images",
            markdown_converter=lambda p: (_ for _ in ()).throw(RuntimeError("mock load failure")),
        )
        pipeline = IngestionPipeline(
            settings=settings,
            integrity_checker=SQLiteIntegrityChecker(db_path=tmp_path / "db.db"),
            loader=loader,
            chroma_persist_directory=tmp_path / "chroma",
            bm25_root=tmp_path / "bm25",
        )
        pdf_path = _write_dummy_pdf(tmp_path / "bad.pdf")

        with pytest.raises(IngestionPipelineError) as exc_info:
            pipeline.run(str(pdf_path), collection="err_col", force=True)

        assert exc_info.value.stage == "load"
        assert "mock load failure" in str(exc_info.value)

    def test_fixture_sample_txt_path_regression_via_injected_loader(self, tmp_path: Path) -> None:
        """辅助回归：使用 fixtures/sample.txt 内容通过注入 Loader 跑通 pipeline。"""
        fixtures_root = Path(__file__).resolve().parents[1] / "fixtures" / "sample_documents"
        sample_txt = fixtures_root / "sample.txt"
        if not sample_txt.is_file():
            pytest.skip("fixtures/sample.txt 不存在")

        text = sample_txt.read_text(encoding="utf-8")
        collection = "txt_regression"
        settings = _settings_with_paths(tmp_path, collection)
        EmbeddingFactory.set_constructor(lambda s: FakeEmbedding(s))

        from core.types import Document

        class _TextLoader:
            def load(self, path: str) -> Document:
                return Document(
                    id="txt_doc",
                    text=text,
                    metadata={
                        "source_path": str(Path(path).resolve()),
                        "doc_type": "txt",
                        "doc_hash": "txt_hash",
                    },
                )

        pipeline = IngestionPipeline(
            settings=settings,
            integrity_checker=SQLiteIntegrityChecker(db_path=tmp_path / "hist.db"),
            loader=_TextLoader(),
            chroma_persist_directory=tmp_path / "chroma",
            bm25_root=tmp_path / "bm25",
        )
        fake_pdf = _write_dummy_pdf(tmp_path / "sample.txt.pdf")
        result = pipeline.run(str(fake_pdf), collection=collection, force=True)
        assert result.skipped is False
        assert result.chunk_count > 0
