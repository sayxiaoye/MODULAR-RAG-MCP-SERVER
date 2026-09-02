"""召回回归 E2E：摄取已知文档后跑 EvalRunner，断言 hit@k 不低于阈值。"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from core.settings import IngestionSettings, RetrievalSettings, Settings, VectorStoreSettings, load_settings
from ingestion.embedding.batch_processor import BatchProcessor
from ingestion.embedding.dense_encoder import DenseEncoder
from ingestion.pipeline import IngestionPipeline
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.evaluator.custom_evaluator import CustomEvaluator
from libs.loader.file_integrity import SQLiteIntegrityChecker
from libs.loader.pdf_loader import PdfLoader
from libs.vector_store.vector_store_factory import VectorStoreFactory
from observability.evaluation.eval_runner import EvalRunner, load_golden_test_set

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "golden_test_set.json"

# 回归阈值写死在测试里：Top-K 内至少 80% 的 query 命中 golden chunk
MIN_HIT_AT_K = 0.8
HIT_AT_K = 5
COLLECTION = "recall_e2e"

_CORPUS: dict[str, str] = {
    "config_guide.pdf": (
        "如何配置 Azure OpenAI：在 Azure 门户创建资源，填写 endpoint 与 api_key，"
        "然后在 settings.yaml 中指定 llm.provider 为 azure。"
    ),
    "retrieval_guide.pdf": (
        "Hybrid Search 如何融合 Dense 与 Sparse：并行召回后使用 RRF 倒数排名融合，"
        "公式为 1/(k+rank_dense)+1/(k+rank_sparse)。"
    ),
    "bm25_guide.pdf": (
        "BM25 如何用 jieba 做中文分词：对查询与文档切词后计算 IDF 与词频，"
        "写入倒排索引再按分数排序。"
    ),
}


class FakeEmbedding(BaseEmbedding):
    """固定维度向量，避免 E2E 依赖外部 Embedding / llama-server。"""

    def __init__(self, settings) -> None:
        self.settings = settings

    def embed(self, texts, trace=None) -> list[list[float]]:
        dim = self.settings.dimensions
        return [[0.1] * dim for _ in texts]


def _write_dummy_pdf(path: Path) -> Path:
    path.write_bytes(b"%PDF-1.4 recall-e2e")
    return path


def _settings_for_recall(tmp_path: Path) -> Settings:
    """指向临时 Chroma/BM25，并把 fusion_top_k 收成 hit@k。"""
    base = load_settings()
    ingestion = base.ingestion
    assert ingestion is not None
    return Settings(
        llm=base.llm,
        embedding=base.embedding,
        vector_store=VectorStoreSettings(
            provider="chroma",
            persist_directory=str(tmp_path / "chroma"),
            collection_name=COLLECTION,
        ),
        retrieval=RetrievalSettings(
            dense_top_k=base.retrieval.dense_top_k,
            sparse_top_k=base.retrieval.sparse_top_k,
            fusion_top_k=HIT_AT_K,
            rrf_k=base.retrieval.rrf_k,
        ),
        rerank=base.rerank,
        evaluation=replace(base.evaluation, provider="custom", backends=["custom"]),
        observability=base.observability,
        ingestion=IngestionSettings(
            chunk_size=ingestion.chunk_size,
            chunk_overlap=ingestion.chunk_overlap,
            splitter=ingestion.splitter,
            batch_size=ingestion.batch_size,
            chunk_refiner=ingestion.chunk_refiner,
            metadata_enricher=ingestion.metadata_enricher,
        ),
        vision_llm=base.vision_llm,
    )


def _build_pipeline(tmp_path: Path, settings: Settings) -> IngestionPipeline:
    """与摄取集成测试相同：FakeEmbedding + 按文件名注入正文。"""
    chroma_dir = tmp_path / "chroma"
    bm25_dir = tmp_path / "bm25"
    images_root = tmp_path / "images"

    def markdown_converter(path: Path) -> str:
        return _CORPUS.get(path.name, "placeholder")

    loader = PdfLoader(
        images_root=images_root,
        markdown_converter=markdown_converter,
    )
    return IngestionPipeline(
        settings=settings,
        integrity_checker=SQLiteIntegrityChecker(db_path=tmp_path / "ingestion_history.db"),
        loader=loader,
        batch_processor=BatchProcessor(
            settings,
            dense_encoder=DenseEncoder(settings, embedding=FakeEmbedding(settings.embedding)),
            batch_size=settings.ingestion.batch_size if settings.ingestion else 100,
        ),
        bm25_indexer=BM25Indexer(collection=COLLECTION, index_root=bm25_dir),
        image_storage=ImageStorage(images_root=images_root, db_path=tmp_path / "image_index.db"),
        images_root=images_root,
        bm25_root=bm25_dir,
        chroma_persist_directory=chroma_dir,
    )


def _build_hybrid_search(settings: Settings, bm25_dir: Path):
    """用同一临时库组装 HybridSearch，供 EvalRunner 检索。"""
    from core.query_engine.dense_retriever import DenseRetriever
    from core.query_engine.hybrid_search import HybridSearch
    from core.query_engine.sparse_retriever import SparseRetriever

    store = VectorStoreFactory.create(settings)
    indexer = BM25Indexer(collection=COLLECTION, index_root=bm25_dir)
    indexer.load()
    return HybridSearch(
        settings,
        dense_retriever=DenseRetriever(
            settings,
            embedding=FakeEmbedding(settings.embedding),
            vector_store=store,
        ),
        sparse_retriever=SparseRetriever(
            settings,
            bm25_indexer=indexer,
            vector_store=store,
        ),
    )


@pytest.fixture(autouse=True)
def _reset_factories() -> None:
    EmbeddingFactory.reset_constructor()
    VectorStoreFactory.reset_constructor()
    yield
    EmbeddingFactory.reset_constructor()
    VectorStoreFactory.reset_constructor()


@pytest.mark.e2e
class TestRecallRegression:
    """黄金集规模与 hit@k 回归阈值。"""

    def test_golden_fixture_has_enough_cases(self) -> None:
        """仓库 golden_test_set 应补齐到至少 4 条，便于后续扩展。"""
        cases = load_golden_test_set(GOLDEN_FIXTURE)
        assert len(cases) >= 4
        queries = {item["query"] for item in cases}
        assert "如何配置 Azure OpenAI？" in queries
        assert "BM25 如何用 jieba 做中文分词？" in queries

    def test_hit_at_k_meets_threshold(self, tmp_path: Path) -> None:
        """摄取区分度文档后，EvalRunner 的 hit_rate 应 >= MIN_HIT_AT_K。"""
        EmbeddingFactory.set_constructor(lambda embedding_settings: FakeEmbedding(embedding_settings))
        settings = _settings_for_recall(tmp_path)
        pipeline = _build_pipeline(tmp_path, settings)

        golden_cases: list[dict[str, object]] = []
        for filename, text in _CORPUS.items():
            pdf = _write_dummy_pdf(tmp_path / filename)
            result = pipeline.run(str(pdf), collection=COLLECTION, force=True)
            assert result.skipped is False
            assert result.chunk_ids, f"{filename} 未写入 chunk"
            # 用正文前若干字作为 query，保证 BM25 能命中该文档
            query = text.split("：")[0] + "？"
            golden_cases.append(
                {
                    "query": query,
                    "expected_chunk_ids": list(result.chunk_ids),
                    "expected_sources": [filename],
                }
            )

        golden_path = tmp_path / "golden_test_set.json"
        golden_path.write_text(
            json.dumps({"test_cases": golden_cases}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        search = _build_hybrid_search(settings, tmp_path / "bm25")
        report = EvalRunner(settings, search, CustomEvaluator()).run(golden_path)

        assert report.case_count == len(golden_cases)
        assert report.hit_rate >= MIN_HIT_AT_K, (
            f"hit@{HIT_AT_K}={report.hit_rate:.4f} 低于阈值 {MIN_HIT_AT_K}；"
            f"明细={[item.to_dict() for item in report.cases]}"
        )
        assert report.mrr > 0.0
