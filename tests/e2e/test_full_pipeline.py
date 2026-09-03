"""全链路 E2E：ingest CLI → query CLI → Dashboard 追踪 → evaluate CLI。

对应 spec I5：用临时 data 根与 FakeEmbedding 走通验收命令，避免占用 llama-server。
"""

from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from core.query_engine.query_pipeline import settings_for_query
from core.settings import load_settings
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.evaluator.custom_evaluator import CustomEvaluator
from libs.vector_store.vector_store_factory import VectorStoreFactory

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DIR = REPO_ROOT / "tests" / "fixtures" / "sample_documents"
INGEST_SCRIPT = REPO_ROOT / "scripts" / "ingest.py"
QUERY_SCRIPT = REPO_ROOT / "scripts" / "query.py"
EVALUATE_SCRIPT = REPO_ROOT / "scripts" / "evaluate.py"
COLLECTION = "test"
ACCEPTANCE_QUERY = "测试查询"


def _load_script(name: str, path: Path) -> ModuleType:
    """按路径加载 scripts/*.py，避免 scripts 目录不是包。"""
    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载脚本: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeEmbedding(BaseEmbedding):
    """固定维度向量，使全链路不依赖外部 Embedding / llama-server。"""

    def __init__(self, settings) -> None:
        self.settings = settings

    def embed(self, texts, trace=None) -> list[list[float]]:
        dim = self.settings.dimensions
        return [[0.1] * dim for _ in texts]


@pytest.fixture(autouse=True)
def _reset_factories() -> None:
    """每个用例前后恢复 Embedding / VectorStore 工厂。"""
    EmbeddingFactory.reset_constructor()
    VectorStoreFactory.reset_constructor()
    yield
    EmbeddingFactory.reset_constructor()
    VectorStoreFactory.reset_constructor()


@pytest.fixture
def isolated_trace_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """把 write_trace 默认落盘改到临时 jsonl，避免污染仓库 logs/traces.jsonl。"""
    trace_file = tmp_path / "traces.jsonl"

    def _resolve(path=None) -> Path:
        if path is not None:
            return Path(path)
        return trace_file

    monkeypatch.setattr("observability.logger.resolve_trace_file", _resolve)
    return trace_file


@pytest.mark.e2e
class TestFullPipelineAcceptance:
    """I5 验收：摄取 sample_documents、查询返回结果、追踪可展示、评估输出指标。"""

    def test_ingest_query_dashboard_evaluate(
        self,
        tmp_path: Path,
        isolated_trace_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """按 spec 命令走通 ingest → query → Dashboard 追踪 → evaluate。"""
        pytest.importorskip("streamlit")
        from streamlit.testing.v1 import AppTest

        data_root = tmp_path / "data"
        EmbeddingFactory.set_constructor(lambda settings: FakeEmbedding(settings))

        ingest = _load_script("ingest_cli_i5", INGEST_SCRIPT)
        ingest_code = ingest.main(
            [
                "--path",
                str(SAMPLE_DIR),
                "--collection",
                COLLECTION,
                "--force",
                "--data-root",
                str(data_root),
            ]
        )
        assert ingest_code == 0
        assert (data_root / "db" / "chroma").exists()
        assert (data_root / "db" / "bm25" / f"{COLLECTION}.json").is_file()

        query = _load_script("query_cli_i5", QUERY_SCRIPT)
        query_out = io.StringIO()
        query_code = query.run_query(
            ACCEPTANCE_QUERY,
            verbose=True,
            collection=COLLECTION,
            data_root=str(data_root),
            no_rerank=True,
            out=query_out,
        )
        assert query_code == 0
        rendered = query_out.getvalue()
        assert "=== Final Top-K ===" in rendered
        assert "未找到相关文档" not in rendered
        assert "=== Dense Results ===" in rendered

        traces_text = isolated_trace_file.read_text(encoding="utf-8")
        assert '"trace_type": "ingestion"' in traces_text
        assert '"trace_type": "query"' in traces_text

        monkeypatch.setenv("I5_TRACE_FILE", str(isolated_trace_file))

        def ingestion_page() -> None:
            import os
            from pathlib import Path as P

            from observability.dashboard.pages.ingestion_traces import render_ingestion_traces
            from observability.dashboard.services.trace_service import TraceService

            render_ingestion_traces(
                TraceService(P(os.environ["I5_TRACE_FILE"])),
                load_data=False,
            )

        ing_app = AppTest.from_function(ingestion_page, default_timeout=30)
        ing_app.run()
        assert not ing_app.exception, ing_app.exception
        assert "Ingestion 追踪" in [str(item.value) for item in ing_app.header]
        assert "查看详情" in [item.label for item in ing_app.selectbox]

        def query_page() -> None:
            import os
            from pathlib import Path as P

            from observability.dashboard.pages.query_traces import render_query_traces
            from observability.dashboard.services.trace_service import TraceService

            render_query_traces(
                TraceService(P(os.environ["I5_TRACE_FILE"])),
                load_data=False,
            )

        q_app = AppTest.from_function(query_page, default_timeout=30)
        q_app.run()
        assert not q_app.exception, q_app.exception
        assert "Query 追踪" in [str(item.value) for item in q_app.header]
        assert "查看详情" in [item.label for item in q_app.selectbox]

        settings, _bm25 = settings_for_query(load_settings(), COLLECTION, str(data_root))
        store = VectorStoreFactory.create(settings)
        records = store.get_by_metadata({"collection": COLLECTION}, collection=COLLECTION)
        assert records, "摄取后集合 test 应有 chunk"
        golden_path = tmp_path / "golden_i5.json"
        golden_path.write_text(
            json.dumps(
                {
                    "test_cases": [
                        {
                            "query": ACCEPTANCE_QUERY,
                            "expected_chunk_ids": [records[0]["id"]],
                            "expected_sources": ["sample.pdf"],
                        }
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        evaluate = _load_script("evaluate_cli_i5", EVALUATE_SCRIPT)
        eval_out = io.StringIO()
        eval_code = evaluate.run_evaluate(
            test_set=str(golden_path),
            collection=COLLECTION,
            data_root=str(data_root),
            as_json=True,
            evaluator=CustomEvaluator(),
            out=eval_out,
        )
        assert eval_code == 0
        report = json.loads(eval_out.getvalue())
        assert report["case_count"] == 1
        assert "hit_rate" in report
        assert "mrr" in report
        assert isinstance(report["hit_rate"], float)
