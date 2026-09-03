"""RagasEvaluator 单元测试：mock 评估函数、缺依赖提示与工厂路由。"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Mapping

import pytest

from core.settings import EvaluationSettings, Settings, load_settings
from libs.evaluator.base_evaluator import EvaluatorError
from libs.evaluator.evaluator_factory import EvaluatorFactory, EvaluatorFactoryError
from observability.evaluation.ragas_evaluator import RagasEvaluator, _run_sync_in_fresh_loop


@pytest.fixture(autouse=True)
def _reset_evaluator_factory() -> None:
    """每个用例前后恢复工厂默认构造器。"""
    EvaluatorFactory.reset_constructor()
    yield
    EvaluatorFactory.reset_constructor()


def _fake_ragas_scores(
    payload: Mapping[str, Any],
    llm: Any | None = None,
    embeddings: Any | None = None,
) -> dict[str, float]:
    """模拟 Ragas 在 mock LLM 下返回的三项指标。"""
    assert payload["question"]
    return {
        "faithfulness": 0.9,
        "answer_relevancy": 0.8,
        "context_precision": 0.7,
    }


@pytest.mark.unit
class TestRagasEvaluator:
    """验证 RagasEvaluator 对 BaseEvaluator 契约与降级行为。"""

    def test_evaluate_returns_faithfulness_and_answer_relevancy(self) -> None:
        """mock 环境下 evaluate 应返回 faithfulness 与 answer_relevancy。"""
        evaluator = RagasEvaluator(evaluate_fn=_fake_ragas_scores)
        metrics = evaluator.evaluate(
            query="如何配置 Azure OpenAI？",
            retrieved_ids=["chunk-a", "chunk-b"],
            golden_ids=["chunk-a"],
            answer="在 Azure 门户创建资源并填写 endpoint。",
            contexts=["Azure OpenAI 配置步骤……"],
        )
        assert metrics["faithfulness"] == 0.9
        assert metrics["answer_relevancy"] == 0.8
        assert metrics["context_precision"] == 0.7

    def test_missing_ragas_raises_install_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """未安装 ragas 时应抛出带 pip 提示的 ImportError。"""

        def _boom() -> tuple[Any, list[Any]]:
            raise ImportError("未安装 Ragas。请执行: python -m pip install '.[evaluation]'")

        monkeypatch.setattr(
            "observability.evaluation.ragas_evaluator._load_ragas",
            _boom,
        )
        evaluator = RagasEvaluator()
        with pytest.raises(ImportError, match=r"\[evaluation\]"):
            evaluator.evaluate("q", ["a"], ["a"], answer="ans")

    def test_invalid_inputs_raise(self) -> None:
        """空 query / 空 golden_ids 应走 BaseEvaluator 校验。"""
        evaluator = RagasEvaluator(evaluate_fn=_fake_ragas_scores)
        with pytest.raises(EvaluatorError, match="query"):
            evaluator.evaluate("  ", ["a"], ["a"])
        with pytest.raises(EvaluatorError, match="golden_ids"):
            evaluator.evaluate("q", ["a"], [])

    def test_metrics_subset_from_settings(self) -> None:
        """配置只点名 faithfulness 时不应返回其它 Ragas 指标。"""
        settings = EvaluationSettings(
            enabled=True,
            provider="ragas",
            metrics=["faithfulness"],
        )
        evaluator = RagasEvaluator(settings, evaluate_fn=_fake_ragas_scores)
        metrics = evaluator.evaluate("q", ["a"], ["a"], answer="x")
        assert metrics == {"faithfulness": 0.9}

    def test_records_trace_stage(self) -> None:
        """传入 trace 时应记录 evaluation 阶段。"""

        class FakeTrace:
            def __init__(self) -> None:
                self.stages: list[dict[str, Any]] = []

            def record_stage(self, name: str, **details: Any) -> None:
                self.stages.append({"name": name, **details})

        trace = FakeTrace()
        evaluator = RagasEvaluator(evaluate_fn=_fake_ragas_scores)
        evaluator.evaluate("q", ["a"], ["a"], answer="x", trace=trace)
        assert trace.stages[0]["name"] == "evaluation"
        assert trace.stages[0]["method"] == "ragas"

    def test_forwards_injected_llm_and_embeddings(self) -> None:
        """注入的 Judge LLM / Embedding 应原样传给 evaluate_fn。"""
        seen: dict[str, Any] = {}

        def _capture(
            payload: Mapping[str, Any],
            llm: Any | None = None,
            embeddings: Any | None = None,
        ) -> dict[str, float]:
            seen["llm"] = llm
            seen["embeddings"] = embeddings
            seen["answer"] = payload.get("answer")
            return {
                "faithfulness": 0.9,
                "answer_relevancy": 0.8,
                "context_precision": 0.7,
            }

        marker_llm = object()
        marker_emb = object()
        evaluator = RagasEvaluator(
            evaluate_fn=_capture,
            llm=marker_llm,
            embeddings=marker_emb,
        )
        evaluator.evaluate("q", ["a"], ["a"], answer="generated")
        assert seen["llm"] is marker_llm
        assert seen["embeddings"] is marker_emb
        assert seen["answer"] == "generated"

    def test_requires_generated_answer(self) -> None:
        """Ragas 路径需要评估前生成答案。"""
        assert RagasEvaluator(evaluate_fn=_fake_ragas_scores).requires_generated_answer is True

    def test_nan_metrics_are_rejected(self) -> None:
        """ragas 在事件循环冲突时会返回 nan，不能当成有效分数。"""

        def _nan_scores(
            payload: Mapping[str, Any],
            llm: Any | None = None,
            embeddings: Any | None = None,
        ) -> dict[str, float]:
            return {
                "faithfulness": float("nan"),
                "answer_relevancy": float("nan"),
                "context_precision": float("nan"),
            }

        evaluator = RagasEvaluator(evaluate_fn=_nan_scores)
        with pytest.raises(EvaluatorError, match="faithfulness"):
            evaluator.evaluate("q", ["a"], ["a"], answer="x")

    def test_fresh_loop_isolates_from_running_parent(self) -> None:
        """父协程已有事件循环时，打分应在新线程里跑。"""
        seen: dict[str, Any] = {}

        def inner() -> int:
            try:
                asyncio.get_running_loop()
                seen["inner_running"] = True
            except RuntimeError:
                seen["inner_running"] = False
            seen["thread"] = threading.current_thread().name
            return 7

        async def parent() -> int:
            seen["parent_thread"] = threading.current_thread().name
            return _run_sync_in_fresh_loop(inner)

        assert asyncio.run(parent()) == 7
        assert seen["inner_running"] is False
        assert seen["thread"] != seen["parent_thread"]


@pytest.mark.unit
class TestRagasFactoryRouting:
    """验证 EvaluatorFactory 可按 provider=ragas 构造 RagasEvaluator。"""

    def test_ragas_provider_returns_ragas_evaluator(self) -> None:
        """evaluation.provider=ragas 应路由到 RagasEvaluator。"""
        base = load_settings()
        settings = Settings(
            llm=base.llm,
            embedding=base.embedding,
            vector_store=base.vector_store,
            retrieval=base.retrieval,
            rerank=base.rerank,
            evaluation=EvaluationSettings(
                enabled=True,
                provider="ragas",
                metrics=["faithfulness", "answer_relevancy"],
            ),
            observability=base.observability,
            ingestion=base.ingestion,
            vision_llm=base.vision_llm,
        )
        evaluator = EvaluatorFactory.create(settings)
        assert isinstance(evaluator, RagasEvaluator)
        assert evaluator._llm is not None
        assert hasattr(evaluator._llm, "chat")
        assert evaluator._embeddings is not None
        assert hasattr(evaluator._embeddings, "embed")

    def test_unknown_provider_still_raises(self) -> None:
        """未注册 provider 仍应失败。"""
        base = load_settings()
        settings = Settings(
            llm=base.llm,
            embedding=base.embedding,
            vector_store=base.vector_store,
            retrieval=base.retrieval,
            rerank=base.rerank,
            evaluation=EvaluationSettings(
                enabled=True,
                provider="not-a-backend",
                metrics=["hit_rate"],
            ),
            observability=base.observability,
            ingestion=base.ingestion,
            vision_llm=base.vision_llm,
        )
        with pytest.raises(EvaluatorFactoryError, match="not-a-backend"):
            EvaluatorFactory.create(settings)
