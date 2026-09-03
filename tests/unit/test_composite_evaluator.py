"""CompositeEvaluator 单元测试：并行合并指标与工厂 backends 组合。"""

from __future__ import annotations

import threading
from typing import Any

import pytest

from core.settings import EvaluationSettings, Settings, load_settings
from libs.evaluator.base_evaluator import BaseEvaluator, EvaluatorError
from libs.evaluator.custom_evaluator import CustomEvaluator
from libs.evaluator.evaluator_factory import EvaluatorFactory, EvaluatorFactoryError
from observability.evaluation.composite_evaluator import CompositeEvaluator
from observability.evaluation.ragas_evaluator import RagasEvaluator


@pytest.fixture(autouse=True)
def _reset_evaluator_factory() -> None:
    """每个用例前后恢复工厂默认构造器。"""
    EvaluatorFactory.reset_constructor()
    yield
    EvaluatorFactory.reset_constructor()


class _FakeEvaluator(BaseEvaluator):
    """测试用固定指标评估器，可注入阻塞以便验证并行。"""

    def __init__(
        self,
        metrics: dict[str, float],
        *,
        barrier: threading.Barrier | None = None,
        boom: Exception | None = None,
    ) -> None:
        self._metrics = metrics
        self._barrier = barrier
        self._boom = boom

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        if self._barrier is not None:
            # 两个评估器都到达才放行；串行执行会在此超时
            self._barrier.wait(timeout=2.0)
        if self._boom is not None:
            raise self._boom
        return dict(self._metrics)


def _settings_with_evaluation(evaluation: EvaluationSettings) -> Settings:
    """基于真实配置覆盖 evaluation 块，避免手写全部 Settings 字段。"""
    base = load_settings()
    return Settings(
        llm=base.llm,
        embedding=base.embedding,
        vector_store=base.vector_store,
        retrieval=base.retrieval,
        rerank=base.rerank,
        evaluation=evaluation,
        observability=base.observability,
        ingestion=base.ingestion,
        vision_llm=base.vision_llm,
    )


@pytest.mark.unit
class TestCompositeEvaluatorMerge:
    """验证组合评估器的构造约束与指标合并。"""

    def test_empty_evaluators_raises(self) -> None:
        """空列表不能构造，避免工厂产出无法评估的空组合。"""
        with pytest.raises(EvaluatorError, match="evaluators"):
            CompositeEvaluator([])

    def test_evaluate_merges_metrics_from_both_evaluators(self) -> None:
        """两个评估器的指标应合并进同一字典（验收：ragas + custom）。"""
        composite = CompositeEvaluator(
            [
                _FakeEvaluator({"faithfulness": 0.9, "answer_relevancy": 0.8}),
                _FakeEvaluator({"hit_rate": 1.0, "mrr": 0.5}),
            ]
        )
        metrics = composite.evaluate(
            query="如何配置 Azure OpenAI？",
            retrieved_ids=["chunk-a", "chunk-b"],
            golden_ids=["chunk-a"],
        )
        assert metrics["faithfulness"] == 0.9
        assert metrics["answer_relevancy"] == 0.8
        assert metrics["hit_rate"] == 1.0
        assert metrics["mrr"] == 0.5

    def test_later_evaluator_overwrites_same_metric_key(self) -> None:
        """同名指标按构造顺序覆盖，与线程完成先后无关。"""
        composite = CompositeEvaluator(
            [
                _FakeEvaluator({"score": 0.1}),
                _FakeEvaluator({"score": 0.9}),
            ]
        )
        assert composite.evaluate("q", ["a"], ["a"]) == {"score": 0.9}

    def test_runs_evaluators_in_parallel(self) -> None:
        """Barrier 握手：仅并行时两个 evaluate 能同时到达。"""
        barrier = threading.Barrier(2)
        composite = CompositeEvaluator(
            [
                _FakeEvaluator({"a": 1.0}, barrier=barrier),
                _FakeEvaluator({"b": 2.0}, barrier=barrier),
            ]
        )
        metrics = composite.evaluate("q", ["a"], ["a"])
        assert metrics == {"a": 1.0, "b": 2.0}

    def test_one_failure_raises_after_others_finish(self) -> None:
        """一路失败仍等另一路跑完，再包装为 EvaluatorError。"""
        barrier = threading.Barrier(2)
        composite = CompositeEvaluator(
            [
                _FakeEvaluator({"ok": 1.0}, barrier=barrier),
                _FakeEvaluator({}, barrier=barrier, boom=RuntimeError("ragas boom")),
            ]
        )
        with pytest.raises(EvaluatorError, match="组合评估失败"):
            composite.evaluate("q", ["a"], ["a"])

    def test_partial_error_merges_successful_metrics(self) -> None:
        """Ragas 部分失败时仍应合并 Custom 指标并上抛 PartialEvaluatorError。"""
        from libs.evaluator.base_evaluator import PartialEvaluatorError

        composite = CompositeEvaluator(
            [
                _FakeEvaluator(
                    {},
                    boom=PartialEvaluatorError(
                        "Ragas 部分指标失败: context_precision",
                        {"faithfulness": 0.9},
                    ),
                ),
                _FakeEvaluator({"hit_rate": 1.0}),
            ]
        )
        with pytest.raises(PartialEvaluatorError) as exc_info:
            composite.evaluate("q", ["a"], ["a"])
        assert exc_info.value.metrics["faithfulness"] == 0.9
        assert exc_info.value.metrics["hit_rate"] == 1.0


@pytest.mark.unit
class TestEvaluatorFactoryBackends:
    """验证 evaluation.backends 驱动工厂自动组合。"""

    def test_two_backends_return_composite_with_both_types(self) -> None:
        """backends=[ragas, custom] 应得到含两类实现的 CompositeEvaluator。"""
        settings = _settings_with_evaluation(
            EvaluationSettings(
                enabled=True,
                provider="custom",
                metrics=["hit_rate", "mrr"],
                backends=["ragas", "custom"],
            )
        )
        evaluator = EvaluatorFactory.create(settings)
        assert isinstance(evaluator, CompositeEvaluator)
        kinds = [type(child) for child in evaluator.evaluators]
        assert kinds == [RagasEvaluator, CustomEvaluator]
        ragas = evaluator.evaluators[0]
        assert isinstance(ragas, RagasEvaluator)
        assert ragas._llm is not None
        assert ragas._embeddings is not None
        assert evaluator.requires_generated_answer is True

    def test_factory_evaluate_includes_both_metric_sets(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """配置两个 backend 时，evaluate 结果同时含 ragas 与 custom 指标。"""

        def _fake_ragas_evaluate(
            self: RagasEvaluator,
            query: str,
            retrieved_ids: list[str],
            golden_ids: list[str],
            trace: Any | None = None,
            **kwargs: Any,
        ) -> dict[str, float]:
            return {"faithfulness": 0.91, "answer_relevancy": 0.82}

        monkeypatch.setattr(RagasEvaluator, "evaluate", _fake_ragas_evaluate)
        settings = _settings_with_evaluation(
            EvaluationSettings(
                enabled=True,
                provider="custom",
                metrics=["hit_rate", "mrr"],
                backends=["ragas", "custom"],
            )
        )
        evaluator = EvaluatorFactory.create(settings)
        metrics = evaluator.evaluate(
            query="q",
            retrieved_ids=["chunk-hit", "other"],
            golden_ids=["chunk-hit"],
        )
        assert metrics["faithfulness"] == 0.91
        assert metrics["answer_relevancy"] == 0.82
        assert metrics["hit_rate"] == 1.0
        assert metrics["mrr"] == 1.0

    def test_custom_metrics_alias_maps_to_custom(self) -> None:
        """spec 中的 custom_metrics 应映射到已注册的 custom。"""
        settings = _settings_with_evaluation(
            EvaluationSettings(
                enabled=True,
                provider="custom",
                metrics=["hit_rate"],
                backends=["custom_metrics"],
            )
        )
        evaluator = EvaluatorFactory.create(settings)
        assert isinstance(evaluator, CustomEvaluator)

    def test_single_backend_is_not_wrapped(self) -> None:
        """仅一项 backend 时不包 Composite，直接返回该实现。"""
        settings = _settings_with_evaluation(
            EvaluationSettings(
                enabled=True,
                provider="custom",
                metrics=["hit_rate"],
                backends=["custom"],
            )
        )
        evaluator = EvaluatorFactory.create(settings)
        assert isinstance(evaluator, CustomEvaluator)
        assert not isinstance(evaluator, CompositeEvaluator)

    def test_missing_backends_falls_back_to_provider(self) -> None:
        """未配置 backends 时仍按 provider 创建，兼容旧配置。"""
        evaluator = EvaluatorFactory.create(load_settings())
        assert isinstance(evaluator, CustomEvaluator)

    def test_unknown_backend_raises(self) -> None:
        """backends 含未注册名称时应失败。"""
        settings = _settings_with_evaluation(
            EvaluationSettings(
                enabled=True,
                provider="custom",
                metrics=["hit_rate"],
                backends=["ragas", "not-a-backend"],
            )
        )
        with pytest.raises(EvaluatorFactoryError, match="not-a-backend"):
            EvaluatorFactory.create(settings)
