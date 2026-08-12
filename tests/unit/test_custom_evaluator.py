"""CustomEvaluator 与 EvaluatorFactory 的单元测试。"""

from __future__ import annotations

import pytest

from core.settings import EvaluationSettings, Settings, load_settings
from libs.evaluator.base_evaluator import EvaluatorError
from libs.evaluator.custom_evaluator import CustomEvaluator
from libs.evaluator.evaluator_factory import EvaluatorFactory, EvaluatorFactoryError


@pytest.fixture(autouse=True)
def _reset_evaluator_factory() -> None:
    """每个用例前后恢复工厂默认构造器。"""
    EvaluatorFactory.reset_constructor()
    yield
    EvaluatorFactory.reset_constructor()


@pytest.mark.unit
class TestCustomEvaluatorMetrics:
    """验证 hit_rate / mrr 计算稳定且符合定义。"""

    def test_hit_and_mrr_when_first_result_matches(self) -> None:
        """首位命中时 hit_rate=1、mrr=1.0。"""
        evaluator = CustomEvaluator()
        metrics = evaluator.evaluate(
            query="如何配置 Azure？",
            retrieved_ids=["chunk-001", "chunk-002"],
            golden_ids=["chunk-001"],
        )
        assert metrics["hit_rate"] == 1.0
        assert metrics["mrr"] == 1.0

    def test_mrr_when_match_at_second_rank(self) -> None:
        """第二位命中时 mrr=0.5。"""
        evaluator = CustomEvaluator()
        metrics = evaluator.evaluate(
            query="q",
            retrieved_ids=["chunk-002", "chunk-001"],
            golden_ids=["chunk-001"],
        )
        assert metrics["hit_rate"] == 1.0
        assert metrics["mrr"] == 0.5

    def test_zero_metrics_when_no_match(self) -> None:
        """无命中时 hit_rate 与 mrr 均为 0。"""
        evaluator = CustomEvaluator()
        metrics = evaluator.evaluate(
            query="q",
            retrieved_ids=["chunk-999"],
            golden_ids=["chunk-001"],
        )
        assert metrics["hit_rate"] == 0.0
        assert metrics["mrr"] == 0.0

    def test_metrics_filtered_by_settings(self) -> None:
        """配置 metrics 子集时只返回指定指标。"""
        settings = EvaluationSettings(
            enabled=True,
            provider="custom",
            metrics=["mrr"],
        )
        evaluator = CustomEvaluator(settings)
        metrics = evaluator.evaluate("q", ["a", "b"], ["b"])
        assert metrics == {"mrr": 0.5}

    def test_empty_golden_ids_raises(self) -> None:
        """golden_ids 为空应失败。"""
        evaluator = CustomEvaluator()
        with pytest.raises(EvaluatorError, match="golden_ids"):
            evaluator.evaluate("q", ["a"], [])


@pytest.mark.unit
class TestEvaluatorFactoryRouting:
    """验证工厂按 evaluation.provider 路由。"""

    def test_custom_provider_routing(self) -> None:
        """默认 settings 的 provider=custom 应返回 CustomEvaluator。"""
        evaluator = EvaluatorFactory.create(load_settings())
        assert isinstance(evaluator, CustomEvaluator)

    def test_unknown_provider_raises(self) -> None:
        """未注册的 provider 应抛出可读错误。"""
        base = load_settings()
        settings = Settings(
            llm=base.llm,
            embedding=base.embedding,
            vector_store=base.vector_store,
            retrieval=base.retrieval,
            rerank=base.rerank,
            evaluation=EvaluationSettings(
                enabled=True,
                provider="unknown_eval_xyz",
                metrics=["hit_rate"],
            ),
            observability=base.observability,
            ingestion=base.ingestion,
            vision_llm=base.vision_llm,
        )
        with pytest.raises(EvaluatorFactoryError, match="unknown_eval_xyz"):
            EvaluatorFactory.create(settings)
