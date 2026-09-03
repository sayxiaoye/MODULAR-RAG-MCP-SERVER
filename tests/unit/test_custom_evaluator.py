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

    def test_blank_query_raises(self) -> None:
        """空白 query 不是合法评估输入。"""
        evaluator = CustomEvaluator()
        with pytest.raises(EvaluatorError, match="query"):
            evaluator.evaluate("  ", ["a"], ["a"])

    def test_empty_retrieved_ids_is_miss(self) -> None:
        """无检索结果时 hit_rate/mrr 均为 0，形状仍是 float 指标字典。"""
        evaluator = CustomEvaluator()
        metrics = evaluator.evaluate("q", [], ["gold-1"])
        assert metrics == {"hit_rate": 0.0, "mrr": 0.0}

    def test_retrieved_ids_must_be_list(self) -> None:
        """retrieved_ids 必须是 list，避免误传入 tuple/set 导致排名失真。"""
        evaluator = CustomEvaluator()
        with pytest.raises(EvaluatorError, match="retrieved_ids"):
            evaluator.evaluate("q", ("a",), ["a"])  # type: ignore[arg-type]

    def test_blank_retrieved_id_raises(self) -> None:
        """空字符串 chunk_id 不是合法检索输出。"""
        evaluator = CustomEvaluator()
        with pytest.raises(EvaluatorError, match="retrieved_ids"):
            evaluator.evaluate("q", ["  "], ["gold"])

    def test_blank_golden_id_raises(self) -> None:
        """空字符串不能作为 golden chunk_id。"""
        evaluator = CustomEvaluator()
        with pytest.raises(EvaluatorError, match="golden_ids"):
            evaluator.evaluate("q", ["a"], [""])

    def test_empty_metrics_config_returns_all(self) -> None:
        """metrics=[] 视为未筛选，应返回 hit_rate 与 mrr。"""
        evaluator = CustomEvaluator(
            EvaluationSettings(enabled=True, provider="custom", metrics=[])
        )
        metrics = evaluator.evaluate("q", ["gold"], ["gold"])
        assert set(metrics) == {"hit_rate", "mrr"}

    def test_first_of_multiple_golden_sets_mrr(self) -> None:
        """多个 golden 时 MRR 取检索列表中最早命中的倒数排名。"""
        evaluator = CustomEvaluator()
        metrics = evaluator.evaluate(
            query="q",
            retrieved_ids=["x", "gold-b", "gold-a"],
            golden_ids=["gold-a", "gold-b"],
        )
        assert metrics["hit_rate"] == 1.0
        assert metrics["mrr"] == 0.5


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

    def test_blank_provider_raises(self) -> None:
        """provider 为空无法路由到注册表。"""
        base = load_settings()
        settings = Settings(
            llm=base.llm,
            embedding=base.embedding,
            vector_store=base.vector_store,
            retrieval=base.retrieval,
            rerank=base.rerank,
            evaluation=EvaluationSettings(
                enabled=True,
                provider="  ",
                metrics=["hit_rate"],
            ),
            observability=base.observability,
            ingestion=base.ingestion,
            vision_llm=base.vision_llm,
        )
        with pytest.raises(EvaluatorFactoryError, match="未知"):
            EvaluatorFactory.create(settings)

    def test_disabled_still_creates_custom(self) -> None:
        """enabled=false 不改变工厂路由，仍返回已配置的 CustomEvaluator。"""
        base = load_settings()
        settings = Settings(
            llm=base.llm,
            embedding=base.embedding,
            vector_store=base.vector_store,
            retrieval=base.retrieval,
            rerank=base.rerank,
            evaluation=EvaluationSettings(
                enabled=False,
                provider="custom",
                metrics=["hit_rate", "mrr"],
            ),
            observability=base.observability,
            ingestion=base.ingestion,
            vision_llm=base.vision_llm,
        )
        evaluator = EvaluatorFactory.create(settings)
        assert isinstance(evaluator, CustomEvaluator)
