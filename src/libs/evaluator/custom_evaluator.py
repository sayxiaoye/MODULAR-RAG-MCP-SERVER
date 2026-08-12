"""自定义轻量评估器：实现 hit_rate 与 MRR，不依赖外部评测框架。"""

from __future__ import annotations

from typing import Any

from core.settings import EvaluationSettings
from libs.evaluator.base_evaluator import BaseEvaluator, EvaluatorError


def _hit_rate(retrieved_ids: list[str], golden_ids: list[str]) -> float:
    """任一 golden 出现在检索结果中则命中，返回 1.0 否则 0.0。"""
    golden_set = set(golden_ids)
    return 1.0 if any(rid in golden_set for rid in retrieved_ids) else 0.0


def _mrr(retrieved_ids: list[str], golden_ids: list[str]) -> float:
    """第一个命中 golden 的倒数排名；未命中则 0.0。"""
    golden_set = set(golden_ids)
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in golden_set:
            return 1.0 / rank
    return 0.0


class CustomEvaluator(BaseEvaluator):
    """基于 ID 匹配的轻量评估器，输出 hit_rate 与 mrr。"""

    def __init__(self, settings: EvaluationSettings | None = None) -> None:
        self.settings = settings

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        self._validate_inputs(query, retrieved_ids, golden_ids)
        metrics = {
            "hit_rate": _hit_rate(retrieved_ids, golden_ids),
            "mrr": _mrr(retrieved_ids, golden_ids),
        }
        # 若配置指定了 metrics 子集，只返回请求的指标
        if self.settings and self.settings.metrics:
            allowed = set(self.settings.metrics)
            return {k: v for k, v in metrics.items() if k in allowed}
        return metrics
