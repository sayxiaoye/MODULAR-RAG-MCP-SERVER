"""Evaluator 抽象层：定义 RAG 检索质量评估接口，供 EvalRunner 与回归测试复用。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class EvaluatorError(Exception):
    """评估输入校验或指标计算失败时抛出。"""


class BaseEvaluator(ABC):
    """Evaluator 抽象基类：根据检索结果与标准答案计算指标。"""

    @abstractmethod
    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        对单次查询的检索结果进行评估。

        Args:
            query: 用户查询文本（部分评估器用于生成类指标）。
            retrieved_ids: 检索返回的 chunk_id 列表（按相关性排序）。
            golden_ids: 标注的正确 chunk_id 列表。
            trace: 可选追踪上下文。
            **kwargs: 扩展参数（如 top_k、生成答案等）。

        Returns:
            指标名 -> 数值 的字典，例如 hit_rate、mrr。
        """

    def _validate_inputs(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
    ) -> None:
        """校验评估输入的基本 shape。"""
        if not isinstance(query, str) or not query.strip():
            raise EvaluatorError("query 必须是非空字符串")
        if not isinstance(retrieved_ids, list):
            raise EvaluatorError("retrieved_ids 必须是 list")
        if not isinstance(golden_ids, list) or not golden_ids:
            raise EvaluatorError("golden_ids 必须是非空 list")
        for index, item in enumerate(retrieved_ids):
            if not isinstance(item, str) or not item.strip():
                raise EvaluatorError(f"retrieved_ids[{index}] 必须是非空字符串")
        for index, item in enumerate(golden_ids):
            if not isinstance(item, str) or not item.strip():
                raise EvaluatorError(f"golden_ids[{index}] 必须是非空字符串")
