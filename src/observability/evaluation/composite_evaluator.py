"""组合评估器：并行执行多个 BaseEvaluator，按配置顺序合并 metrics（H2）。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

from libs.evaluator.base_evaluator import BaseEvaluator, EvaluatorError


class CompositeEvaluator(BaseEvaluator):
    """
    将多个 Evaluator 组合成一次评估。

    对应 spec H2：``evaluate()`` 并行跑全部子评估器，再按构造顺序合并指标。
    空列表在构造时即失败，避免工厂产出无法评估的空组合。
    """

    def __init__(self, evaluators: Sequence[BaseEvaluator]) -> None:
        """
        Args:
            evaluators: 至少一个 BaseEvaluator；顺序决定同名指标的覆盖先后。

        Raises:
            EvaluatorError: 列表为空。
        """
        if not evaluators:
            raise EvaluatorError("evaluators 不能为空")
        self._evaluators = list(evaluators)

    @property
    def evaluators(self) -> list[BaseEvaluator]:
        """只读副本，便于测试断言工厂组合结果。"""
        return list(self._evaluators)

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        并行调用全部子评估器，按列表顺序合并返回的 metrics。

        先等所有任务结束再决定成败，避免一路失败就取消另一路。
        同名指标以后写入者为准（与列表顺序一致，与完成先后无关）。

        Args:
            query: 用户查询。
            retrieved_ids: 检索返回的 chunk_id。
            golden_ids: 标注正确的 chunk_id。
            trace: 可选追踪上下文，原样传给子评估器。
            **kwargs: 透传给各子评估器（如 answer、contexts）。

        Returns:
            合并后的指标字典。

        Raises:
            EvaluatorError: 输入非法，或任一子评估器失败。
            ImportError: 子评估器因缺依赖抛出时原样上抛（如未安装 ragas）。
        """
        self._validate_inputs(query, retrieved_ids, golden_ids)

        # 线程数与子评估器数量对齐，避免无谓排队
        worker_count = max(1, len(self._evaluators))
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = [
                pool.submit(
                    evaluator.evaluate,
                    query,
                    retrieved_ids,
                    golden_ids,
                    trace,
                    **kwargs,
                )
                for evaluator in self._evaluators
            ]
            # 按提交顺序取结果，保证合并顺序稳定
            parts: list[dict[str, float]] = []
            first_error: BaseException | None = None
            for future in futures:
                try:
                    parts.append(future.result())
                except ImportError:
                    raise
                except Exception as exc:
                    if first_error is None:
                        first_error = exc

        if first_error is not None:
            raise EvaluatorError(f"组合评估失败: {first_error}") from first_error

        merged: dict[str, float] = {}
        for part in parts:
            merged.update(part)

        if trace is not None and hasattr(trace, "record_stage"):
            trace.record_stage(
                "evaluation",
                method="composite",
                evaluator_count=len(self._evaluators),
                metric_names=sorted(merged.keys()),
            )
        return merged
