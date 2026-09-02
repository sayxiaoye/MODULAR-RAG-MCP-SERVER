"""Ragas 评估器：封装 ragas 框架，实现 BaseEvaluator（H1）。"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from core.settings import EvaluationSettings
from libs.evaluator.base_evaluator import BaseEvaluator, EvaluatorError

# Ragas 默认产出的指标名（与 spec 一致）
RAGAS_METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision")

_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "faithfulness": ("faithfulness",),
    "answer_relevancy": ("answer_relevancy", "answer_relevance"),
    "context_precision": ("context_precision", "llm_context_precision_without_reference"),
}

EvaluateFn = Callable[..., Mapping[str, Any]]


class RagasEvaluator(BaseEvaluator):
    """
    用 Ragas 计算 Faithfulness / Answer Relevancy / Context Precision。

    未安装 ragas 时在 ``evaluate()`` 抛出带安装提示的 ImportError。
    测试可注入 ``evaluate_fn``，避免真实调用 LLM。
    """

    def __init__(
        self,
        settings: EvaluationSettings | None = None,
        *,
        evaluate_fn: EvaluateFn | None = None,
        llm: Any | None = None,
    ) -> None:
        self.settings = settings
        self._evaluate_fn = evaluate_fn
        self._llm = llm

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: Any | None = None,
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        对单次查询调用 Ragas（或注入的 evaluate_fn）。

        Args:
            query: 用户查询。
            retrieved_ids: 检索 chunk_id（无 contexts 时作为占位上下文）。
            golden_ids: 标注正确 chunk_id；可作为 ground_truth 回退。
            trace: 可选 TraceContext。
            **kwargs: ``answer`` / ``generated_answer``、``contexts`` / ``retrieved_texts``、
                ``ground_truth`` / ``reference``。

        Returns:
            至少含 faithfulness、answer_relevancy 的指标字典。

        Raises:
            ImportError: 未安装 ragas 且未注入 evaluate_fn。
            EvaluatorError: 输入非法或 Ragas 执行失败。
        """
        self._validate_inputs(query, retrieved_ids, golden_ids)
        payload = _build_payload(query, retrieved_ids, golden_ids, kwargs)
        runner = self._evaluate_fn or _run_ragas_evaluate
        try:
            raw = runner(payload, llm=self._llm)
        except ImportError:
            raise
        except Exception as exc:
            raise EvaluatorError(f"Ragas 评估失败: {exc}") from exc

        metrics = _normalize_metrics(raw)
        if "faithfulness" not in metrics or "answer_relevancy" not in metrics:
            raise EvaluatorError("Ragas 结果缺少 faithfulness 或 answer_relevancy")

        if self.settings and self.settings.metrics:
            allowed = set(self.settings.metrics)
            ragas_requested = allowed.intersection(RAGAS_METRIC_NAMES)
            # 仅当配置点名了 Ragas 指标时才裁剪，避免 provider=ragas 却只配了 hit_rate 时得到空字典
            if ragas_requested:
                metrics = {key: value for key, value in metrics.items() if key in ragas_requested}

        if trace is not None and hasattr(trace, "record_stage"):
            trace.record_stage(
                "evaluation",
                method="ragas",
                provider="ragas",
                metric_names=sorted(metrics.keys()),
            )
        return metrics


def _build_payload(
    query: str,
    retrieved_ids: list[str],
    golden_ids: list[str],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    """把 BaseEvaluator 入参整理成 Ragas 样本字段。"""
    answer = kwargs.get("answer", kwargs.get("generated_answer"))
    if answer is None:
        answer = ""
    contexts = kwargs.get("contexts", kwargs.get("retrieved_texts"))
    if contexts is None:
        # 未传入文本时用 chunk_id 占位，保证 Ragas 样本结构完整
        contexts = list(retrieved_ids)
    if not isinstance(contexts, list):
        raise EvaluatorError("contexts 必须是 list")
    ground_truth = kwargs.get("ground_truth", kwargs.get("reference"))
    if ground_truth is None:
        ground_truth = ", ".join(golden_ids)
    return {
        "question": query.strip(),
        "answer": str(answer),
        "contexts": [str(item) for item in contexts],
        "ground_truth": str(ground_truth),
    }


def _load_ragas() -> tuple[Any, list[Any]]:
    """延迟导入 ragas；缺失时给出可执行的安装提示。"""
    try:
        from ragas import evaluate as ragas_evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
    except ImportError as exc:
        raise ImportError(
            "未安装 Ragas。请执行: python -m pip install '.[evaluation]'"
        ) from exc
    return ragas_evaluate, [faithfulness, answer_relevancy, context_precision]


def _run_ragas_evaluate(payload: Mapping[str, Any], llm: Any | None = None) -> Mapping[str, Any]:
    """调用 ragas.evaluate；单条样本，返回指标映射。"""
    ragas_evaluate, metrics = _load_ragas()
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError(
            "未安装 datasets（Ragas 依赖）。请执行: python -m pip install '.[evaluation]'"
        ) from exc

    dataset = Dataset.from_dict(
        {
            "question": [payload["question"]],
            "answer": [payload["answer"]],
            "contexts": [list(payload["contexts"])],
            "ground_truth": [payload["ground_truth"]],
        }
    )
    kwargs: dict[str, Any] = {"metrics": metrics}
    if llm is not None:
        kwargs["llm"] = llm
    result = ragas_evaluate(dataset, **kwargs)
    return _result_to_mapping(result)


def _result_to_mapping(result: Any) -> dict[str, Any]:
    """兼容 ragas 返回 dict / pandas / 带 to_pandas 的对象。"""
    if isinstance(result, Mapping):
        return dict(result)
    to_pandas = getattr(result, "to_pandas", None)
    if callable(to_pandas):
        frame = to_pandas()
        if getattr(frame, "empty", True):
            return {}
        row = frame.iloc[0].to_dict()
        return {str(key): value for key, value in row.items()}
    try:
        return dict(result)
    except Exception as exc:
        raise EvaluatorError(f"无法解析 Ragas 返回值: {exc}") from exc


def _normalize_metrics(raw: Mapping[str, Any]) -> dict[str, float]:
    """把 Ragas 原始列名归一成 spec 中的三个指标名。"""
    normalized: dict[str, float] = {}
    for canonical, aliases in _METRIC_ALIASES.items():
        for alias in aliases:
            if alias not in raw:
                continue
            try:
                normalized[canonical] = float(raw[alias])
            except (TypeError, ValueError):
                continue
            break
    return normalized
