"""Ragas 评估器：封装 ragas 框架，实现 BaseEvaluator（H1）。"""

from __future__ import annotations

import asyncio
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Mapping, TypeVar

from core.settings import EvaluationSettings
from libs.evaluator.base_evaluator import BaseEvaluator, EvaluatorError, PartialEvaluatorError

_T = TypeVar("_T")

# Ragas 默认产出的指标名（与 spec 一致）
RAGAS_METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision")
# context_precision 对每条 chunk 打一轮 JSON；日语集合检索长，限制条数降低解析失败面
_CONTEXT_PRECISION_MAX_CHUNKS = 5

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
        embeddings: Any | None = None,
    ) -> None:
        self.settings = settings
        self._evaluate_fn = evaluate_fn
        self._llm = llm
        self._embeddings = embeddings

    @property
    def requires_generated_answer(self) -> bool:
        """Faithfulness / Answer Relevancy 依赖生成答案字段。"""
        return True

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
            已成功解析的指标字典；部分失败时抛 ``PartialEvaluatorError``（带已有分数）。

        Raises:
            ImportError: 未安装 ragas 且未注入 evaluate_fn。
            PartialEvaluatorError: 至少一项成功、其余解析失败。
            EvaluatorError: 输入非法或全部指标失败。
        """
        self._validate_inputs(query, retrieved_ids, golden_ids)
        payload = _build_payload(query, retrieved_ids, golden_ids, kwargs)
        runner = self._evaluate_fn or _run_ragas_evaluate
        try:
            raw = runner(payload, llm=self._llm, embeddings=self._embeddings)
        except ImportError:
            raise
        except PartialEvaluatorError:
            raise
        except Exception as exc:
            raise EvaluatorError(f"Ragas 评估失败: {exc}") from exc

        metrics, errors = _split_metric_errors(raw)
        metrics = _normalize_metrics(metrics)
        if self.settings and self.settings.metrics:
            allowed = set(self.settings.metrics)
            ragas_requested = allowed.intersection(RAGAS_METRIC_NAMES)
            # 仅当配置点名了 Ragas 指标时才裁剪，避免 provider=ragas 却只配了 hit_rate 时得到空字典
            if ragas_requested:
                metrics = {key: value for key, value in metrics.items() if key in ragas_requested}
        if not metrics:
            detail = "; ".join(errors) if errors else "无有效分数"
            raise EvaluatorError(f"Ragas 结果缺少有效指标: {detail}")
        if errors:
            raise PartialEvaluatorError(
                "Ragas 部分指标失败: " + "; ".join(errors),
                metrics,
            )

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
    from observability.evaluation.cjk_text import normalize_judge_text

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
        "question": normalize_judge_text(query.strip()),
        "answer": normalize_judge_text(str(answer)),
        "contexts": [normalize_judge_text(str(item)) for item in contexts],
        "ground_truth": normalize_judge_text(str(ground_truth)),
    }


def _load_ragas() -> tuple[Any, list[Any]]:
    """延迟导入 ragas 指标类；每次返回新实例，避免改到模块级单例。"""
    try:
        from ragas.metrics import AnswerRelevancy, Faithfulness
    except ImportError as exc:
        raise ImportError(
            "未安装 Ragas。请执行: python -m pip install '.[evaluation]'"
        ) from exc
    from observability.evaluation.ragas_cjk_prompts import (
        apply_cjk_judge_prompts,
        patch_cjk_statement_split,
    )

    faithfulness = Faithfulness()
    apply_cjk_judge_prompts(faithfulness)
    patch_cjk_statement_split(faithfulness)
    relevancy = AnswerRelevancy(strictness=1)
    apply_cjk_judge_prompts(relevancy)
    # 黄金集通常没有自然语言 reference，用 without_reference 避免拿 chunk_id 当标准答案
    try:
        from ragas.metrics import LLMContextPrecisionWithoutReference

        context_metric = LLMContextPrecisionWithoutReference()
    except Exception:
        from ragas.metrics import ContextPrecision as context_metric_cls

        context_metric = context_metric_cls()
    apply_cjk_judge_prompts(context_metric)
    # strictness=1：不要 gather 多次 Judge，避免 exclusive_gpu 并发抢模型
    return None, [faithfulness, relevancy, context_metric]


def _run_sync_in_fresh_loop(func: Callable[[], _T]) -> _T:
    """在独立线程执行，避免占用 Streamlit 的事件循环。"""
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="ragas-eval") as pool:
        return pool.submit(func).result()


def _run_ragas_evaluate(
    payload: Mapping[str, Any],
    llm: Any | None = None,
    embeddings: Any | None = None,
) -> Mapping[str, Any]:
    """
    对单条样本打分。

    不调用 ``ragas.evaluate()``：它内部 ``asyncio.wait_for`` 在 Python 3.14 +
    nest_asyncio 下会立刻 ``Timeout should be used inside a task`` 并返回 nan。
    """
    _, metrics = _load_ragas()
    return _run_sync_in_fresh_loop(
        lambda: _score_metrics_without_evaluate(payload, llm, embeddings, metrics)
    )


def _score_metrics_without_evaluate(
    payload: Mapping[str, Any],
    llm: Any,
    embeddings: Any,
    metrics: list[Any],
) -> dict[str, Any]:
    """在新事件循环里逐个 ``_single_turn_ascore``，绕过 wait_for。"""
    from ragas.dataset_schema import SingleTurnSample
    from ragas.metrics.base import MetricWithEmbeddings, MetricWithLLM
    from ragas.run_config import RunConfig

    from observability.evaluation.ragas_adapters import (
        wrap_project_embeddings_for_ragas,
        wrap_project_llm_for_ragas,
    )

    wrapped_llm = wrap_project_llm_for_ragas(llm)
    wrapped_emb = wrap_project_embeddings_for_ragas(embeddings)
    run_config = RunConfig(max_workers=1, timeout=180)
    for metric in metrics:
        if isinstance(metric, MetricWithLLM) and metric.llm is None:
            metric.llm = wrapped_llm
        if isinstance(metric, MetricWithEmbeddings) and metric.embeddings is None:
            metric.embeddings = wrapped_emb
        metric.init(run_config)

    sample = SingleTurnSample(
        user_input=payload["question"],
        response=payload["answer"],
        retrieved_contexts=list(payload["contexts"]),
        reference=str(payload.get("ground_truth") or ""),
    )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    raw: dict[str, Any] = {}
    errors: list[str] = []
    try:
        for metric in metrics:
            sample_for_metric = _sample_for_metric(metric, sample, payload)
            try:
                score = loop.run_until_complete(
                    metric._single_turn_ascore(sample=sample_for_metric, callbacks=[])
                )
                number = _finite_float(score)
                if number is None:
                    errors.append(
                        f"{metric.name}: 返回非数值 {score!r}（Judge 输出可能无法解析）"
                    )
                else:
                    raw[metric.name] = number
            except Exception as exc:
                errors.append(f"{metric.name}: {type(exc).__name__}: {exc}")
    finally:
        try:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        except Exception:
            pass
        loop.close()
        asyncio.set_event_loop(None)

    if errors:
        raw["_metric_errors"] = errors
    return raw


def _split_metric_errors(raw: Any) -> tuple[dict[str, Any], list[str]]:
    """从打分结果里抽出 ``_metric_errors``，避免当成指标列。"""
    if not isinstance(raw, Mapping):
        try:
            return dict(raw), []
        except Exception:
            return {}, []
    data = dict(raw)
    errors_raw = data.pop("_metric_errors", None)
    errors: list[str] = []
    if isinstance(errors_raw, list):
        errors = [str(item) for item in errors_raw if str(item).strip()]
    elif isinstance(errors_raw, str) and errors_raw.strip():
        errors = [errors_raw]
    return data, errors


def _sample_for_metric(metric: Any, sample: Any, payload: Mapping[str, Any]) -> Any:
    """context_precision 只对前若干条 chunk 打分，降低日语长检索的 JSON 失败面。"""
    name = str(getattr(metric, "name", "") or "")
    if "context_precision" not in name:
        return sample
    from ragas.dataset_schema import SingleTurnSample

    contexts = list(payload["contexts"])[:_CONTEXT_PRECISION_MAX_CHUNKS]
    return SingleTurnSample(
        user_input=sample.user_input,
        response=sample.response,
        retrieved_contexts=contexts,
        reference=getattr(sample, "reference", None),
    )


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
    """把 Ragas 原始列名归一成 spec 中的三个指标名；丢弃 nan/inf。"""
    normalized: dict[str, float] = {}
    for canonical, aliases in _METRIC_ALIASES.items():
        for alias in aliases:
            if alias not in raw:
                continue
            value = _finite_float(raw[alias])
            if value is None:
                continue
            normalized[canonical] = value
            break
    return normalized


def _finite_float(value: Any) -> float | None:
    """把可解析的有限浮点数取出；nan 视为无效。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number
