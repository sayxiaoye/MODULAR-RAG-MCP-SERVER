"""评估运行器：读取黄金测试集，跑检索并汇总 hit_rate / mrr（H3）。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from core.settings import Settings
from libs.evaluator.base_evaluator import BaseEvaluator, EvaluatorError, PartialEvaluatorError
from libs.llm.base_llm import BaseLLM


class EvalRunnerError(Exception):
    """黄金测试集无法读取，或评估流程无法启动时抛出。"""


@dataclass
class EvalCaseResult:
    """单条 query 的评估明细，供报告与 Dashboard 展示。"""

    query: str
    retrieved_ids: list[str]
    golden_ids: list[str]
    expected_sources: list[str]
    retrieved_sources: list[str]
    metrics: dict[str, float]
    error: str | None = None
    generated_answer: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """序列化为可 JSON 化的字典。"""
        payload: dict[str, Any] = {
            "query": self.query,
            "retrieved_ids": list(self.retrieved_ids),
            "golden_ids": list(self.golden_ids),
            "expected_sources": list(self.expected_sources),
            "retrieved_sources": list(self.retrieved_sources),
            "metrics": dict(self.metrics),
        }
        if self.generated_answer:
            payload["generated_answer"] = self.generated_answer
        if self.error:
            payload["error"] = self.error
        return payload


@dataclass
class EvalReport:
    """整次评估报告：宏观 hit_rate / mrr + 各 query 详情。"""

    hit_rate: float
    mrr: float
    case_count: int
    cases: list[EvalCaseResult] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """序列化报告，便于 CLI 打印与后续落盘。"""
        return {
            "hit_rate": self.hit_rate,
            "mrr": self.mrr,
            "case_count": self.case_count,
            "metrics": dict(self.metrics),
            "cases": [item.to_dict() for item in self.cases],
        }


def load_golden_test_set(test_set_path: str | Path) -> list[dict[str, Any]]:
    """
    读取 ``golden_test_set.json``，校验 test_cases 结构。

    Args:
        test_set_path: JSON 文件路径。

    Returns:
        规范化后的用例列表，每项含 query / expected_chunk_ids / expected_sources。

    Raises:
        EvalRunnerError: 文件不存在、JSON 非法或缺少有效用例。
    """
    path = Path(test_set_path)
    if not path.is_file():
        raise EvalRunnerError(f"找不到黄金测试集: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EvalRunnerError(f"黄金测试集 JSON 非法: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise EvalRunnerError("黄金测试集根节点必须是对象")
    cases = raw.get("test_cases")
    if not isinstance(cases, list) or not cases:
        raise EvalRunnerError("黄金测试集缺少非空 test_cases")

    parsed: list[dict[str, Any]] = []
    for index, item in enumerate(cases):
        if not isinstance(item, Mapping):
            raise EvalRunnerError(f"test_cases[{index}] 必须是对象")
        query = item.get("query")
        if not isinstance(query, str) or not query.strip():
            raise EvalRunnerError(f"test_cases[{index}].query 必须是非空字符串")
        chunk_ids = _as_str_list(item.get("expected_chunk_ids"), f"test_cases[{index}].expected_chunk_ids")
        if not chunk_ids:
            raise EvalRunnerError(f"test_cases[{index}] 缺少 expected_chunk_ids")
        sources = _as_str_list(item.get("expected_sources"), f"test_cases[{index}].expected_sources")
        parsed.append(
            {
                "query": query.strip(),
                "expected_chunk_ids": chunk_ids,
                "expected_sources": sources,
            }
        )
    return parsed


def _as_str_list(value: Any, field_name: str) -> list[str]:
    """把 JSON 字段规范成非空字符串列表；缺省视为空列表。"""
    if value is None:
        return []
    if not isinstance(value, list):
        raise EvalRunnerError(f"{field_name} 必须是数组")
    result: list[str] = []
    for index, item in enumerate(value):
        text = str(item).strip()
        if not text:
            raise EvalRunnerError(f"{field_name}[{index}] 必须是非空字符串")
        result.append(text)
    return result


def _source_label(metadata: Mapping[str, Any] | None) -> str:
    """从检索结果 metadata 提取来源文件名。"""
    if not metadata:
        return ""
    for key in ("source_path", "source_file", "file_name", "source"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


class EvalRunner:
    """
    对黄金测试集逐条检索并评估。

    对应 spec H3：``run(test_set_path) -> EvalReport``。
    单条检索/评估失败会记入该条 ``error``，不中断整次运行。
    """

    def __init__(
        self,
        settings: Settings,
        hybrid_search: Any,
        evaluator: BaseEvaluator,
        *,
        answer_llm: BaseLLM | None = None,
        answer_fn: Callable[[str, list[str]], str] | None = None,
    ) -> None:
        """
        Args:
            settings: 全局配置（用于 top_k 等）。
            hybrid_search: 具备 ``search(query, ...)`` 的混合检索器。
            evaluator: Custom / Ragas / Composite 等 BaseEvaluator。
            answer_llm: 生成答案用的项目 LLM；缺省且需要答案时由工厂创建。
            answer_fn: 测试注入的 ``(query, contexts) -> answer``，优先于 answer_llm。
        """
        if hybrid_search is None:
            raise EvalRunnerError("hybrid_search 不能为空")
        if evaluator is None:
            raise EvalRunnerError("evaluator 不能为空")
        self._settings = settings
        self._hybrid_search = hybrid_search
        self._evaluator = evaluator
        self._answer_llm = answer_llm
        self._answer_fn = answer_fn

    def run(self, test_set_path: str | Path) -> EvalReport:
        """
        读取测试集、逐条检索并评估，返回汇总报告。

        Args:
            test_set_path: ``golden_test_set.json`` 路径。

        Returns:
            含宏观 hit_rate / mrr 与各 query 明细的 EvalReport。
        """
        cases = load_golden_test_set(test_set_path)
        results = [self._run_one(case) for case in cases]
        return _build_report(results)


    def _run_one(self, case: Mapping[str, Any]) -> EvalCaseResult:
        """跑单条用例；检索或评估失败时写入 error，指标按 0 处理。"""
        query = str(case["query"])
        golden_ids = list(case["expected_chunk_ids"])
        expected_sources = list(case["expected_sources"])
        try:
            hits = self._hybrid_search.search(query)
        except Exception as exc:
            return EvalCaseResult(
                query=query,
                retrieved_ids=[],
                golden_ids=golden_ids,
                expected_sources=expected_sources,
                retrieved_sources=[],
                metrics={},
                error=f"检索失败: {exc}",
            )

        retrieved_ids, retrieved_sources, texts = _unpack_hits(hits)
        answer: str | None = None
        if self._evaluator.requires_generated_answer:
            try:
                answer = self._resolve_answer(query, texts)
            except Exception as exc:
                return EvalCaseResult(
                    query=query,
                    retrieved_ids=retrieved_ids,
                    golden_ids=golden_ids,
                    expected_sources=expected_sources,
                    retrieved_sources=retrieved_sources,
                    metrics={},
                    error=f"生成答案失败: {exc}",
                )
        try:
            metrics = self._evaluator.evaluate(
                query,
                retrieved_ids,
                golden_ids,
                contexts=texts,
                answer=answer,
            )
        except PartialEvaluatorError as exc:
            return EvalCaseResult(
                query=query,
                retrieved_ids=retrieved_ids,
                golden_ids=golden_ids,
                expected_sources=expected_sources,
                retrieved_sources=retrieved_sources,
                metrics={key: float(value) for key, value in exc.metrics.items()},
                error=f"评估部分失败: {exc}",
                generated_answer=answer,
            )
        except Exception as exc:
            return EvalCaseResult(
                query=query,
                retrieved_ids=retrieved_ids,
                golden_ids=golden_ids,
                expected_sources=expected_sources,
                retrieved_sources=retrieved_sources,
                metrics={},
                error=f"评估失败: {exc}",
                generated_answer=answer,
            )
        return EvalCaseResult(
            query=query,
            retrieved_ids=retrieved_ids,
            golden_ids=golden_ids,
            expected_sources=expected_sources,
            retrieved_sources=retrieved_sources,
            metrics={key: float(value) for key, value in metrics.items()},
            generated_answer=answer,
        )

    def _resolve_answer(self, query: str, texts: list[str]) -> str:
        """生成供 Ragas 使用的 RAG 答案；测试可注入 answer_fn。"""
        if self._answer_fn is not None:
            return self._answer_fn(query, texts)
        llm = self._answer_llm
        if llm is None:
            from libs.llm.llm_factory import LLMFactory

            llm = LLMFactory.create(self._settings)
            self._answer_llm = llm
        from observability.evaluation.answer_generator import generate_rag_answer

        return generate_rag_answer(query, texts, llm)


def _unpack_hits(hits: Sequence[Any]) -> tuple[list[str], list[str], list[str]]:
    """从 RetrievalResult（或同类对象）抽出 id / source / text。"""
    retrieved_ids: list[str] = []
    retrieved_sources: list[str] = []
    texts: list[str] = []
    for item in hits:
        chunk_id = getattr(item, "chunk_id", None)
        if chunk_id is None and isinstance(item, Mapping):
            chunk_id = item.get("chunk_id", item.get("id"))
        retrieved_ids.append(str(chunk_id or "").strip())
        metadata = getattr(item, "metadata", None)
        if metadata is None and isinstance(item, Mapping):
            metadata = item.get("metadata")
        retrieved_sources.append(_source_label(metadata if isinstance(metadata, Mapping) else None))
        text = getattr(item, "text", None)
        if text is None and isinstance(item, Mapping):
            text = item.get("text")
        texts.append(str(text or ""))
    return retrieved_ids, retrieved_sources, texts


def _build_report(results: list[EvalCaseResult]) -> EvalReport:
    """按用例平均 hit_rate / mrr 及其余数值指标；部分失败仍计入已算出的分数。"""
    hit_rates: list[float] = []
    mrrs: list[float] = []
    extras: dict[str, list[float]] = {}
    for item in results:
        metrics = item.metrics
        hit_rates.append(float(metrics.get("hit_rate", 0.0)))
        mrrs.append(float(metrics.get("mrr", 0.0)))
        for key, value in metrics.items():
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(number):
                continue
            extras.setdefault(key, []).append(number)

    aggregated = {key: _mean(values) for key, values in extras.items()}
    hit_rate = _mean(hit_rates)
    mrr = _mean(mrrs)
    aggregated.setdefault("hit_rate", hit_rate)
    aggregated.setdefault("mrr", mrr)
    return EvalReport(
        hit_rate=hit_rate,
        mrr=mrr,
        case_count=len(results),
        cases=results,
        metrics=aggregated,
    )


def _mean(values: Sequence[float]) -> float:
    """空序列视为 0，避免除零。"""
    if not values:
        return 0.0
    return sum(values) / len(values)
