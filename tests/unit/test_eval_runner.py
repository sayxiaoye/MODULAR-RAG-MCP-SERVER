"""EvalRunner 与 evaluate.py CLI 单元测试：mock 检索，验证报告与输出。"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from core.settings import load_settings
from core.types import RetrievalResult
from libs.evaluator.custom_evaluator import CustomEvaluator
from observability.evaluation.eval_runner import (
    EvalReport,
    EvalRunner,
    EvalRunnerError,
    load_golden_test_set,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "golden_test_set.json"
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from evaluate import build_arg_parser, run_evaluate  # noqa: E402


def _hit(
    chunk_id: str,
    *,
    source: str = "config_guide.pdf",
    text: str = "body",
) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=0.9,
        text=text,
        metadata={"source_path": source},
    )


class FakeHybridSearch:
    """按 query 返回预设检索结果；可对指定 query 抛错。"""

    def __init__(
        self,
        mapping: dict[str, list[RetrievalResult]],
        *,
        fail_on: str | None = None,
    ) -> None:
        self._mapping = mapping
        self._fail_on = fail_on
        self.queries: list[str] = []

    def search(
        self,
        query: str,
        top_k: int | None = None,
        filters: Any = None,
        trace: Any = None,
    ) -> list[RetrievalResult]:
        self.queries.append(query)
        if self._fail_on is not None and query == self._fail_on:
            raise RuntimeError("search boom")
        return list(self._mapping.get(query, []))


@pytest.mark.unit
class TestLoadGoldenTestSet:
    """验证黄金测试集加载与校验。"""

    def test_fixture_contains_spec_example_case(self) -> None:
        """仓库内 golden_test_set.json 应含 spec 示例 query。"""
        cases = load_golden_test_set(_FIXTURE)
        assert len(cases) >= 1
        assert cases[0]["query"] == "如何配置 Azure OpenAI？"
        assert "chunk_abc_001" in cases[0]["expected_chunk_ids"]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        """文件不存在时应失败。"""
        with pytest.raises(EvalRunnerError, match="找不到"):
            load_golden_test_set(tmp_path / "missing.json")

    def test_empty_cases_raises(self, tmp_path: Path) -> None:
        """test_cases 为空应失败。"""
        path = tmp_path / "empty.json"
        path.write_text(json.dumps({"test_cases": []}), encoding="utf-8")
        with pytest.raises(EvalRunnerError, match="test_cases"):
            load_golden_test_set(path)

    def test_missing_chunk_ids_raises(self, tmp_path: Path) -> None:
        """缺少 expected_chunk_ids 应失败。"""
        path = tmp_path / "bad.json"
        path.write_text(
            json.dumps({"test_cases": [{"query": "q", "expected_sources": ["a.pdf"]}]}),
            encoding="utf-8",
        )
        with pytest.raises(EvalRunnerError, match="expected_chunk_ids"):
            load_golden_test_set(path)


@pytest.mark.unit
class TestEvalRunner:
    """验证检索 + CustomEvaluator 产出含 hit_rate / mrr 的报告。"""

    def test_run_aggregates_hit_rate_and_mrr(self, tmp_path: Path) -> None:
        """两条用例：一条命中首位、一条未命中，宏观指标为平均值。"""
        path = tmp_path / "set.json"
        path.write_text(
            json.dumps(
                {
                    "test_cases": [
                        {
                            "query": "命中",
                            "expected_chunk_ids": ["gold-1"],
                            "expected_sources": ["a.pdf"],
                        },
                        {
                            "query": "未命中",
                            "expected_chunk_ids": ["gold-2"],
                            "expected_sources": ["b.pdf"],
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        search = FakeHybridSearch(
            {
                "命中": [_hit("gold-1", source="a.pdf")],
                "未命中": [_hit("other", source="x.pdf")],
            }
        )
        report = EvalRunner(load_settings(), search, CustomEvaluator()).run(path)
        assert isinstance(report, EvalReport)
        assert report.case_count == 2
        assert report.hit_rate == 0.5
        assert report.mrr == 0.5
        assert report.cases[0].metrics["hit_rate"] == 1.0
        assert report.cases[1].metrics["hit_rate"] == 0.0
        assert report.cases[0].retrieved_ids == ["gold-1"]
        assert "命中" in [item.query for item in report.cases]

    def test_search_error_isolated(self, tmp_path: Path) -> None:
        """单条检索失败不中断，该条 error 有记录，宏观 hit_rate 按 0 计入。"""
        path = tmp_path / "set.json"
        path.write_text(
            json.dumps(
                {
                    "test_cases": [
                        {
                            "query": "ok",
                            "expected_chunk_ids": ["gold-1"],
                            "expected_sources": [],
                        },
                        {
                            "query": "bad",
                            "expected_chunk_ids": ["gold-2"],
                            "expected_sources": [],
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )
        search = FakeHybridSearch(
            {"ok": [_hit("gold-1")]},
            fail_on="bad",
        )
        report = EvalRunner(load_settings(), search, CustomEvaluator()).run(path)
        assert report.case_count == 2
        assert report.cases[1].error is not None
        assert "检索失败" in report.cases[1].error
        assert report.hit_rate == 0.5


@pytest.mark.unit
class TestEvaluateCli:
    """验证 evaluate.py 可运行并输出 metrics。"""

    def test_build_arg_parser_defaults(self) -> None:
        """无参数时应使用默认黄金测试集。"""
        args = build_arg_parser().parse_args([])
        assert args.test_set is None
        assert args.json is False

    def test_run_evaluate_prints_metrics(self) -> None:
        """注入 Fake 检索器后 CLI 应打印 hit_rate / mrr。"""
        search = FakeHybridSearch(
            {
                "如何配置 Azure OpenAI？": [_hit("chunk_abc_001")],
                "Hybrid Search 如何融合 Dense 与 Sparse？": [_hit("chunk_hybrid_001")],
            }
        )
        buffer = io.StringIO()
        code = run_evaluate(
            test_set=str(_FIXTURE),
            out=buffer,
            hybrid_search=search,
            evaluator=CustomEvaluator(),
        )
        output = buffer.getvalue()
        assert code == 0
        assert "hit_rate:" in output
        assert "mrr:" in output
        assert "cases: 2" in output

    def test_run_evaluate_missing_set_returns_error(self, tmp_path: Path) -> None:
        """测试集不存在时应返回退出码 1。"""
        buffer = io.StringIO()
        code = run_evaluate(
            test_set=str(tmp_path / "nope.json"),
            out=buffer,
            hybrid_search=FakeHybridSearch({}),
            evaluator=CustomEvaluator(),
        )
        assert code == 1
        assert "评估失败" in buffer.getvalue()
