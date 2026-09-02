"""黄金测试集生成：按集合取样 chunk、LLM/规则出题、落盘 JSON。"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Mapping

import pytest

from libs.llm.base_llm import ChatResponse
from observability.evaluation.eval_runner import load_golden_test_set
from observability.evaluation.golden_generator import (
    ChunkCandidate,
    GoldenGeneratorError,
    build_test_cases,
    fallback_query_from_text,
    generate_golden_test_set,
    parse_llm_query,
    records_to_candidates,
    sample_chunks,
    sanitize_collection_name,
    write_golden_test_set,
)


def _record(chunk_id: str, text: str, source: str, collection: str = "col_a") -> dict[str, Any]:
    return {
        "id": chunk_id,
        "text": text,
        "metadata": {"source_path": source, "collection": collection},
    }


class FakeStore:
    """只实现 get_by_metadata 的向量库替身。"""

    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = list(records)

    def get_by_metadata(
        self,
        filters: Mapping[str, Any] | None = None,
        trace: Any = None,
        *,
        collection: str | None = None,
    ) -> list[dict[str, Any]]:
        if not collection:
            return list(self.records)
        return [
            item
            for item in self.records
            if str(item.get("metadata", {}).get("collection") or "") == collection
        ]


class FakeLLM:
    """按调用次数返回预设问题。"""

    def __init__(self, answers: list[str] | None = None, *, fail: bool = False) -> None:
        self.answers = list(answers or [])
        self.fail = fail
        self.calls = 0

    def chat(self, messages, trace=None) -> ChatResponse:
        self.calls += 1
        if self.fail:
            raise RuntimeError("llm down")
        content = self.answers[self.calls - 1] if self.calls <= len(self.answers) else "默认问题？"
        return ChatResponse(content=content)


@pytest.mark.unit
class TestGoldenHelpers:
    """规则问句、取样与 JSON 落盘。"""

    def test_fallback_query_uses_first_sentence(self) -> None:
        """首句加问号；已是问句则保持。"""
        assert fallback_query_from_text("如何配置 Azure OpenAI：在门户创建资源。") == "如何配置 Azure OpenAI？"
        assert fallback_query_from_text("什么是 RRF？其余忽略") == "什么是 RRF？"

    def test_parse_llm_query_strips_quotes_and_numbering(self) -> None:
        """模型偶发带编号或引号时应抽出纯问题。"""
        assert parse_llm_query('1. "Hybrid Search 如何融合？"', fallback_text="x") == "Hybrid Search 如何融合？"

    def test_sample_chunks_spreads_across_sources(self) -> None:
        """N=3 时应尽量覆盖三个来源。"""
        candidates = [
            ChunkCandidate("a1", "t1", "doc_a.pdf", "col_a"),
            ChunkCandidate("a2", "t2", "doc_a.pdf", "col_a"),
            ChunkCandidate("b1", "t3", "doc_b.pdf", "col_a"),
            ChunkCandidate("c1", "t4", "doc_c.pdf", "col_a"),
        ]
        picked = sample_chunks(candidates, 3, rng=random.Random(0))
        assert len(picked) == 3
        sources = {item.source_path for item in picked}
        assert len(sources) == 3

    def test_sanitize_collection_name(self) -> None:
        """特殊字符应换成下划线。"""
        assert sanitize_collection_name("new 456!") == "new_456"

    def test_write_and_load_roundtrip(self, tmp_path: Path) -> None:
        """写出的 JSON 应能被 EvalRunner 加载。"""
        path = tmp_path / "golden_col_a.json"
        write_golden_test_set(
            [
                {
                    "query": "如何配置 Azure OpenAI？",
                    "expected_chunk_ids": ["real-hash-1"],
                    "expected_sources": ["guide.pdf"],
                }
            ],
            path,
            collection="col_a",
        )
        cases = load_golden_test_set(path)
        assert cases[0]["expected_chunk_ids"] == ["real-hash-1"]
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["collection"] == "col_a"


@pytest.mark.unit
class TestGenerateGoldenTestSet:
    """generate_golden_test_set 使用真实 chunk id。"""

    def test_uses_real_ids_and_llm_queries(self, tmp_path: Path) -> None:
        """落盘 expected_chunk_ids 必须来自向量库，query 来自 LLM。"""
        store = FakeStore(
            [
                _record("hash-azure", "如何配置 Azure OpenAI：填写 endpoint。", "config.pdf"),
                _record("hash-rrf", "Hybrid Search 使用 RRF 融合 Dense 与 Sparse。", "retr.pdf"),
            ]
        )
        llm = FakeLLM(["Azure 要怎么配？", "RRF 怎么融合两路召回？"])
        path = generate_golden_test_set(
            "col_a",
            2,
            vector_store=store,
            llm=llm,
            output_path=tmp_path / "out.json",
            rng=random.Random(1),
        )
        cases = load_golden_test_set(path)
        assert len(cases) == 2
        ids = {item["expected_chunk_ids"][0] for item in cases}
        assert ids == {"hash-azure", "hash-rrf"}
        queries = {item["query"] for item in cases}
        assert "Azure 要怎么配？" in queries
        assert "RRF 怎么融合两路召回？" in queries
        assert llm.calls == 2

    def test_llm_failure_falls_back_to_rule_query(self, tmp_path: Path) -> None:
        """LLM 抛错时仍写出用例，query 为规则问句。"""
        store = FakeStore(
            [_record("hash-1", "BM25 如何用 jieba 做中文分词：切词后算 IDF。", "bm25.pdf")]
        )
        path = generate_golden_test_set(
            "col_a",
            1,
            vector_store=store,
            llm=FakeLLM(fail=True),
            output_path=tmp_path / "out.json",
        )
        cases = load_golden_test_set(path)
        assert cases[0]["expected_chunk_ids"] == ["hash-1"]
        assert cases[0]["query"].endswith("？")

    def test_empty_collection_raises(self, tmp_path: Path) -> None:
        """没有 chunk 时应给出明确错误。"""
        with pytest.raises(GoldenGeneratorError, match="没有可出题"):
            generate_golden_test_set(
                "empty_col",
                4,
                vector_store=FakeStore([]),
                ask=lambda text: "q",
                output_path=tmp_path / "out.json",
            )

    def test_ask_override_skips_llm(self, tmp_path: Path) -> None:
        """注入 ask 时不调用 llm。"""
        store = FakeStore([_record("id-1", "正文足够长可以出题。", "a.pdf")])
        llm = FakeLLM(["不应出现"])
        path = generate_golden_test_set(
            "col_a",
            1,
            vector_store=store,
            llm=llm,
            ask=lambda text: "自定义问题？",
            output_path=tmp_path / "out.json",
        )
        cases = load_golden_test_set(path)
        assert cases[0]["query"] == "自定义问题？"
        assert llm.calls == 0


@pytest.mark.unit
class TestBuildTestCases:
    """ask / llm 分支。"""

    def test_records_to_candidates_skips_empty_text(self) -> None:
        """无正文或无 id 的记录应丢弃。"""
        records = [
            _record("ok", "有正文", "a.pdf"),
            {"id": "no-text", "text": "  ", "metadata": {}},
            {"id": "", "text": "x", "metadata": {}},
        ]
        items = records_to_candidates(records, "col_a")
        assert [item.chunk_id for item in items] == ["ok"]

    def test_build_test_cases_with_ask(self) -> None:
        """ask 回调决定 query。"""
        samples = [ChunkCandidate("id-1", "chunk body", "src.pdf", "col_a")]
        cases = build_test_cases(samples, ask=lambda text: f"问：{text[:4]}")
        assert cases[0]["expected_chunk_ids"] == ["id-1"]
        assert cases[0]["query"].startswith("问：")
