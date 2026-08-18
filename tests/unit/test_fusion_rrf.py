"""RRFFusion 单元测试：验证 RRF 融合分数与确定性排序。"""

from __future__ import annotations

import pytest

from core.settings import load_settings
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult
from core.query_engine.fusion import FusionError, RRFFusion


def _result(chunk_id: str, score: float = 0.0, text: str = "", metadata: dict | None = None) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        score=score,
        text=text or f"text-{chunk_id}",
        metadata=metadata or {"source_path": f"{chunk_id}.md"},
    )


@pytest.mark.unit
class TestRRFFusion:
    """验证 RRF 公式、top_k 截断与 k 参数可配置。"""

    def test_fuses_dense_and_sparse_deterministic(self) -> None:
        """双路输入应产出稳定排序，双路均靠前的 chunk 分数更高。"""
        settings = load_settings()
        fusion = RRFFusion(settings, rrf_k=60)
        dense = [_result("a"), _result("b"), _result("c")]
        sparse = [_result("b"), _result("c"), _result("d")]

        first = fusion.fuse([dense, sparse], top_k=10)
        second = fusion.fuse([dense, sparse], top_k=10)

        assert first == second
        assert [item.chunk_id for item in first] == ["b", "c", "a", "d"]
        assert first[0].score > first[1].score > first[2].score

    def test_custom_rrf_k_changes_scores(self) -> None:
        """较小 k 会放大头部排名差异。"""
        settings = load_settings()
        dense = [_result("a"), _result("b")]
        sparse = [_result("b"), _result("a")]

        low_k = RRFFusion(settings, rrf_k=1).fuse([dense, sparse], top_k=2)
        high_k = RRFFusion(settings, rrf_k=100).fuse([dense, sparse], top_k=2)

        assert low_k[0].chunk_id == high_k[0].chunk_id == "a"
        assert low_k[0].score != high_k[0].score

    def test_single_list_still_scores_with_rrf(self) -> None:
        """仅一路结果时仍按 RRF 排名赋分。"""
        settings = load_settings()
        fusion = RRFFusion(settings, rrf_k=60)
        results = fusion.fuse([[_result("x"), _result("y")]], top_k=2)

        assert len(results) == 2
        assert results[0].chunk_id == "x"
        assert results[0].score == pytest.approx(1.0 / 61)
        assert results[1].score == pytest.approx(1.0 / 62)

    def test_empty_inputs_return_empty(self) -> None:
        settings = load_settings()
        fusion = RRFFusion(settings)
        assert fusion.fuse([[], []], top_k=5) == []
        assert fusion.fuse([], top_k=5) == []

    def test_top_k_limits_output(self) -> None:
        settings = load_settings()
        fusion = RRFFusion(settings, rrf_k=60)
        dense = [_result("a"), _result("b"), _result("c")]
        sparse = [_result("d"), _result("e")]

        results = fusion.fuse([dense, sparse], top_k=2)
        assert len(results) == 2

    def test_default_top_k_uses_settings(self) -> None:
        settings = load_settings()
        fusion = RRFFusion(settings, rrf_k=60)
        candidates = [_result(str(i)) for i in range(20)]

        results = fusion.fuse([candidates])
        assert len(results) == settings.retrieval.fusion_top_k

    def test_preserves_text_and_metadata(self) -> None:
        settings = load_settings()
        fusion = RRFFusion(settings, rrf_k=60)
        dense = [
            _result("a", text="dense-a", metadata={"source_path": "a.pdf", "lane": "dense"}),
        ]
        sparse = [
            _result("a", text="sparse-a", metadata={"source_path": "a.pdf", "lane": "sparse"}),
        ]

        results = fusion.fuse([dense, sparse], top_k=1)
        # 首次出现的记录用于回填正文
        assert results[0].text == "dense-a"
        assert results[0].metadata["lane"] == "dense"

    def test_records_trace_stage(self) -> None:
        settings = load_settings()
        fusion = RRFFusion(settings, rrf_k=60)
        trace = TraceContext(trace_type="query")
        fusion.fuse([[_result("a")]], top_k=1, trace=trace)

        stages = [stage["name"] for stage in trace.finish()["stages"]]
        assert "fusion" in stages

    def test_invalid_top_k_raises(self) -> None:
        settings = load_settings()
        fusion = RRFFusion(settings)
        with pytest.raises(FusionError, match="top_k"):
            fusion.fuse([[_result("a")]], top_k=0)

    def test_invalid_rrf_k_raises(self) -> None:
        settings = load_settings()
        with pytest.raises(FusionError, match="rrf_k"):
            RRFFusion(settings, rrf_k=0)
