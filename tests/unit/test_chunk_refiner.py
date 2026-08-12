"""ChunkRefiner 单元测试：规则去噪、LLM mock 与降级行为。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.settings import IngestionSettings, Settings, load_settings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from ingestion.transform.chunk_refiner import ChunkRefiner, load_chunk_refinement_prompt
from libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse, LLMError


FIXTURES_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "noisy_chunks.json"


class FakeLLM(BaseLLM):
    """测试用 LLM：返回预设文本或模拟失败。"""

    def __init__(self, content: str = "LLM refined text", should_fail: bool = False) -> None:
        self.content = content
        self.should_fail = should_fail
        self.last_messages: list[ChatMessage] | None = None

    def chat(self, messages, trace=None) -> ChatResponse:
        self.last_messages = list(messages)
        if self.should_fail:
            raise LLMError("mock llm failure")
        return ChatResponse(content=self.content, model="fake-llm")


def _ingestion(**overrides: object) -> IngestionSettings:
    base = load_settings()
    ingestion_data = {
        "chunk_size": base.ingestion.chunk_size,
        "chunk_overlap": base.ingestion.chunk_overlap,
        "splitter": base.ingestion.splitter,
        "batch_size": base.ingestion.batch_size,
        "chunk_refiner": base.ingestion.chunk_refiner,
    }
    ingestion_data.update(overrides)
    return IngestionSettings(**ingestion_data)


def _settings(**ingestion_overrides: object) -> Settings:
    base = load_settings()
    ingestion = _ingestion(**ingestion_overrides)
    return Settings(
        llm=base.llm,
        embedding=base.embedding,
        vector_store=base.vector_store,
        retrieval=base.retrieval,
        rerank=base.rerank,
        evaluation=base.evaluation,
        observability=base.observability,
        ingestion=ingestion,
        vision_llm=base.vision_llm,
    )


def _chunk(text: str, chunk_id: str = "doc_0000_abcd1234") -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={"source_path": "sample.pdf", "chunk_index": 0},
        start_offset=10,
        end_offset=10 + len(text),
        source_ref="doc-hash",
    )


@pytest.fixture
def noisy_fixtures() -> dict[str, dict[str, object]]:
    return json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))


@pytest.mark.unit
class TestChunkRefinementPrompt:
    """验证 prompt 加载与占位符契约。"""

    def test_load_default_prompt_contains_text_placeholder(self) -> None:
        template = load_chunk_refinement_prompt()
        assert "{text}" in template

    def test_load_custom_prompt_path(self, tmp_path: Path) -> None:
        custom = tmp_path / "custom_refine.txt"
        custom.write_text("Refine: {text}", encoding="utf-8")
        template = load_chunk_refinement_prompt(custom)
        assert "{text}" in template

    def test_load_prompt_without_text_placeholder_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.txt"
        bad.write_text("no placeholder", encoding="utf-8")
        with pytest.raises(Exception, match="text"):
            load_chunk_refinement_prompt(bad)


@pytest.mark.unit
class TestRuleBasedRefineFixtures:
    """对 8 个噪声 fixture 验证规则去噪效果。"""

    @pytest.mark.parametrize(
        "scenario",
        [
            "typical_noise_scenario",
            "ocr_errors",
            "page_header_footer",
            "excessive_whitespace",
            "format_markers",
            "clean_text",
            "code_blocks",
            "mixed_noise",
        ],
    )
    def test_fixture_scenarios(self, noisy_fixtures: dict, scenario: str) -> None:
        refiner = ChunkRefiner(_settings())
        case = noisy_fixtures[scenario]
        input_text = str(case["input"])
        output = refiner._rule_based_refine(input_text)

        if "must_equal" in case:
            assert output == case["must_equal"]
        for token in case.get("must_contain", []):
            assert str(token) in output
        for token in case.get("must_not_contain", []):
            assert str(token) not in output
        if case.get("collapse_spaces"):
            assert "  " not in output
        max_nl = case.get("max_newline_run")
        if max_nl is not None:
            assert "\n\n\n" not in output


@pytest.mark.unit
class TestChunkRefinerTransform:
    """验证 transform 管线、LLM 模式与降级。"""

    def test_rule_only_when_use_llm_false(self) -> None:
        chunk = _chunk("Page 1 of 2\n正文")
        refiner = ChunkRefiner(_settings(chunk_refiner={"use_llm": False}))
        result = refiner.transform([chunk])[0]
        assert result.metadata["refined_by"] == "rule"
        assert "Page 1" not in result.text

    def test_llm_success_marks_refined_by_llm(self) -> None:
        chunk = _chunk("噪声文本")
        refiner = ChunkRefiner(
            _settings(chunk_refiner={"use_llm": True}),
            llm=FakeLLM("精炼后文本"),
        )
        result = refiner.transform([chunk])[0]
        assert result.text == "精炼后文本"
        assert result.metadata["refined_by"] == "llm"

    def test_llm_failure_fallback_to_rule(self) -> None:
        chunk = _chunk("Page 1 of 2\n保留内容")
        refiner = ChunkRefiner(
            _settings(chunk_refiner={"use_llm": True}),
            llm=FakeLLM(should_fail=True),
        )
        result = refiner.transform([chunk])[0]
        assert result.metadata["refined_by"] == "rule"
        assert result.metadata.get("refinement_fallback") == "llm_failed_or_empty"
        assert "保留内容" in result.text
        assert "Page 1" not in result.text

    def test_llm_empty_response_fallback(self) -> None:
        chunk = _chunk("正文内容")
        refiner = ChunkRefiner(
            _settings(chunk_refiner={"use_llm": True}),
            llm=FakeLLM("   "),
        )
        result = refiner.transform([chunk])[0]
        assert result.metadata["refined_by"] == "rule"
        assert result.metadata.get("refinement_fallback") == "llm_failed_or_empty"

    def test_single_chunk_rule_failure_keeps_original(self) -> None:
        """规则阶段若出现异常，transform 应保留原 Chunk。"""
        chunk = _chunk("正常文本")
        refiner = ChunkRefiner(_settings())

        def _boom(text: str) -> str:
            if "BOOM" in text:
                raise RuntimeError("rule failed")
            return text

        refiner._rule_based_refine = _boom  # type: ignore[method-assign]
        broken = _chunk("BOOM")
        results = refiner.transform([chunk, broken])
        assert results[0].text == "正常文本"
        assert results[1].text == "BOOM"

    def test_transform_preserves_ids_and_source_ref(self) -> None:
        chunk = _chunk("Page 1\n内容", "chunk_id_fixed")
        refiner = ChunkRefiner(_settings())
        result = refiner.transform([chunk])[0]
        assert result.id == "chunk_id_fixed"
        assert result.source_ref == "doc-hash"

    def test_transform_updates_end_offset(self) -> None:
        chunk = _chunk("Page 1 of 2\n短文本")
        refiner = ChunkRefiner(_settings())
        result = refiner.transform([chunk])[0]
        assert result.end_offset == chunk.start_offset + len(result.text)

    def test_trace_records_stages(self) -> None:
        trace = TraceContext(trace_type="ingestion")
        refiner = ChunkRefiner(
            _settings(chunk_refiner={"use_llm": True}),
            llm=FakeLLM("ok"),
        )
        refiner.transform([_chunk("Page 1\n内容")], trace=trace)
        summary = trace.finish()
        stage_names = [stage["name"] for stage in summary["stages"]]
        assert "chunk_refiner_rule" in stage_names
        assert "chunk_refiner_llm" in stage_names

    def test_empty_chunks_list(self) -> None:
        refiner = ChunkRefiner(_settings())
        assert refiner.transform([]) == []

    def test_image_placeholder_preserved_in_rule(self) -> None:
        text = "说明 [IMAGE: img_1] 结束"
        refiner = ChunkRefiner(_settings())
        output = refiner._rule_based_refine(text)
        assert "[IMAGE: img_1]" in output

    def test_llm_receives_formatted_prompt(self) -> None:
        fake = FakeLLM("done")
        refiner = ChunkRefiner(
            _settings(chunk_refiner={"use_llm": True}),
            llm=fake,
        )
        refiner.transform([_chunk("原始 chunk")])
        assert fake.last_messages is not None
        assert "原始 chunk" in fake.last_messages[0].content

    def test_base_transform_is_abstract(self) -> None:
        assert issubclass(ChunkRefiner, BaseTransform)
        class Incomplete(BaseTransform):
            pass
        with pytest.raises(TypeError):
            Incomplete()

    def test_batch_multiple_chunks(self) -> None:
        chunks = [_chunk("Page 1 of 2\nA"), _chunk("Page 2 of 3\nB", "chunk_b")]
        refiner = ChunkRefiner(_settings())
        results = refiner.transform(chunks)
        assert len(results) == 2
        assert all("Page" not in item.text for item in results)

    def test_rule_refine_empty_string(self) -> None:
        refiner = ChunkRefiner(_settings())
        assert refiner._rule_based_refine("") == ""

    def test_use_llm_false_when_no_ingestion_chunk_refiner(self) -> None:
        base = load_settings()
        ingestion = IngestionSettings(
            chunk_size=base.ingestion.chunk_size,
            chunk_overlap=base.ingestion.chunk_overlap,
            splitter=base.ingestion.splitter,
            batch_size=base.ingestion.batch_size,
            chunk_refiner=None,
        )
        settings = Settings(
            llm=base.llm,
            embedding=base.embedding,
            vector_store=base.vector_store,
            retrieval=base.retrieval,
            rerank=base.rerank,
            evaluation=base.evaluation,
            observability=base.observability,
            ingestion=ingestion,
            vision_llm=base.vision_llm,
        )
        refiner = ChunkRefiner(settings)
        assert refiner._use_llm() is False

    def test_llm_refine_direct_returns_none_on_failure(self) -> None:
        refiner = ChunkRefiner(
            _settings(chunk_refiner={"use_llm": True}),
            llm=FakeLLM(should_fail=True),
        )
        assert refiner._llm_refine("text", None) is None

    def test_load_missing_prompt_raises(self, tmp_path: Path) -> None:
        with pytest.raises(Exception, match="不存在"):
            load_chunk_refinement_prompt(tmp_path / "missing.txt")
