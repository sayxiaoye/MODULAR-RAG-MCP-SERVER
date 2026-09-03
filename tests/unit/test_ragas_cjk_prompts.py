"""Ragas 中日 Judge prompt 与拆句补丁。"""

from __future__ import annotations

import pytest

pytest.importorskip("ragas")

from ragas.metrics import Faithfulness

from observability.evaluation.ragas_cjk_prompts import (
    apply_cjk_judge_prompts,
    patch_cjk_statement_split,
)


@pytest.mark.unit
class TestRagasCjkPrompts:
    """验证 Faithfulness 换上含日语的 few-shot，且 CJK 走规则拆句。"""

    def test_faithfulness_examples_include_japanese(self) -> None:
        """prompt 示例应出现憂鬱（ゆううつ），避免 7B 只模仿英文 JSON。"""
        metric = Faithfulness()
        apply_cjk_judge_prompts(metric)
        blob = " ".join(str(prompt) for prompt in metric.get_prompts().values())
        assert "憂鬱（ゆううつ）" in blob
        assert "只输出一个 JSON" in blob or "双引号" in blob

    def test_cjk_statement_split_skips_llm(self) -> None:
        """含日语的答案不应再调用 LLM 拆句。"""
        import asyncio

        from ragas.metrics._faithfulness import StatementGeneratorOutput

        metric = Faithfulness()
        patch_cjk_statement_split(metric)

        async def _should_not_run(*args: object, **kwargs: object) -> None:
            raise AssertionError("CJK 拆句不应回退 LLM")

        metric.statement_generator_prompt.generate = _should_not_run  # type: ignore[method-assign]
        result = asyncio.run(
            metric._create_statements(
                {
                    "user_input": "忧郁是什么意思？",
                    "response": "忧郁写作憂鬱（ゆううつ）。表示心情低落。",
                },
                callbacks=[],
            )
        )
        assert isinstance(result, StatementGeneratorOutput)
        assert any("憂鬱（ゆううつ）" in item for item in result.statements)
        assert len(result.statements) == 2
