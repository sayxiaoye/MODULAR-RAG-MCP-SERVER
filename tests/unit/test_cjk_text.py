"""中日文拆句与 Judge 文本规范化。"""

from __future__ import annotations

import pytest

from observability.evaluation.cjk_text import (
    contains_cjk,
    normalize_judge_text,
    sanitize_judge_output,
    split_cjk_statements,
    split_sentences,
)


@pytest.mark.unit
class TestCjkText:
    """验证日语读音括号不被切开，英文不走规则拆句。"""

    def test_split_sentences_keeps_period_inside_parens(self) -> None:
        """括号内的句号不是句界。"""
        parts = split_sentences("说明（内部。不要切）后面。下一句。")
        assert parts == ["说明（内部。不要切）后面。", "下一句。"]

    def test_keeps_furigana_with_kanji(self) -> None:
        """憂鬱（ゆううつ）应与汉字留在同一条陈述。"""
        answer = "忧郁在日语里写作憂鬱（ゆううつ），表示心情低落、闷闷不乐。"
        parts = split_cjk_statements(answer)
        assert len(parts) == 1
        assert "憂鬱（ゆううつ）" in parts[0]

    def test_splits_on_ideographic_period(self) -> None:
        """句号应拆成两条。"""
        parts = split_cjk_statements("第一句。第二句。")
        assert parts == ["第一句。", "第二句。"]

    def test_english_returns_empty_for_llm_fallback(self) -> None:
        """纯英文应交回 LLM 拆句。"""
        assert split_cjk_statements("Albert Einstein was born in Germany.") == []
        assert contains_cjk("hello") is False

    def test_normalize_replaces_corner_brackets(self) -> None:
        """中日引号换成双引号，假名保留。"""
        text = normalize_judge_text("「憂鬱（ゆううつ）」")
        assert "「" not in text
        assert "ゆううつ" in text

    def test_sanitize_output_quotes(self) -> None:
        """模型误用「」时应能换成 JSON 双引号。"""
        assert '"verdict"' in sanitize_judge_output("「verdict」")
