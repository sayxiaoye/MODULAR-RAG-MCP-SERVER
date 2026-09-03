"""给 Ragas Judge 换成中文指令 + 含日语的 few-shot，避免 7B 模仿英文示例却吐不出 JSON。"""

from __future__ import annotations

from typing import Any

_JSON_RULES = (
    "只输出一个 JSON 对象，不要 Markdown、不要前言。"
    "字符串必须用英文双引号 \", 禁止「」『』和单引号。"
    "日文汉字、平假名、片假名以及括号读音（如憂鬱（ゆううつ））原样写入字符串，不要翻译成罗马字。"
    "JSON 的 key 必须与 schema 完全一致（英文）。"
)


def apply_cjk_judge_prompts(metric: Any) -> None:
    """把指标上的英文 PydanticPrompt 换成中日 few-shot 版本。"""
    if not hasattr(metric, "get_prompts") or not hasattr(metric, "set_prompts"):
        return
    library = _cjk_prompt_library()
    replacements: dict[str, Any] = {}
    for name, prompt in metric.get_prompts().items():
        cjk = library.get(type(prompt).__name__)
        if cjk is None:
            continue
        cjk.name = name
        replacements[name] = cjk
    if replacements:
        metric.set_prompts(**replacements)


def patch_cjk_statement_split(metric: Any) -> None:
    """中日答案用规则拆句，避免 7B 在「（ゆううつ）」处切碎或整段当一句。"""
    if getattr(metric, "name", "") != "faithfulness":
        return
    if not hasattr(metric, "_create_statements"):
        return

    original = metric._create_statements

    async def _create_statements(row: dict[str, Any], callbacks: Any) -> Any:
        from ragas.metrics._faithfulness import StatementGeneratorOutput

        from observability.evaluation.cjk_text import split_cjk_statements

        statements = split_cjk_statements(str(row.get("response") or ""))
        if statements:
            return StatementGeneratorOutput(statements=statements)
        return await original(row, callbacks)

    metric._create_statements = _create_statements


def _cjk_prompt_library() -> dict[str, Any]:
    """按 ragas 原 Prompt 类名提供替换实例。"""
    from ragas.metrics._answer_relevance import (
        ResponseRelevanceInput,
        ResponseRelevanceOutput,
        ResponseRelevancePrompt,
    )
    from ragas.metrics._context_precision import (
        ContextPrecisionPrompt,
        QAC,
        Verification,
    )
    from ragas.metrics._faithfulness import (
        NLIStatementInput,
        NLIStatementOutput,
        NLIStatementPrompt,
        StatementFaithfulnessAnswer,
        StatementGeneratorInput,
        StatementGeneratorOutput,
        StatementGeneratorPrompt,
    )

    class CjkStatementGeneratorPrompt(StatementGeneratorPrompt):
        instruction = (
            "根据问题和答案，把答案拆成若干条独立、完整、不含代词的陈述。"
            "中文和日文按句号（。）、问号、感叹号或换行拆分；"
            "括号里的读音必须和汉字留在同一条，例如「憂鬱（ゆううつ）」不可从「（」切开。"
            + _JSON_RULES
        )
        examples = [
            (
                StatementGeneratorInput(
                    question="忧郁是什么意思？",
                    answer="忧郁在日语里写作憂鬱（ゆううつ），表示心情低落、闷闷不乐。",
                ),
                StatementGeneratorOutput(
                    statements=[
                        "忧郁在日语里写作憂鬱（ゆううつ）。",
                        "憂鬱表示心情低落、闷闷不乐。",
                    ]
                ),
            ),
            (
                StatementGeneratorInput(
                    question="Who was Albert Einstein and what is he best known for?",
                    answer=(
                        "He was a German-born theoretical physicist, widely acknowledged "
                        "to be one of the greatest and most influential physicists of all time."
                    ),
                ),
                StatementGeneratorOutput(
                    statements=[
                        "Albert Einstein was a German-born theoretical physicist.",
                        "Albert Einstein is recognized as one of the greatest physicists of all time.",
                    ]
                ),
            ),
        ]

    class CjkNLIStatementPrompt(NLIStatementPrompt):
        instruction = (
            "根据给定上下文，判断每条陈述是否能由上下文直接推出。"
            "能直接推出则 verdict 为 1，否则为 0。"
            "statement 必须与输入逐字一致（含日文）。reason 可用中文简述。"
            + _JSON_RULES
        )
        examples = [
            (
                NLIStatementInput(
                    context="憂鬱（ゆううつ）：心情低落、闷闷不乐。N2 词汇。",
                    statements=[
                        "憂鬱（ゆううつ）表示心情低落。",
                        "憂鬱是一种食物。",
                    ],
                ),
                NLIStatementOutput(
                    statements=[
                        StatementFaithfulnessAnswer(
                            statement="憂鬱（ゆううつ）表示心情低落。",
                            reason="上下文写明憂鬱表示心情低落、闷闷不乐。",
                            verdict=1,
                        ),
                        StatementFaithfulnessAnswer(
                            statement="憂鬱是一种食物。",
                            reason="上下文没有把憂鬱说成食物。",
                            verdict=0,
                        ),
                    ]
                ),
            ),
            (
                NLIStatementInput(
                    context="Photosynthesis is a process used by plants to convert light energy into chemical energy.",
                    statements=["Albert Einstein was a genius."],
                ),
                NLIStatementOutput(
                    statements=[
                        StatementFaithfulnessAnswer(
                            statement="Albert Einstein was a genius.",
                            reason="上下文与陈述无关。",
                            verdict=0,
                        )
                    ]
                ),
            ),
        ]

    class CjkResponseRelevancePrompt(ResponseRelevancePrompt):
        instruction = (
            "根据给定答案反推一个问题，并判断答案是否含糊其辞。"
            "noncommittal=1 表示答案在回避（例如不知道、不确定），否则为 0。"
            + _JSON_RULES
        )
        examples = [
            (
                ResponseRelevanceInput(
                    response="忧郁在日语里写作憂鬱（ゆううつ），表示心情低落。"
                ),
                ResponseRelevanceOutput(question="忧郁是什么意思？", noncommittal=0),
            ),
            (
                ResponseRelevanceInput(response="Albert Einstein was born in Germany."),
                ResponseRelevanceOutput(
                    question="Where was Albert Einstein born?",
                    noncommittal=0,
                ),
            ),
            (
                ResponseRelevanceInput(response="I don't know."),
                ResponseRelevanceOutput(
                    question="What is the answer?",
                    noncommittal=1,
                ),
            ),
        ]

    class CjkContextPrecisionPrompt(ContextPrecisionPrompt):
        instruction = (
            "给定问题、答案和一段检索上下文，判断这段上下文是否有助于得出该答案。"
            "有帮助则 verdict=1，否则 0。"
            + _JSON_RULES
        )
        examples = [
            (
                QAC(
                    question="忧郁是什么意思？",
                    context="憂鬱（ゆううつ）：心情低落、闷闷不乐。",
                    answer="忧郁在日语里写作憂鬱（ゆううつ），表示心情低落。",
                ),
                Verification(
                    reason="上下文给出了憂鬱的读音和释义，足以支持答案。",
                    verdict=1,
                ),
            ),
            (
                QAC(
                    question="What is the tallest mountain in the world?",
                    context="The Andes is the longest continental mountain range in South America.",
                    answer="Mount Everest.",
                ),
                Verification(
                    reason="上下文讲安第斯山脉，与珠峰无关。",
                    verdict=0,
                ),
            ),
        ]

    return {
        "StatementGeneratorPrompt": CjkStatementGeneratorPrompt(),
        "NLIStatementPrompt": CjkNLIStatementPrompt(),
        "ResponseRelevancePrompt": CjkResponseRelevancePrompt(),
        "ContextPrecisionPrompt": CjkContextPrecisionPrompt(),
    }
