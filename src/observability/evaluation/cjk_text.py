"""中日文句界与 Judge 文本规范化。句号拆句供 Faithfulness 与黄金集兜底共用。"""

from __future__ import annotations

import unicodedata

_SENTENCE_END = frozenset("。！？!?")
_OPEN_PARENS = frozenset("（(")
_CLOSE_PARENS = frozenset("）)")


def contains_cjk(text: str) -> bool:
    """文本是否含中日韩字符。"""
    return any(
        "\u3040" <= ch <= "\u30ff"
        or "\u3400" <= ch <= "\u9fff"
        or "\uac00" <= ch <= "\ud7af"
        for ch in text
    )


def split_sentences(text: str) -> list[str]:
    """
    按句号、问号、感叹号或换行拆句；括号内（含「憂鬱（ゆううつ）」）不切开。

    空字符串返回空列表。无句界时整段作为一句。
    """
    stripped = text.strip()
    if not stripped:
        return []

    parts: list[str] = []
    buf: list[str] = []
    depth = 0
    for ch in stripped:
        if ch in _OPEN_PARENS:
            depth += 1
            buf.append(ch)
        elif ch in _CLOSE_PARENS:
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch in _SENTENCE_END and depth == 0:
            buf.append(ch)
            piece = "".join(buf).strip()
            if piece:
                parts.append(piece)
            buf = []
        elif ch == "\n" and depth == 0:
            piece = "".join(buf).strip()
            if piece:
                parts.append(piece)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)
    return parts or [stripped]


def normalize_judge_text(text: str) -> str:
    """
    规范化送进 Judge 的文本：NFKC + 把中日引号换成 JSON 可用的双引号。

    不改变假名/汉字本身，避免把「憂鬱（ゆううつ）」罗马化。
    """
    normalized = unicodedata.normalize("NFKC", text)
    return (
        normalized.replace("「", '"')
        .replace("」", '"')
        .replace("『", '"')
        .replace("』", '"')
        .replace("“", '"')
        .replace("”", '"')
    )


def sanitize_judge_output(text: str) -> str:
    """把模型误用的中文引号换成 JSON 可用的双引号。"""
    return (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("「", '"')
        .replace("」", '"')
        .replace("『", '"')
        .replace("』", '"')
    )


def split_cjk_statements(text: str) -> list[str]:
    """
    中日答案拆成 Faithfulness 陈述；不含 CJK 时返回空列表，调用方回退 LLM 拆句。
    """
    stripped = text.strip()
    if not stripped or not contains_cjk(stripped):
        return []
    return split_sentences(stripped)
