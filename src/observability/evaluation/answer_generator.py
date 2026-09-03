"""评估链路用：根据 query + 检索上下文生成 RAG 答案，供 Ragas 打分。"""

from __future__ import annotations

from typing import Sequence

from libs.llm.base_llm import BaseLLM, LLMError

_ANSWER_PROMPT = (
    "根据下面检索到的资料回答用户问题。只依据资料作答，不要编造。"
    "若资料不足以回答，请明确说明不知道。\n\n"
    "问题：{query}\n\n"
    "资料：\n{contexts}\n\n"
    "答案："
)
_MAX_CONTEXT_CHARS = 1500
_MAX_CONTEXTS = 8


def generate_rag_answer(query: str, contexts: Sequence[str], llm: BaseLLM) -> str:
    """
    用项目 LLM 根据检索片段生成答案。

    Args:
        query: 用户问题。
        contexts: 检索到的 chunk 正文。
        llm: 项目 BaseLLM（会走 llamacpp 按需启停等）。

    Returns:
        模型生成的答案文本。

    Raises:
        LLMError: 调用失败或返回空文本。
    """
    if llm is None:
        raise LLMError("生成答案需要 LLM，请配置 settings.llm")
    prompt = _ANSWER_PROMPT.format(
        query=(query or "").strip() or "（空问题）",
        contexts=_format_contexts(contexts),
    )
    response = llm.chat([{"role": "user", "content": prompt}])
    content = (getattr(response, "content", None) or "").strip()
    if not content:
        raise LLMError("生成答案为空")
    return content


def _format_contexts(contexts: Sequence[str]) -> str:
    """把检索片段压成 prompt 中的资料块。"""
    blocks: list[str] = []
    for index, text in enumerate(contexts or (), start=1):
        if len(blocks) >= _MAX_CONTEXTS:
            break
        snippet = " ".join((text or "").split())
        if not snippet:
            continue
        if len(snippet) > _MAX_CONTEXT_CHARS:
            snippet = snippet[:_MAX_CONTEXT_CHARS] + "…"
        blocks.append(f"[{index}] {snippet}")
    if not blocks:
        return "（无检索资料）"
    return "\n\n".join(blocks)
