"""LLM Reranker：读取 rerank prompt，调用 LLM 对候选进行结构化重排。"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.settings import REPO_ROOT, RerankSettings, resolve_path
from libs.llm.base_llm import BaseLLM, ChatMessage, LLMError
from libs.reranker.base_reranker import BaseReranker, RerankerError, RerankerFallbackSignal

DEFAULT_RERANK_PROMPT_PATH = REPO_ROOT / "config" / "prompts" / "rerank.txt"
_JSON_OUTPUT_HINT = (
    "Respond with JSON only, using this schema: "
    '{"ranked_ids": ["candidate_id_1", "candidate_id_2", ...]}'
)


def load_rerank_prompt_template(path: str | Path | None = None) -> str:
    """
    从文件加载 rerank prompt 模板。

    Args:
        path: 模板路径；默认 ``config/prompts/rerank.txt``。

    Returns:
        含 ``{query}`` 与 ``{passages}`` 占位符的模板文本。
    """
    prompt_path = Path(path) if path is not None else DEFAULT_RERANK_PROMPT_PATH
    if not prompt_path.is_absolute():
        prompt_path = resolve_path(prompt_path)
    if not prompt_path.is_file():
        raise RerankerError(f"rerank prompt 文件不存在: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")


def _format_passages(candidates: Sequence[Mapping[str, Any]]) -> str:
    """将候选格式化为 prompt 中的 passages 段落。"""
    lines: list[str] = []
    for index, item in enumerate(candidates):
        lines.append(f"{index}. id={item['id']} | {item['text']}")
    return "\n".join(lines)


def _extract_json_payload(content: str) -> dict[str, Any]:
    """从 LLM 响应中提取 JSON 对象（容忍 markdown 代码块包裹）。"""
    text = content.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1)
    else:
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            text = brace_match.group(0)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RerankerError(f"LLM 响应不是合法 JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RerankerError("LLM 响应 JSON 必须是对象")
    return payload


def _parse_ranked_ids(
    payload: dict[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> list[str]:
    """解析 ranked_ids 或 ranked_indices，返回候选 id 列表。"""
    valid_ids = {str(item["id"]) for item in candidates}
    if "ranked_ids" in payload:
        raw_ids = payload["ranked_ids"]
        if not isinstance(raw_ids, list) or not raw_ids:
            raise RerankerError("ranked_ids 必须是非空列表")
        ranked = [str(item) for item in raw_ids]
    elif "ranked_indices" in payload:
        raw_indices = payload["ranked_indices"]
        if not isinstance(raw_indices, list) or not raw_indices:
            raise RerankerError("ranked_indices 必须是非空列表")
        ranked = []
        for index in raw_indices:
            if not isinstance(index, int) or index < 0 or index >= len(candidates):
                raise RerankerError(f"ranked_indices 含非法下标: {index}")
            ranked.append(str(candidates[index]["id"]))
    else:
        raise RerankerError("JSON 缺少 ranked_ids 或 ranked_indices 字段")

    unknown = [item for item in ranked if item not in valid_ids]
    if unknown:
        raise RerankerError(f"ranked_ids 含未知候选 id: {', '.join(unknown)}")
    return ranked


class LLMReranker(BaseReranker):
    """使用 LLM 对检索候选进行精排，输出严格结构化的 ranked ids。"""

    def __init__(
        self,
        settings: RerankSettings,
        llm: BaseLLM | None = None,
        prompt_template: str | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        self.settings = settings
        if llm is None:
            from core.settings import load_settings
            from libs.llm.llm_factory import LLMFactory

            self._llm = LLMFactory.create(load_settings())
        else:
            self._llm = llm
        if prompt_template is not None:
            self._prompt_template = prompt_template
        else:
            self._prompt_template = load_rerank_prompt_template(prompt_path)

    def _build_prompt(self, query: str, candidates: Sequence[Mapping[str, Any]]) -> str:
        """用 query 与 passages 填充 prompt 模板，并附加 JSON 输出约束。"""
        body = self._prompt_template.format(
            query=query,
            passages=_format_passages(candidates),
        )
        return f"{body.strip()}\n\n{_JSON_OUTPUT_HINT}"

    def rerank(
        self,
        query: str,
        candidates: Sequence[Mapping[str, Any]],
        trace: Any | None = None,
    ) -> list[dict[str, Any]]:
        validated_query = self._validate_query(query)
        validated_candidates = self._validate_candidates(candidates)
        if not validated_candidates:
            return []

        prompt = self._build_prompt(validated_query, validated_candidates)
        try:
            response = self._llm.chat(
                [ChatMessage(role="user", content=prompt)],
                trace=trace,
            )
        except LLMError as exc:
            raise RerankerFallbackSignal(
                f"[llm_reranker] LLM 调用失败，建议回退 fusion 排名: {exc}"
            ) from exc

        try:
            payload = _extract_json_payload(response.content)
            ranked_ids = _parse_ranked_ids(payload, validated_candidates)
        except RerankerError:
            raise
        except Exception as exc:
            raise RerankerFallbackSignal(
                f"[llm_reranker] 解析 LLM 响应失败，建议回退 fusion 排名: {exc}"
            ) from exc

        by_id = {item["id"]: item for item in validated_candidates}
        reranked = [by_id[item_id] for item_id in ranked_ids if item_id in by_id]

        # 补齐 LLM 未列出但仍在候选集中的 id（保持 fusion 相对顺序）
        seen = {item["id"] for item in reranked}
        for item in validated_candidates:
            if item["id"] not in seen:
                reranked.append(item)

        top_k = max(1, self.settings.top_k)
        return reranked[:top_k]
