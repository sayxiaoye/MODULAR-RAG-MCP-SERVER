"""MetadataEnricher：为 Chunk 生成 title/summary/tags，支持 LLM 增强与降级。"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Sequence

from core.settings import REPO_ROOT, Settings, resolve_path
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform, TransformError
from libs.llm.base_llm import BaseLLM, ChatMessage, LLMError
from libs.llm.llm_factory import LLMFactory

logger = logging.getLogger(__name__)

DEFAULT_METADATA_PROMPT_PATH = REPO_ROOT / "config" / "prompts" / "metadata_enrichment.txt"
_FALLBACK_DEFAULT_PROMPT = (
    "Analyze the text chunk below and respond with JSON only using this schema:\n"
    '{"title": "short title", "summary": "one-sentence summary", "tags": ["tag1", "tag2"]}\n\n'
    "{text}\n"
)
_HEADING_PATTERN = re.compile(r"^#+\s*(.+)$")
_HASHTAG_PATTERN = re.compile(r"#(\w+)")
_WORD_PATTERN = re.compile(r"[A-Za-z\u4e00-\u9fff]{3,}")


def load_metadata_enrichment_prompt(path: str | Path | None = None) -> str:
    """加载元数据增强 prompt 模板，必须含 ``{text}`` 占位符。"""
    prompt_path = Path(path) if path is not None else DEFAULT_METADATA_PROMPT_PATH
    if not prompt_path.is_absolute():
        prompt_path = resolve_path(prompt_path)
    if not prompt_path.is_file():
        raise TransformError(f"metadata enrichment prompt 文件不存在: {prompt_path}")
    template = prompt_path.read_text(encoding="utf-8")
    if "{text}" not in template:
        raise TransformError("metadata enrichment prompt 必须包含 {text} 占位符")
    return template


class MetadataEnricher(BaseTransform):
    """规则生成 title/summary/tags，可选 LLM 语义增强，失败时降级。"""

    def __init__(
        self,
        settings: Settings,
        llm: BaseLLM | None = None,
        prompt_path: str | Path | None = None,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._prompt_path = prompt_path
        self._prompt_template: str | None = None

    def transform(
        self,
        chunks: Sequence[Chunk],
        trace: Any | None = None,
    ) -> list[Chunk]:
        """逐 Chunk 增强 metadata，单条失败时尽量保留规则结果。"""
        enriched: list[Chunk] = []
        for chunk in chunks:
            try:
                enriched.append(self._enrich_single_chunk(chunk, trace))
            except Exception as exc:
                logger.warning("元数据增强失败，保留原 Chunk: %s", exc)
                enriched.append(chunk)
        return enriched

    def _enrich_single_chunk(self, chunk: Chunk, trace: Any | None) -> Chunk:
        rule_start = time.perf_counter()
        rule_fields = self._rule_based_enrich(chunk.text)
        rule_ms = (time.perf_counter() - rule_start) * 1000
        self._record_trace(
            trace,
            "metadata_enricher_rule",
            elapsed_ms=rule_ms,
            method="rule",
            chunk_id=chunk.id,
        )

        fields = dict(rule_fields)
        metadata = dict(chunk.metadata)
        metadata["enriched_by"] = "rule"

        if self._use_llm():
            llm_start = time.perf_counter()
            llm_fields = self._llm_enrich(chunk.text, trace)
            llm_ms = (time.perf_counter() - llm_start) * 1000
            if llm_fields is not None:
                fields = llm_fields
                metadata["enriched_by"] = "llm"
                self._record_trace(
                    trace,
                    "metadata_enricher_llm",
                    elapsed_ms=llm_ms,
                    method="llm",
                    chunk_id=chunk.id,
                )
            else:
                metadata["enrichment_fallback"] = "llm_failed_or_invalid"
                self._record_trace(
                    trace,
                    "metadata_enricher_llm",
                    elapsed_ms=llm_ms,
                    method="rule",
                    fallback=True,
                    chunk_id=chunk.id,
                )

        metadata["title"] = fields["title"]
        metadata["summary"] = fields["summary"]
        metadata["tags"] = fields["tags"]
        return self._build_enriched_chunk(chunk, metadata)

    def _rule_based_enrich(self, text: str) -> dict[str, Any]:
        """规则兜底：从标题行/首行生成 title，截断文本作 summary，提取 tags。"""
        stripped = text.strip()
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]

        title = "Untitled"
        body_lines: list[str] = []
        if lines:
            first = lines[0]
            heading_match = _HEADING_PATTERN.match(first)
            if heading_match:
                title = heading_match.group(1).strip() or "Untitled"
                body_lines = lines[1:]
            else:
                title = first[:80].strip() or "Untitled"
                body_lines = lines[1:] if len(lines) > 1 else [first]

        summary_source = "\n".join(body_lines).strip() if body_lines else stripped
        summary = (summary_source[:200].strip() or title).strip()
        tags = self._extract_rule_tags(stripped)
        if not tags:
            tags = ["general"]

        return {
            "title": title,
            "summary": summary,
            "tags": tags,
        }

    def _extract_rule_tags(self, text: str) -> list[str]:
        """从 Markdown 标签或关键词提取 tags 列表。"""
        hashtags = _HASHTAG_PATTERN.findall(text)
        if hashtags:
            return hashtags[:5]

        seen: list[str] = []
        for word in _WORD_PATTERN.findall(text):
            token = word.lower()
            if token not in seen:
                seen.append(token)
            if len(seen) >= 5:
                break
        return seen

    def _llm_enrich(self, text: str, trace: Any | None) -> dict[str, Any] | None:
        """调用 LLM 生成结构化 metadata；失败返回 None 触发降级。"""
        llm = self._llm
        if llm is None:
            try:
                llm = LLMFactory.create(self._settings)
            except LLMError as exc:
                logger.warning("无法创建 LLM，跳过元数据 LLM 增强: %s", exc)
                return None

        prompt = self._load_prompt().replace("{text}", text)
        try:
            response = llm.chat([ChatMessage(role="user", content=prompt)], trace=trace)
            return self._parse_llm_payload(response.content)
        except (LLMError, TransformError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("LLM 元数据增强失败，将回退规则结果: %s", exc)
            return None

    def _parse_llm_payload(self, content: str) -> dict[str, Any]:
        """解析 LLM JSON 响应并校验 title/summary/tags。"""
        text = content.strip()
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1)
        else:
            brace_match = re.search(r"\{.*\}", text, re.DOTALL)
            if brace_match:
                text = brace_match.group(0)

        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise TransformError("LLM metadata 响应必须是 JSON 对象")

        title = str(payload.get("title", "")).strip()
        summary = str(payload.get("summary", "")).strip()
        raw_tags = payload.get("tags")
        if not title or not summary:
            raise TransformError("LLM metadata 缺少 title 或 summary")
        if not isinstance(raw_tags, list) or not raw_tags:
            raise TransformError("LLM metadata tags 必须是非空列表")

        tags = [str(tag).strip() for tag in raw_tags if str(tag).strip()]
        if not tags:
            raise TransformError("LLM metadata tags 不能为空")

        return {"title": title, "summary": summary, "tags": tags[:10]}

    def _load_prompt(self) -> str:
        if self._prompt_template is not None:
            return self._prompt_template
        try:
            self._prompt_template = load_metadata_enrichment_prompt(self._prompt_path)
        except TransformError:
            self._prompt_template = _FALLBACK_DEFAULT_PROMPT
        return self._prompt_template

    def _use_llm(self) -> bool:
        ingestion = self._settings.ingestion
        if ingestion is None:
            return False
        enricher_cfg = ingestion.metadata_enricher or {}
        return bool(enricher_cfg.get("use_llm", False))

    @staticmethod
    def _build_enriched_chunk(chunk: Chunk, metadata: dict[str, Any]) -> Chunk:
        enriched = Chunk(
            id=chunk.id,
            text=chunk.text,
            metadata=metadata,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            source_ref=chunk.source_ref,
        )
        enriched.validate()
        return enriched

    @staticmethod
    def _record_trace(
        trace: Any | None,
        name: str,
        *,
        elapsed_ms: float,
        **details: Any,
    ) -> None:
        if isinstance(trace, TraceContext):
            trace.record_stage(name, elapsed_ms=elapsed_ms, **details)
