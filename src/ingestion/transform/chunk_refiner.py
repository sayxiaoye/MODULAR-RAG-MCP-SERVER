"""ChunkRefiner：规则去噪 + 可选 LLM 增强，失败时降级不阻塞摄取。"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.settings import REPO_ROOT, Settings, resolve_path
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform, TransformError
from libs.llm.base_llm import BaseLLM, ChatMessage, LLMError
from libs.llm.llm_factory import LLMFactory

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_REFINEMENT_PROMPT_PATH = REPO_ROOT / "config" / "prompts" / "chunk_refinement.txt"

# 规则去噪用正则：页眉页脚、HTML 注释、多余空白等
_CODE_BLOCK_PATTERN = re.compile(r"```[\w-]*\n.*?```", re.DOTALL)
_HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)
_PAGE_HEADER_FOOTER_PATTERNS = [
    re.compile(r"^Page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\d+\s*/\s*\d+\s*$", re.MULTILINE),
    re.compile(r"^-{3,}\s*$", re.MULTILINE),
]
_COLLAPSE_SPACES_PATTERN = re.compile(r"[ \t]+")
_COLLAPSE_NEWLINES_PATTERN = re.compile(r"\n{3,}")


def load_chunk_refinement_prompt(path: str | Path | None = None) -> str:
    """
    加载 chunk 精炼 prompt 模板。

    Args:
        path: 模板路径，默认 ``config/prompts/chunk_refinement.txt``。

    Returns:
        含 ``{text}`` 占位符的模板字符串。
    """
    prompt_path = Path(path) if path is not None else DEFAULT_CHUNK_REFINEMENT_PROMPT_PATH
    if not prompt_path.is_absolute():
        prompt_path = resolve_path(prompt_path)
    if not prompt_path.is_file():
        raise TransformError(f"chunk refinement prompt 文件不存在: {prompt_path}")
    template = prompt_path.read_text(encoding="utf-8")
    if "{text}" not in template:
        raise TransformError("chunk refinement prompt 必须包含 {text} 占位符")
    return template


class ChunkRefiner(BaseTransform):
    """规则去噪优先，可选 LLM 二次精炼，LLM 失败回退规则结果。"""

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
        """逐 Chunk 精炼；单条失败时保留原文，不阻塞批次。"""
        refined: list[Chunk] = []
        for chunk in chunks:
            try:
                refined.append(self._refine_single_chunk(chunk, trace))
            except Exception as exc:
                logger.warning("Chunk 精炼失败，保留原文: %s", exc)
                refined.append(chunk)
        return refined

    def _refine_single_chunk(self, chunk: Chunk, trace: Any | None) -> Chunk:
        rule_start = time.perf_counter()
        rule_text = self._rule_based_refine(chunk.text)
        rule_ms = (time.perf_counter() - rule_start) * 1000
        self._record_trace(
            trace,
            "chunk_refiner_rule",
            elapsed_ms=rule_ms,
            method="rule",
            chunk_id=chunk.id,
        )

        metadata = dict(chunk.metadata)
        final_text = rule_text
        metadata["refined_by"] = "rule"

        if self._use_llm():
            llm_start = time.perf_counter()
            llm_text = self._llm_refine(rule_text, trace)
            llm_ms = (time.perf_counter() - llm_start) * 1000
            if llm_text is not None:
                final_text = llm_text
                metadata["refined_by"] = "llm"
                self._record_trace(
                    trace,
                    "chunk_refiner_llm",
                    elapsed_ms=llm_ms,
                    method="llm",
                    chunk_id=chunk.id,
                )
            else:
                metadata["refinement_fallback"] = "llm_failed_or_empty"
                self._record_trace(
                    trace,
                    "chunk_refiner_llm",
                    elapsed_ms=llm_ms,
                    method="rule",
                    fallback=True,
                    chunk_id=chunk.id,
                )

        return self._build_refined_chunk(chunk, final_text, metadata)

    def _use_llm(self) -> bool:
        ingestion = self._settings.ingestion
        if ingestion is None:
            return False
        refiner_cfg = ingestion.chunk_refiner or {}
        return bool(refiner_cfg.get("use_llm", False))

    def _rule_based_refine(self, text: str) -> str:
        """规则去噪：空白/页眉页脚/HTML 注释；代码块内容原样保留。"""
        if not text:
            return text

        code_blocks: dict[str, str] = {}

        def _stash_code_block(match: re.Match[str]) -> str:
            key = f"__CODE_BLOCK_{len(code_blocks)}__"
            code_blocks[key] = match.group(0)
            return key

        protected = _CODE_BLOCK_PATTERN.sub(_stash_code_block, text)
        protected = _HTML_COMMENT_PATTERN.sub("", protected)
        for pattern in _PAGE_HEADER_FOOTER_PATTERNS:
            protected = pattern.sub("", protected)
        protected = _COLLAPSE_SPACES_PATTERN.sub(" ", protected)
        protected = _COLLAPSE_NEWLINES_PATTERN.sub("\n\n", protected)
        protected = protected.strip()

        for key, block in code_blocks.items():
            protected = protected.replace(key, block)
        return protected.strip()

    def _llm_refine(self, text: str, trace: Any | None) -> str | None:
        """调用 LLM 重写文本；失败或空响应时返回 None 以触发降级。"""
        llm = self._llm
        if llm is None:
            try:
                llm = LLMFactory.create(self._settings)
            except LLMError as exc:
                logger.warning("无法创建 LLM，跳过 LLM 精炼: %s", exc)
                return None

        template = self._load_prompt()
        prompt = template.format(text=text)
        try:
            response = llm.chat([ChatMessage(role="user", content=prompt)], trace=trace)
            refined = response.content.strip()
            return refined if refined else None
        except LLMError as exc:
            logger.warning("LLM 精炼失败，将回退规则结果: %s", exc)
            return None

    def _load_prompt(self) -> str:
        if self._prompt_template is None:
            self._prompt_template = load_chunk_refinement_prompt(self._prompt_path)
        return self._prompt_template

    @staticmethod
    def _build_refined_chunk(
        chunk: Chunk,
        refined_text: str,
        metadata: dict[str, Any],
    ) -> Chunk:
        """基于精炼文本构造新 Chunk，并同步 end_offset。"""
        end_offset = chunk.start_offset + len(refined_text)
        refined = Chunk(
            id=chunk.id,
            text=refined_text,
            metadata=metadata,
            start_offset=chunk.start_offset,
            end_offset=end_offset,
            source_ref=chunk.source_ref,
        )
        refined.validate()
        return refined

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
