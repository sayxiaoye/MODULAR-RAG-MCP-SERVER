"""从已摄入集合生成黄金测试集：LLM 出题，chunk_id 取自向量库。"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from core.settings import Settings, load_settings, resolve_path
from libs.llm.base_llm import BaseLLM, ChatMessage, ChatResponse, LLMError
from libs.vector_store.base_vector_store import BaseVectorStore
from observability.evaluation.eval_runner import load_golden_test_set

logger = logging.getLogger(__name__)

DEFAULT_CASE_COUNT = 8
MAX_CASE_COUNT = 30
_QUERY_PROMPT = (
    "根据下面这段知识库原文，生成一个用户可能会问的中文问题。\n"
    "要求：问题必须能由这段原文直接回答；只输出问题本身，不要编号、引号或解释。\n\n"
    "原文：\n{text}"
)
_MAX_CHUNK_CHARS = 1200
_FILENAME_SAFE = re.compile(r"[^a-zA-Z0-9_-]+")


class GoldenGeneratorError(Exception):
    """集合为空、无法写文件或生成条数无效时抛出。"""


@dataclass(frozen=True)
class ChunkCandidate:
    """用于出题的一条 chunk：真实 id + 正文 + 来源。"""

    chunk_id: str
    text: str
    source_path: str
    collection: str


AskFn = Callable[[str], str]


def generated_golden_dir() -> Path:
    """生成集落盘目录（runtime，已被 gitignore 的 data/）。"""
    return resolve_path("data") / "eval"


def sanitize_collection_name(collection: str) -> str:
    """集合名转文件名安全片段。"""
    cleaned = _FILENAME_SAFE.sub("_", (collection or "").strip()).strip("_")
    return cleaned[:40] or "collection"


def fallback_query_from_text(text: str) -> str:
    """LLM 不可用时：取首句加问号，保证仍能写出可评估的 query。"""
    stripped = " ".join((text or "").split())
    if not stripped:
        return "这段内容在讲什么？"
    first = re.split(r"[。！？?\n]|：|:", stripped, maxsplit=1)[0].strip()
    if not first:
        first = stripped[:40].strip()
    if first.endswith(("？", "?")):
        return first
    if first.endswith("："):
        first = first[:-1].strip()
    return f"{first}？"


def parse_llm_query(raw: str, *, fallback_text: str) -> str:
    """从模型输出抽出单行问题；空或无效时回退规则 query。"""
    if not isinstance(raw, str) or not raw.strip():
        return fallback_query_from_text(fallback_text)
    line = raw.strip().splitlines()[0].strip()
    line = re.sub(r"^\s*\d+[\.\)、]\s*", "", line)
    line = line.strip("\"'`“”‘’ \t")
    line = re.sub(r'^["\'“”‘’]+|["\'“”‘’]+$', "", line)
    if not line or len(line) < 2:
        return fallback_query_from_text(fallback_text)
    return line


def records_to_candidates(
    records: Sequence[Mapping[str, Any]],
    collection: str,
) -> list[ChunkCandidate]:
    """把 Chroma get_by_metadata 记录转成可出题候选。"""
    coll = (collection or "").strip()
    result: list[ChunkCandidate] = []
    for item in records:
        chunk_id = str(item.get("id") or item.get("chunk_id") or "").strip()
        text = str(item.get("text") or "").strip()
        if not chunk_id or not text:
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
        source = ""
        if isinstance(metadata, Mapping):
            for key in ("source_path", "source_file", "file_name", "source"):
                value = metadata.get(key)
                if isinstance(value, str) and value.strip():
                    source = value.strip()
                    break
        result.append(
            ChunkCandidate(
                chunk_id=chunk_id,
                text=text,
                source_path=source,
                collection=coll,
            )
        )
    return result


def sample_chunks(
    candidates: Sequence[ChunkCandidate],
    count: int,
    *,
    rng: random.Random | None = None,
) -> list[ChunkCandidate]:
    """尽量按文档轮询取样，避免 N 条全来自同一文件。"""
    if count < 1:
        raise GoldenGeneratorError("生成条数必须 >= 1")
    usable = [item for item in candidates if item.chunk_id and item.text.strip()]
    if not usable:
        return []
    picker = rng or random.Random()
    by_source: dict[str, list[ChunkCandidate]] = {}
    for item in usable:
        key = item.source_path or item.chunk_id
        by_source.setdefault(key, []).append(item)
    for group in by_source.values():
        picker.shuffle(group)

    ordered_keys = list(by_source.keys())
    picker.shuffle(ordered_keys)
    picked: list[ChunkCandidate] = []
    seen: set[str] = set()
    while len(picked) < count:
        progressed = False
        for key in ordered_keys:
            group = by_source[key]
            if not group:
                continue
            item = group.pop()
            if item.chunk_id in seen:
                continue
            seen.add(item.chunk_id)
            picked.append(item)
            progressed = True
            if len(picked) >= count:
                break
        if not progressed:
            break
    return picked


def ask_query_with_llm(llm: BaseLLM, text: str) -> str:
    """调用 LLM 根据 chunk 正文出一道题。"""
    excerpt = text.strip()
    if len(excerpt) > _MAX_CHUNK_CHARS:
        excerpt = excerpt[:_MAX_CHUNK_CHARS]
    prompt = _QUERY_PROMPT.format(text=excerpt)
    response = llm.chat([ChatMessage(role="user", content=prompt)])
    content = response.content if isinstance(response, ChatResponse) else str(response)
    return parse_llm_query(content, fallback_text=text)


def build_test_cases(
    samples: Sequence[ChunkCandidate],
    *,
    ask: AskFn | None = None,
    llm: BaseLLM | None = None,
) -> list[dict[str, Any]]:
    """为取样 chunk 生成 query；LLM 失败时用规则问句。"""
    cases: list[dict[str, Any]] = []
    for item in samples:
        query = fallback_query_from_text(item.text)
        try:
            if ask is not None:
                query = parse_llm_query(ask(item.text), fallback_text=item.text)
            elif llm is not None:
                query = ask_query_with_llm(llm, item.text)
        except Exception as exc:
            logger.warning("出题失败，使用规则问句 chunk_id=%s: %s", item.chunk_id, exc)
        sources = [item.source_path] if item.source_path else []
        cases.append(
            {
                "query": query,
                "expected_chunk_ids": [item.chunk_id],
                "expected_sources": sources,
            }
        )
    return cases


def write_golden_test_set(
    cases: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    collection: str = "",
) -> Path:
    """写入 EvalRunner 可读取的 JSON，并校验 test_cases。"""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "collection": collection,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "test_cases": list(cases),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    load_golden_test_set(path)
    return path


def generate_golden_test_set(
    collection: str,
    count: int = DEFAULT_CASE_COUNT,
    *,
    settings: Settings | None = None,
    vector_store: BaseVectorStore | None = None,
    llm: BaseLLM | None = None,
    ask: AskFn | None = None,
    output_path: str | Path | None = None,
    rng: random.Random | None = None,
) -> Path:
    """
    从指定集合取样 chunk，生成 N 条黄金用例并落盘。

    ``expected_chunk_ids`` 始终使用向量库中的真实 id；query 优先 LLM，失败则规则问句。
    """
    name = (collection or "").strip()
    if not name:
        raise GoldenGeneratorError("集合名称不能为空")
    if count < 1:
        raise GoldenGeneratorError("生成条数必须 >= 1")
    n = min(int(count), MAX_CASE_COUNT)

    store = vector_store
    if store is None:
        from libs.vector_store.vector_store_factory import VectorStoreFactory

        store = VectorStoreFactory.create(settings or load_settings())
    records = store.get_by_metadata(None, collection=name)
    candidates = records_to_candidates(records, name)
    samples = sample_chunks(candidates, n, rng=rng)
    if not samples:
        raise GoldenGeneratorError(f"集合 {name!r} 没有可出题的 chunk，请先摄入文档")

    asker_llm = llm
    if ask is None and asker_llm is None:
        try:
            from libs.llm.llm_factory import LLMFactory

            asker_llm = LLMFactory.create(settings or load_settings())
        except Exception as exc:
            logger.warning("无法创建 LLM，黄金集将使用规则问句: %s", exc)
            asker_llm = None

    cases = build_test_cases(samples, ask=ask, llm=asker_llm)
    target = Path(output_path) if output_path is not None else _default_output_path(name)
    return write_golden_test_set(cases, target, collection=name)


def _default_output_path(collection: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"golden_{sanitize_collection_name(collection)}_{stamp}.json"
    return generated_golden_dir() / filename
