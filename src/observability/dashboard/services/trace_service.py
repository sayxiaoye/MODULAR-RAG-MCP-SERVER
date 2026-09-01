"""Dashboard Trace 读取服务：解析 traces.jsonl，按 trace_type 分类（G5）。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from core.settings import Settings, load_settings, resolve_path
from observability.logger import resolve_trace_file

# 摄取瀑布图使用的规范阶段（与 F4 埋点一致）
INGESTION_WATERFALL_STAGES = ("load", "split", "transform", "embed", "upsert")
# 查询瀑布图使用的规范阶段（与 F3 埋点一致）；兼容组件级阶段名
QUERY_WATERFALL_STAGES = (
    "query_processing",
    "dense_retrieval",
    "sparse_retrieval",
    "fusion",
    "rerank",
)
_QUERY_STAGE_ALIASES: dict[str, tuple[str, ...]] = {
    "query_processing": ("query_processing", "query_processor"),
    "dense_retrieval": ("dense_retrieval", "dense_retriever"),
    "sparse_retrieval": ("sparse_retrieval", "sparse_retriever"),
    "fusion": ("fusion",),
    "rerank": ("rerank",),
}


@dataclass(frozen=True)
class StageView:
    """单条 stage 的展示结构。"""

    name: str
    elapsed_ms: float
    method: str | None = None
    provider: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TraceRecord:
    """
    一条已持久化的 Trace，对应 TraceContext.to_dict()。

    供 Dashboard 列表与详情使用；raw 保留原始 JSON 便于展开。
    """

    trace_id: str
    trace_type: str
    started_at: str | None
    finished_at: str | None
    total_elapsed_ms: float
    stages: list[StageView] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        """根据阶段名与 finish 状态推断成功 / 失败 / 跳过 / 删除。"""
        names = {item.name for item in self.stages}
        if "error" in names:
            return "失败"
        if "deleted" in names:
            return "删除"
        if "skipped" in names:
            return "跳过"
        if self.finished_at:
            return "成功"
        return "进行中"

    @property
    def source_path(self) -> str | None:
        """优先 pipeline_complete / load 阶段上的 source_path。"""
        return _first_detail(self.stages, "source_path")

    @property
    def source_name(self) -> str:
        path = self.source_path
        if path:
            return Path(path).name
        return self.trace_id[:8]

    @property
    def collection(self) -> str | None:
        return _first_detail(self.stages, "collection")

    @property
    def chunk_count(self) -> int:
        for name in ("pipeline_complete", "deleted", "split", "embed", "upsert"):
            for stage in reversed(self.stages):
                if stage.name != name:
                    continue
                value = stage.details.get("chunk_count", stage.details.get("vector_count"))
                if value is not None:
                    try:
                        return int(value)
                    except (TypeError, ValueError):
                        return 0
        return 0

    @property
    def image_count(self) -> int:
        raw = _first_detail(self.stages, "image_count")
        try:
            return int(raw) if raw is not None else 0
        except (TypeError, ValueError):
            return 0

    @property
    def query_text(self) -> str | None:
        """查询原文，来自 query_complete.query。"""
        return _first_detail(self.stages, "query")

    def matches_keyword(self, keyword: str) -> bool:
        """按查询文本 / 来源路径 / trace_id 做不区分大小写匹配。"""
        needle = keyword.strip().lower()
        if not needle:
            return True
        haystacks = [self.query_text or "", self.source_path or "", self.trace_id]
        return any(needle in item.lower() for item in haystacks)

    def query_waterfall_rows(self) -> list[tuple[str, float]]:
        """按 F3 规范阶段顺序输出耗时；兼容 dense_retriever 等组件级名称。"""
        by_name = {item.name: item.elapsed_ms for item in self.stages}
        rows: list[tuple[str, float]] = []
        for canonical in QUERY_WATERFALL_STAGES:
            elapsed = None
            for alias in _QUERY_STAGE_ALIASES[canonical]:
                if alias in by_name:
                    elapsed = float(by_name[alias])
                    break
            if elapsed is not None:
                rows.append((canonical, elapsed))
        return rows

    def lane_hits(self, lane: str) -> list[dict[str, Any]]:
        """读取 query_complete 中 dense/sparse/fusion/rerank 命中快照。"""
        complete = _stage_by_name(self.stages, "query_complete")
        if complete is None:
            return []
        key = f"{lane}_hits"
        raw = complete.details.get(key)
        if not isinstance(raw, list):
            return []
        return [dict(item) for item in raw if isinstance(item, Mapping)]

    def rerank_rank_changes(self) -> list[dict[str, Any]]:
        """融合排名 vs 精排排名：正 delta 表示跃升。"""
        fusion_hits = self.lane_hits("fusion")
        rerank_hits = self.lane_hits("rerank")
        before_rank = {
            str(item.get("chunk_id")): int(item.get("rank"))
            for item in fusion_hits
            if item.get("chunk_id") is not None and item.get("rank") is not None
        }
        rows: list[dict[str, Any]] = []
        for item in rerank_hits:
            chunk_id = str(item.get("chunk_id") or "")
            after = item.get("rank")
            try:
                after_rank = int(after)
            except (TypeError, ValueError):
                continue
            before = before_rank.get(chunk_id)
            delta = None if before is None else before - after_rank
            if delta is None:
                mark = "—"
            elif delta > 0:
                mark = "↑"
            elif delta < 0:
                mark = "↓"
            else:
                mark = "—"
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "title": str(item.get("title") or ""),
                    "source_path": str(item.get("source_path") or ""),
                    "score": item.get("score"),
                    "fusion_rank": before,
                    "rerank_rank": after_rank,
                    "delta": delta,
                    "mark": mark,
                }
            )
        return rows

    def waterfall_rows(self) -> list[tuple[str, float]]:
        """按 F4 规范阶段顺序输出 (name, elapsed_ms)，供横向条形图。"""
        by_name = {item.name: item.elapsed_ms for item in self.stages}
        return [
            (name, float(by_name[name]))
            for name in INGESTION_WATERFALL_STAGES
            if name in by_name
        ]


class TraceService:
    """
    读取 JSON Lines 追踪日志，对应 G5 TraceService。

    跳过非 Trace 的日志行（JSONFormatter 文本日志）与损坏行。
    """

    def __init__(self, trace_file: str | Path | None = None) -> None:
        self._trace_file = Path(trace_file) if trace_file is not None else None

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> TraceService:
        """按 Settings.observability.trace_file 定位 jsonl。"""
        resolved = settings or load_settings()
        return cls(resolve_path(resolved.observability.trace_file))

    def list_traces(self, trace_type: str | None = None, keyword: str | None = None) -> list[TraceRecord]:
        """
        读取全部 Trace，按 started_at 倒序。

        Args:
            trace_type: 如 ``ingestion`` / ``query``；空则不过滤。
            keyword: 对 query 文本 / source_path / trace_id 做子串筛选。
        """
        path = self._resolve_path()
        records = _read_jsonl(path)
        if trace_type:
            wanted = trace_type.strip()
            records = [item for item in records if item.trace_type == wanted]
        if keyword and keyword.strip():
            records = [item for item in records if item.matches_keyword(keyword)]
        records.sort(key=lambda item: item.started_at or "", reverse=True)
        return records

    def get_trace(self, trace_id: str) -> TraceRecord | None:
        """按 trace_id 精确查找；未找到返回 None。"""
        if not trace_id or not trace_id.strip():
            return None
        needle = trace_id.strip()
        for item in self.list_traces():
            if item.trace_id == needle:
                return item
        return None

    def _resolve_path(self) -> Path:
        if self._trace_file is not None:
            candidate = self._trace_file
            return candidate if candidate.is_absolute() else resolve_path(candidate)
        return resolve_trace_file()


def parse_trace_payload(payload: Mapping[str, Any]) -> TraceRecord | None:
    """把一行 JSON 解析为 TraceRecord；缺少 trace_id/stages 的日志行返回 None。"""
    trace_id = str(payload.get("trace_id") or "").strip()
    stages_raw = payload.get("stages")
    if not trace_id or not isinstance(stages_raw, list):
        return None
    stages = [_parse_stage(item) for item in stages_raw if isinstance(item, Mapping)]
    try:
        total = float(payload.get("total_elapsed_ms") or 0.0)
    except (TypeError, ValueError):
        total = 0.0
    return TraceRecord(
        trace_id=trace_id,
        trace_type=str(payload.get("trace_type") or ""),
        started_at=_optional_str(payload.get("started_at")),
        finished_at=_optional_str(payload.get("finished_at")),
        total_elapsed_ms=total,
        stages=stages,
        raw=dict(payload),
    )


def _read_jsonl(path: Path) -> list[TraceRecord]:
    if not path.is_file():
        return []
    records: list[TraceRecord] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        record = parse_trace_payload(payload)
        if record is not None:
            records.append(record)
    return records


def _parse_stage(item: Mapping[str, Any]) -> StageView:
    name = str(item.get("name") or "")
    try:
        elapsed = float(item.get("elapsed_ms") or 0.0)
    except (TypeError, ValueError):
        elapsed = 0.0
    method = item.get("method")
    provider = item.get("provider")
    details = {
        key: value
        for key, value in item.items()
        if key not in {"name", "elapsed_ms", "method", "provider"}
    }
    return StageView(
        name=name,
        elapsed_ms=elapsed,
        method=None if method is None else str(method),
        provider=None if provider is None else str(provider),
        details=details,
    )


def _stage_by_name(stages: list[StageView], name: str) -> StageView | None:
    for stage in reversed(stages):
        if stage.name == name:
            return stage
    return None


def _first_detail(stages: list[StageView], key: str) -> str | None:
    for stage in reversed(stages):
        value = stage.details.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
