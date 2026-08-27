"""Trace 上下文：query / ingestion 链路的阶段打点与耗时统计。"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any

ALLOWED_TRACE_TYPES = ("query", "ingestion")


class TraceContext:
    """
    请求级追踪上下文，对应 F1：trace_type、finish、elapsed_ms、to_dict。

    使用 monotonic 时钟计算耗时，ISO 时间戳用于序列化与 Dashboard 展示。
    """

    def __init__(self, trace_type: str = "query") -> None:
        if trace_type not in ALLOWED_TRACE_TYPES:
            raise ValueError(
                f"trace_type 必须是 {ALLOWED_TRACE_TYPES} 之一，收到: {trace_type!r}"
            )
        self.trace_id = str(uuid.uuid4())
        self.trace_type = trace_type
        self._started_at_mono = time.perf_counter()
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.finished_at: str | None = None
        self._finished = False
        self._total_elapsed_ms: float | None = None
        self._stages: list[dict[str, Any]] = []

    @property
    def is_finished(self) -> bool:
        """是否已调用 finish()。"""
        return self._finished

    def record_stage(
        self,
        name: str,
        *,
        elapsed_ms: float | None = None,
        **details: Any,
    ) -> None:
        """
        记录单个处理阶段。

        Args:
            name: 阶段名称（如 chunk_refiner_rule、dense）。
            elapsed_ms: 可选耗时（毫秒）。
            **details: 附加结构化字段（method、fallback 原因等）。
        """
        stage: dict[str, Any] = {"name": name}
        if elapsed_ms is not None:
            stage["elapsed_ms"] = round(float(elapsed_ms), 3)
        stage.update(details)
        self._stages.append(stage)

    def finish(self) -> None:
        """标记 trace 结束并冻结总耗时；重复调用视为幂等。"""
        if self._finished:
            return
        self._total_elapsed_ms = (time.perf_counter() - self._started_at_mono) * 1000
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self._finished = True

    def elapsed_ms(self, stage_name: str | None = None) -> float:
        """
        获取耗时（毫秒）。

        Args:
            stage_name: 指定阶段名时返回该阶段 elapsed_ms；为 None 时返回总耗时。
        """
        if stage_name is None:
            if self._total_elapsed_ms is not None:
                return round(self._total_elapsed_ms, 3)
            return round((time.perf_counter() - self._started_at_mono) * 1000, 3)

        for stage in reversed(self._stages):
            if stage.get("name") == stage_name:
                value = stage.get("elapsed_ms", 0.0)
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """
        序列化为可 json.dumps 的字典。

        未 finish 时 finished_at 为 None，total_elapsed_ms 为当前累计耗时。
        """
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_elapsed_ms": self.elapsed_ms(),
            "stages": list(self._stages),
        }
