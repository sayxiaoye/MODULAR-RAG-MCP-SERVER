"""Trace 上下文：摄取/查询链路的阶段打点占位实现（F 阶段完善）。"""

from __future__ import annotations

import time
import uuid
from typing import Any


class TraceContext:
    """最小 Trace 上下文：生成 trace_id 并记录各阶段耗时与详情。"""

    def __init__(self, trace_type: str = "ingestion") -> None:
        self.trace_id = str(uuid.uuid4())
        self.trace_type = trace_type
        self._started_at = time.perf_counter()
        self._stages: list[dict[str, Any]] = []

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
            name: 阶段名称（如 chunk_refiner_rule、chunk_refiner_llm）。
            elapsed_ms: 可选耗时（毫秒）。
            **details: 附加结构化字段（method、fallback 原因等）。
        """
        stage: dict[str, Any] = {"name": name}
        if elapsed_ms is not None:
            stage["elapsed_ms"] = round(elapsed_ms, 3)
        stage.update(details)
        self._stages.append(stage)

    def finish(self) -> dict[str, Any]:
        """结束追踪并返回摘要 payload。"""
        total_ms = (time.perf_counter() - self._started_at) * 1000
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "total_elapsed_ms": round(total_ms, 3),
            "stages": list(self._stages),
        }
