"""Trace 收集器：汇总 TraceContext 并触发持久化钩子（实际写文件由 F2 完成）。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core.trace.trace_context import TraceContext

PersistFn = Callable[[dict[str, Any]], None]


class TraceCollector:
    """
    收集已结束的 trace，并可选调用 persist 回调。

    F1 只负责内存收集与钩子；F2 将 persist_fn 接到 JSON Lines 写入。
    """

    def __init__(self, persist_fn: PersistFn | None = None) -> None:
        self._persist_fn = persist_fn
        self._traces: list[dict[str, Any]] = []

    def collect(self, trace: TraceContext) -> None:
        """
        收集一条 trace：若尚未 finish 则先结束，再序列化并触发持久化。

        Args:
            trace: 待收集的 TraceContext。
        """
        if not trace.is_finished:
            trace.finish()
        payload = trace.to_dict()
        self._traces.append(payload)
        if self._persist_fn is not None:
            self._persist_fn(payload)

    @property
    def traces(self) -> list[dict[str, Any]]:
        """已收集的 trace 快照（按收集顺序）。"""
        return list(self._traces)
