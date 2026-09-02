"""Ingestion 追踪页面：摄取历史列表与阶段耗时瀑布图。"""

from __future__ import annotations

import streamlit as st

from observability.dashboard.services.trace_service import TraceRecord, TraceService


def render_ingestion_traces(
    trace_service: TraceService | None = None,
    *,
    load_data: bool = True,
) -> None:
    """
    渲染摄取追踪页。

    Args:
        trace_service: 可注入的 Trace 读取服务，测试时指向临时 jsonl。
        load_data: 为 False 时不访问默认 traces.jsonl（测试用）。
    """
    st.header("Ingestion 追踪")
    st.caption("按时间倒序查看摄取 Trace，以及 load/split/transform/embed/upsert 耗时")

    service = trace_service
    if service is None:
        if not load_data:
            st.info("未注入 TraceService。")
            return
        try:
            service = TraceService.from_settings()
        except Exception:
            st.info("暂无法读取追踪日志。")
            return

    try:
        traces = service.list_traces("ingestion")
    except Exception:
        st.info("读取 traces.jsonl 失败。")
        return

    if not traces:
        st.info("暂无摄取追踪记录。可先在 Ingestion 管理页执行摄取。")
        return

    st.subheader("摄取历史")
    st.dataframe(_history_table(traces), width="stretch", hide_index=True)

    labels = [
        f"{item.source_name} · {item.status} · {item.total_elapsed_ms:.1f} ms"
        for item in traces
    ]
    selected = st.selectbox("查看详情", list(range(len(traces))), format_func=lambda index: labels[index])
    _render_trace_detail(traces[int(selected)])


def _history_table(traces: list[TraceRecord]) -> list[dict[str, str | float | int]]:
    """历史列表：文件名、集合、总耗时、状态。"""
    rows: list[dict[str, str | float | int]] = []
    for item in traces:
        rows.append(
            {
                "文件": item.source_name,
                "集合": item.collection or "—",
                "状态": item.status,
                "总耗时 (ms)": round(item.total_elapsed_ms, 2),
                "chunk 数": item.chunk_count,
                "开始时间": item.started_at or "—",
            }
        )
    return rows


def _render_trace_detail(trace: TraceRecord) -> None:
    """单次摄取：统计指标、横向瀑布图、各阶段 method/provider。"""
    st.subheader("摄取详情")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("状态", trace.status)
    col_b.metric("总耗时 (ms)", f"{trace.total_elapsed_ms:.1f}")
    col_c.metric("Chunk 数", trace.chunk_count)
    col_d.metric("图片数", trace.image_count)
    if trace.source_path:
        st.caption(trace.source_path)
    if trace.collection:
        st.caption(f"集合：{trace.collection}")

    waterfall = trace.waterfall_rows()
    if waterfall:
        st.markdown("**阶段耗时瀑布图**")
        try:
            import pandas as pd

            frame = pd.DataFrame(waterfall, columns=["阶段", "耗时 (ms)"])
            st.bar_chart(frame, x="阶段", y="耗时 (ms)", horizontal=True)
        except Exception:
            st.bar_chart({name: elapsed for name, elapsed in waterfall})

    st.markdown("**各阶段详情**")
    for stage in trace.stages:
        heading = f"{stage.name} · {stage.elapsed_ms:.1f} ms"
        extras = [part for part in (stage.method, stage.provider) if part]
        if extras:
            heading = f"{heading} · {' / '.join(extras)}"
        with st.expander(heading):
            if stage.method:
                st.caption(f"method: {stage.method}")
            if stage.provider:
                st.caption(f"provider: {stage.provider}")
            if stage.details:
                st.json(stage.details)
            else:
                st.caption("无额外 details")


def render() -> None:
    """Streamlit 页面入口。"""
    render_ingestion_traces()
