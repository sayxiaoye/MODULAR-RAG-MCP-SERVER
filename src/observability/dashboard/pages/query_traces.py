"""Query 追踪页面：查询历史、耗时瀑布图、Dense/Sparse 对比与 Rerank 变化。"""

from __future__ import annotations

from typing import Any

import streamlit as st

from observability.dashboard.services.trace_service import TraceRecord, TraceService


def render_query_traces(
    trace_service: TraceService | None = None,
    *,
    load_data: bool = True,
) -> None:
    """
    渲染 Query 追踪页。

    Args:
        trace_service: 可注入的 Trace 读取服务，测试时指向临时 jsonl。
        load_data: 为 False 时不访问默认 traces.jsonl（测试用）。
    """
    st.header("Query 追踪")
    st.caption("按时间倒序查看查询 Trace，对比 Dense/Sparse 召回与 Rerank 排名变化")

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

    keyword = st.text_input("按 Query 关键词筛选", placeholder="匹配查询文本")
    try:
        traces = service.list_traces("query", keyword=keyword.strip() or None)
    except Exception:
        st.info("读取 traces.jsonl 失败。")
        return

    if not traces:
        st.info("暂无查询追踪记录。可先通过 query CLI 或 MCP 执行一次检索。")
        return

    st.subheader("查询历史")
    st.dataframe(_history_table(traces), width="stretch", hide_index=True)

    labels = [
        f"{(item.query_text or item.trace_id)[:48]} · {item.total_elapsed_ms:.1f} ms"
        for item in traces
    ]
    selected = st.selectbox("查看详情", list(range(len(traces))), format_func=lambda index: labels[index])
    _render_trace_detail(traces[int(selected)])


def _history_table(traces: list[TraceRecord]) -> list[dict[str, str | float]]:
    """历史列表：查询文本、集合、总耗时、状态。"""
    rows: list[dict[str, str | float]] = []
    for item in traces:
        query = item.query_text or "—"
        rows.append(
            {
                "Query": query if len(query) <= 80 else query[:77] + "…",
                "集合": item.collection or "—",
                "状态": item.status,
                "总耗时 (ms)": round(item.total_elapsed_ms, 2),
                "开始时间": item.started_at or "—",
            }
        )
    return rows


def _render_trace_detail(trace: TraceRecord) -> None:
    """单次查询：瀑布图、两路召回对比、Rerank 排名变化、最终 Top-K。"""
    st.subheader("查询详情")
    if trace.query_text:
        st.markdown(f"**Query：** {trace.query_text}")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("状态", trace.status)
    col_b.metric("总耗时 (ms)", f"{trace.total_elapsed_ms:.1f}")
    col_c.metric("最终命中", len(trace.lane_hits("rerank") or trace.lane_hits("fusion")))
    if trace.collection:
        st.caption(f"集合：{trace.collection}")

    waterfall = trace.query_waterfall_rows()
    if waterfall:
        st.markdown("**阶段耗时瀑布图**")
        try:
            import pandas as pd

            frame = pd.DataFrame(waterfall, columns=["阶段", "耗时 (ms)"])
            st.bar_chart(frame, x="阶段", y="耗时 (ms)", horizontal=True)
        except Exception:
            st.bar_chart({name: elapsed for name, elapsed in waterfall})

    dense_hits = trace.lane_hits("dense")
    sparse_hits = trace.lane_hits("sparse")
    if dense_hits or sparse_hits:
        st.markdown("**Dense vs Sparse 对比**")
        col_dense, col_sparse = st.columns(2)
        with col_dense:
            st.caption("Dense Top-N")
            st.dataframe(_hits_table(dense_hits), width="stretch", hide_index=True)
        with col_sparse:
            st.caption("Sparse Top-N")
            st.dataframe(_hits_table(sparse_hits), width="stretch", hide_index=True)

    changes = trace.rerank_rank_changes()
    if changes:
        st.markdown("**Rerank 前后排名变化**")
        st.dataframe(_rank_change_table(changes), width="stretch", hide_index=True)

    final_hits = trace.lane_hits("rerank") or trace.lane_hits("fusion")
    if final_hits:
        st.markdown("**最终结果 Top-K**")
        st.dataframe(_hits_table(final_hits), width="stretch", hide_index=True)

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
            details = {
                key: value
                for key, value in stage.details.items()
                if key not in {"dense_hits", "sparse_hits", "fusion_hits", "rerank_hits"}
            }
            if details:
                st.json(details)


def _hits_table(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in hits:
        rows.append(
            {
                "rank": item.get("rank"),
                "chunk_id": item.get("chunk_id"),
                "score": item.get("score"),
                "title": item.get("title") or "—",
                "source": item.get("source_path") or "—",
            }
        )
    return rows


def _rank_change_table(changes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in changes:
        rows.append(
            {
                "变化": item.get("mark"),
                "chunk_id": item.get("chunk_id"),
                "融合排名": item.get("fusion_rank") if item.get("fusion_rank") is not None else "—",
                "精排排名": item.get("rerank_rank"),
                "Δ": item.get("delta") if item.get("delta") is not None else "—",
                "score": item.get("score"),
                "title": item.get("title") or "—",
                "source": item.get("source_path") or "—",
            }
        )
    return rows


def render() -> None:
    """Streamlit 页面入口。"""
    render_query_traces()
