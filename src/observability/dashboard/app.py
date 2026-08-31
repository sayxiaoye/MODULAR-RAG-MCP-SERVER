"""Dashboard 入口：Streamlit 多页面导航（G1 六页面骨架）。"""

from __future__ import annotations

import sys
from pathlib import Path

# streamlit run 时保证 src/ 在 path 上
_SRC_ROOT = Path(__file__).resolve().parents[2]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import streamlit as st

from observability.dashboard.pages.data_browser import render as render_data_browser
from observability.dashboard.pages.ingestion_manager import render as render_ingestion_manager
from observability.dashboard.pages.ingestion_traces import render as render_ingestion_traces
from observability.dashboard.pages.overview import render as render_overview
from observability.dashboard.pages.query_traces import render as render_query_traces

st.set_page_config(page_title="Modular RAG Dashboard", layout="wide")


def _placeholder(title: str, hint: str):
    """未实现页面的占位渲染器，后续 H4 再替换为真实页面。"""

    def render() -> None:
        st.header(title)
        st.info(hint)

    render.__name__ = f"placeholder_{title}"
    return render


_NAV = st.navigation(
    {
        "观测": [
            st.Page(render_overview, title="系统总览", url_path="overview", default=True),
            st.Page(render_data_browser, title="数据浏览器", url_path="data-browser"),
        ],
        "摄取": [
            st.Page(render_ingestion_manager, title="Ingestion 管理", url_path="ingestion"),
            st.Page(render_ingestion_traces, title="Ingestion 追踪", url_path="ingestion-traces"),
        ],
        "查询与评估": [
            st.Page(render_query_traces, title="Query 追踪", url_path="query-traces"),
            st.Page(
                _placeholder("评估面板", "评估模块尚未启用，将在阶段 H 实现。"),
                title="评估面板",
                url_path="evaluation",
            ),
        ],
    }
)
_NAV.run()
