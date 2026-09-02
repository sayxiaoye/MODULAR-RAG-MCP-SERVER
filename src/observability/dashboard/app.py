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
from observability.dashboard.pages.evaluation_panel import render as render_evaluation_panel
from observability.dashboard.pages.ingestion_manager import render as render_ingestion_manager
from observability.dashboard.pages.ingestion_traces import render as render_ingestion_traces
from observability.dashboard.pages.overview import render as render_overview
from observability.dashboard.pages.query_traces import render as render_query_traces

st.set_page_config(page_title="Modular RAG Dashboard", layout="wide")


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
            st.Page(render_evaluation_panel, title="评估面板", url_path="evaluation"),
        ],
    }
)
_NAV.run()
