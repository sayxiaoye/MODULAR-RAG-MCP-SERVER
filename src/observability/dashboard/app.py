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
from observability.dashboard.pages.overview import render as render_overview

st.set_page_config(page_title="Modular RAG Dashboard", layout="wide")


def _placeholder(title: str, hint: str):
    """未实现页面的占位渲染器，后续 G5–G6 / H4 再替换为真实页面。"""

    def render() -> None:
        st.header(title)
        st.info(hint)

    render.__name__ = f"placeholder_{title}"
    return render


_NAV = st.navigation(
    {
        "观测": [
            st.Page(render_overview, title="系统总览", default=True),
            st.Page(render_data_browser, title="数据浏览器"),
        ],
        "摄取": [
            st.Page(render_ingestion_manager, title="Ingestion 管理"),
            st.Page(
                _placeholder("Ingestion 追踪", "摄取追踪将在 G5 实现。"),
                title="Ingestion 追踪",
            ),
        ],
        "查询与评估": [
            st.Page(
                _placeholder("Query 追踪", "查询追踪将在 G6 实现。"),
                title="Query 追踪",
            ),
            st.Page(
                _placeholder("评估面板", "评估模块尚未启用，将在阶段 H 实现。"),
                title="评估面板",
            ),
        ],
    }
)
_NAV.run()
