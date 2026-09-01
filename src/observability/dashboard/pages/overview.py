"""系统总览页：组件配置卡片 + Chroma 集合资产统计。"""

from __future__ import annotations

from typing import Any, Protocol

import streamlit as st

from libs.vector_store.chroma_store import CollectionStats
from libs.vector_store.vector_store_factory import VectorStoreFactory
from observability.dashboard.services.config_service import ConfigService


class _StatsProvider(Protocol):
    def get_collection_stats(self, collection: str | None = None) -> CollectionStats: ...


def _load_chroma_stats(settings: Any) -> CollectionStats | None:
    """汇总持久化目录下全部 Chroma 集合的文档/chunk 数。"""
    try:
        store = VectorStoreFactory.create(settings)
        if not hasattr(store, "get_collection_stats"):
            return None
        stats_store: _StatsProvider = store  # type: ignore[assignment]
        names_fn = getattr(store, "list_collection_names", None)
        names: list[str] = []
        if callable(names_fn):
            names = [str(item).strip() for item in names_fn() if str(item).strip()]
        if not names:
            return stats_store.get_collection_stats()
        chunk_count = 0
        document_count = 0
        for name in names:
            item = stats_store.get_collection_stats(name)
            chunk_count += item.chunk_count
            document_count += item.document_count
        label = "、".join(names) if len(names) <= 3 else f"{len(names)} 个集合"
        return CollectionStats(
            collection=label,
            chunk_count=chunk_count,
            document_count=document_count,
        )
    except Exception:
        return None


def render_overview(
    config_service: ConfigService | None = None,
    stats: CollectionStats | None = None,
    *,
    load_stats: bool = True,
) -> None:
    """
    渲染系统总览。

    Args:
        config_service: 可注入的配置服务，测试时传入固定 Settings。
        stats: 可注入的集合统计；为 None 且 load_stats=True 时现场读取 Chroma。
        load_stats: 测试可关闭真实向量库访问。
    """
    service = config_service or ConfigService()
    st.header("系统总览")
    st.caption("当前可插拔组件配置与知识库资产统计")

    st.subheader("组件配置")
    cards = service.component_cards()
    columns = st.columns(len(cards) or 1)
    for column, card in zip(columns, cards):
        with column:
            st.metric(card.title, card.summary)
            st.caption(f"provider: {card.provider}")

    st.subheader("数据资产")
    resolved_stats = stats
    if resolved_stats is None and load_stats:
        resolved_stats = _load_chroma_stats(service.settings)

    if resolved_stats is None:
        st.info("暂无法读取向量库统计（集合可能尚未创建）。")
        return

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("集合", resolved_stats.collection)
    col_b.metric("文档数", resolved_stats.document_count)
    col_c.metric("Chunk 数", resolved_stats.chunk_count)


def render() -> None:
    """Streamlit 页面入口。"""
    render_overview()
