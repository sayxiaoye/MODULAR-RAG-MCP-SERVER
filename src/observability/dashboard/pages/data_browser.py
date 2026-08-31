"""数据浏览器页面：文档列表、Chunk 详情与关联图片预览。"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from observability.dashboard.services.data_service import (
    ChunkView,
    DataService,
    DocumentRow,
    ImagePreview,
)


def render_data_browser(
    data_service: DataService | None = None,
    *,
    load_data: bool = True,
) -> None:
    """
    渲染数据浏览器。

    Args:
        data_service: 可注入的读取服务，测试时传入 Fake 存储。
        load_data: 为 False 时不访问真实向量库（测试用）。
    """
    st.header("数据浏览器")
    st.caption("浏览已摄入文档、Chunk 详情与关联图片")

    service = data_service
    if service is None:
        if not load_data:
            st.info("未注入 DataService。")
            return
        try:
            service = DataService.from_settings()
        except Exception:
            st.info("暂无法读取向量库（集合可能尚未创建）。")
            return

    try:
        collections = service.list_collections()
    except Exception:
        st.info("暂无法读取向量库（集合可能尚未创建）。")
        return

    options = ["全部"] + collections
    selected = st.selectbox("集合", options)
    keyword = st.text_input("关键词搜索", placeholder="按 source_path 筛选")

    collection = None if selected == "全部" else selected
    needle = keyword.strip() or None
    try:
        documents = service.list_documents(collection=collection, keyword=needle)
    except Exception:
        st.info("读取文档列表失败。")
        return

    if not documents:
        st.info("暂无已摄入文档。")
        return

    st.subheader("文档列表")
    st.dataframe(_documents_table(documents), use_container_width=True, hide_index=True)

    st.subheader("Chunk 详情")
    for document in documents:
        _render_document_expander(service, document)


def _documents_table(documents: list[DocumentRow]) -> list[dict[str, str | int]]:
    """把文档行转成 dataframe 友好的字典列表。"""
    rows: list[dict[str, str | int]] = []
    for item in documents:
        rows.append(
            {
                "source_path": item.source_path,
                "集合": item.collection,
                "chunk 数": item.chunk_count,
                "图片数": item.image_count,
                "摄入时间": item.processed_at or "—",
            }
        )
    return rows


def _render_document_expander(service: DataService, document: DocumentRow) -> None:
    """展开单个文档：元信息、全部 chunk、文档级图片。"""
    title = Path(document.source_path).name or document.source_path
    label = f"{title} · {document.collection} · {document.chunk_count} chunks"
    with st.expander(label):
        st.caption(document.source_path)
        if document.processed_at:
            st.caption(f"摄入时间：{document.processed_at}")

        doc_images = service.list_images(
            collection=document.collection or None,
            doc_hash=document.doc_id,
        )
        if doc_images:
            st.markdown("**关联图片**")
            _render_images(doc_images)

        try:
            chunks = service.get_chunks(document.source_path, document.collection)
        except Exception as exc:
            st.warning(f"读取 chunk 失败：{exc}")
            return

        if not chunks:
            st.info("该文档没有 chunk。")
            return

        for chunk in chunks:
            _render_chunk(chunk)


def _render_chunk(chunk: ChunkView) -> None:
    """折叠展示单个 chunk 的正文、metadata 与图片。"""
    preview = chunk.text.strip().replace("\n", " ")[:48] or chunk.chunk_id
    heading = f"{chunk.chunk_id[:12]}… · {preview}" if len(chunk.chunk_id) > 12 else f"{chunk.chunk_id} · {preview}"
    with st.expander(heading):
        st.markdown("**正文**")
        st.text(chunk.text)
        st.markdown("**Metadata**")
        st.json(chunk.metadata)
        if chunk.images:
            st.markdown("**关联图片**")
            _render_images(chunk.images)


def _render_images(images: list[ImagePreview]) -> None:
    """按列展示缩略图；文件缺失时只显示 image_id。"""
    columns = st.columns(min(4, len(images)))
    for column, image in zip(columns, images):
        with column:
            if image.exists and image.file_path and Path(image.file_path).is_file():
                st.image(image.file_path, caption=image.image_id, use_container_width=True)
            else:
                st.caption(image.image_id)
                st.caption("文件不存在")


def render() -> None:
    """Streamlit 页面入口。"""
    render_data_browser()
