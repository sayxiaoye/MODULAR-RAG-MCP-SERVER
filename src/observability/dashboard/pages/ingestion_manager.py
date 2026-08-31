"""Ingestion 管理页面：上传触发摄取、进度条与文档删除。"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Protocol, Sequence

import streamlit as st

from core.settings import load_settings
from ingestion.document_manager import DeleteResult, DocumentManager
from ingestion.pipeline import IngestionPipeline, IngestionPipelineError, IngestionResult
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from libs.loader.file_integrity import SQLiteIntegrityChecker
from libs.vector_store.vector_store_factory import VectorStoreFactory
from observability.dashboard.services.data_service import DataService, DocumentRow

# F5 规范阶段顺序，用于把 on_progress 映射到进度条 0~1
_STAGE_ORDER = ("integrity", "load", "split", "transform", "embed", "upsert")
_NEW_COLLECTION_OPTION = "新建集合…"


class _PipelineLike(Protocol):
    def run(
        self,
        source_path: str,
        collection: str = "default",
        force: bool = False,
        on_progress: Any | None = None,
        trace: Any | None = None,
    ) -> IngestionResult: ...


class _ManagerLike(Protocol):
    def delete_document(self, source_path: str, collection: str) -> DeleteResult: ...


def progress_fraction(stage: str, current: int, total: int) -> float:
    """
    将 Pipeline on_progress 映射为 st.progress 所需的 0~1。

    Args:
        stage: 规范阶段名（integrity/load/split/transform/embed/upsert）。
        current: 当前进度（从 1 计）。
        total: 该阶段总量。

    Returns:
        夹在 [0, 1] 的进度比例；未知阶段返回 0。
    """
    if stage not in _STAGE_ORDER:
        return 0.0
    index = _STAGE_ORDER.index(stage)
    span = 1.0 / len(_STAGE_ORDER)
    inner = 0.0 if total <= 0 else min(1.0, max(0.0, current / total))
    return min(1.0, (index + inner) * span)


def collect_pdf_paths(path: str) -> list[Path]:
    """收集单文件或目录顶层 PDF，供路径输入触发摄取。"""
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"路径不存在: {path}")
    if candidate.is_file():
        if candidate.suffix.lower() != ".pdf":
            raise ValueError(f"仅支持 PDF 文件: {path}")
        return [candidate.resolve()]
    if candidate.is_dir():
        return [item.resolve() for item in sorted(candidate.glob("*.pdf"))]
    raise FileNotFoundError(f"无法读取路径: {path}")


def save_uploaded_files(uploaded_files: Sequence[Any], staging_dir: Path | None = None) -> list[Path]:
    """把 Streamlit UploadedFile 落到临时目录，供 Pipeline.run 读取本地路径。"""
    if not uploaded_files:
        return []
    root = staging_dir or Path(tempfile.mkdtemp(prefix="modular_rag_upload_"))
    root.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for item in uploaded_files:
        name = Path(getattr(item, "name", "upload.pdf")).name or "upload.pdf"
        dest = root / name
        dest.write_bytes(item.getvalue())
        saved.append(dest)
    return saved


def render_ingestion_manager(
    pipeline: _PipelineLike | None = None,
    document_manager: _ManagerLike | None = None,
    data_service: DataService | None = None,
    *,
    load_deps: bool = True,
    default_collection: str | None = None,
) -> None:
    """
    渲染 Ingestion 管理页。

    Args:
        pipeline: 可注入的摄取流水线，测试时传入 Fake。
        document_manager: 可注入的文档删除协调器。
        data_service: 可注入的文档列表读取服务。
        load_deps: 为 False 时不访问真实存储（测试用）。
        default_collection: 集合下拉的默认项。
    """
    st.header("Ingestion 管理")
    st.caption("上传或指定 PDF，触发摄取并管理已摄入文档")

    _show_flash("ingest_flash")
    _show_flash("delete_flash")

    resolved = _resolve_deps(
        pipeline,
        document_manager,
        data_service,
        load_deps=load_deps,
        default_collection=default_collection,
    )
    if resolved is None:
        return
    pipeline, document_manager, data_service, fallback_collection = resolved

    collection = _render_collection_picker(data_service, fallback_collection)
    uploaded = st.file_uploader("上传 PDF", type=["pdf"], accept_multiple_files=True)
    path_input = st.text_input("或输入本地文件/目录路径")
    force = st.checkbox("强制重新摄取", value=False)

    if st.button("开始摄取", type="primary"):
        _handle_ingest(
            pipeline,
            collection=collection,
            uploaded_files=uploaded or [],
            path_input=path_input,
            force=force,
        )

    st.subheader("已摄入文档")
    try:
        documents = data_service.list_documents(collection=collection or None)
    except Exception:
        st.info("读取文档列表失败。")
        return

    if not documents:
        st.info("暂无已摄入文档。")
        return

    for index, document in enumerate(documents):
        _render_document_row(document_manager, document, index)


def _resolve_deps(
    pipeline: _PipelineLike | None,
    document_manager: _ManagerLike | None,
    data_service: DataService | None,
    *,
    load_deps: bool,
    default_collection: str | None,
) -> tuple[_PipelineLike, _ManagerLike, DataService, str] | None:
    if pipeline is not None and document_manager is not None and data_service is not None:
        fallback = default_collection or "knowledge_hub"
        return pipeline, document_manager, data_service, fallback

    if not load_deps:
        st.info("未注入摄取依赖。")
        return None

    try:
        return _load_default_deps()
    except Exception:
        st.info("暂无法初始化摄取组件（请检查配置与向量库）。")
        return None


def _load_default_deps() -> tuple[IngestionPipeline, DocumentManager, DataService, str]:
    """用同一套 Chroma / ImageStorage / Integrity 实例组装 Pipeline 与 DocumentManager。"""
    settings = load_settings()
    collection = settings.vector_store.collection_name
    chroma = VectorStoreFactory.create(settings)
    images = ImageStorage()
    integrity = SQLiteIntegrityChecker()
    bm25 = BM25Indexer(collection=collection)
    pipeline = IngestionPipeline(
        settings=settings,
        integrity_checker=integrity,
        image_storage=images,
        vector_store=chroma,
    )
    manager = DocumentManager(chroma, bm25, images, integrity)
    data = DataService(
        chroma,
        images,
        integrity,
        default_collection=collection,
    )
    return pipeline, manager, data, collection


def _render_collection_picker(data_service: DataService, fallback: str) -> str:
    """集合下拉；选「新建」时展示文本框。"""
    try:
        names = list(data_service.list_collections())
    except Exception:
        names = []
    if fallback and fallback not in names:
        names = [fallback] + names
    if not names:
        names = [fallback or "default"]
    options = names + [_NEW_COLLECTION_OPTION]
    selected = st.selectbox("目标集合", options)
    if selected == _NEW_COLLECTION_OPTION:
        return st.text_input("新集合名称", value="").strip()
    return str(selected)


def _handle_ingest(
    pipeline: _PipelineLike,
    *,
    collection: str,
    uploaded_files: Sequence[Any],
    path_input: str,
    force: bool,
) -> None:
    if not collection:
        st.warning("请选择或输入集合名称。")
        return

    try:
        paths = list(save_uploaded_files(uploaded_files))
        raw = (path_input or "").strip()
        if raw:
            paths.extend(collect_pdf_paths(raw))
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.error(str(exc))
        return

    if not paths:
        st.warning("请上传 PDF 或输入本地路径。")
        return

    bar = st.progress(0)
    status = st.empty()

    def on_progress(stage: str, current: int, total: int) -> None:
        bar.progress(progress_fraction(stage, current, total))
        status.caption(f"当前阶段：{stage}（{current}/{total}）")

    summaries: list[str] = []
    for source in paths:
        try:
            result = pipeline.run(
                str(source),
                collection=collection,
                force=force,
                on_progress=on_progress,
            )
        except IngestionPipelineError as exc:
            st.error(f"摄取失败 [{exc.stage}]：{exc} — {source}")
            continue
        except Exception as exc:
            st.error(f"摄取失败：{exc} — {source}")
            continue

        if result.skipped:
            summaries.append(f"已跳过（未变更）：{Path(result.source_path).name}")
        else:
            summaries.append(
                f"摄取完成：{Path(result.source_path).name} · {result.chunk_count} chunks"
            )

    bar.progress(1.0)
    status.caption("完成")
    if summaries:
        st.session_state["ingest_flash"] = "\n".join(summaries)
        st.rerun()


def _render_document_row(document_manager: _ManagerLike, document: DocumentRow, index: int) -> None:
    """单行文档信息 + 删除按钮。"""
    name = Path(document.source_path).name or document.source_path
    col_name, col_meta, col_action = st.columns([4, 3, 1])
    col_name.write(name)
    col_name.caption(document.source_path)
    col_meta.write(f"{document.collection} · {document.chunk_count} chunks")
    if document.processed_at:
        col_meta.caption(document.processed_at)
    if col_action.button("删除", key=f"delete-{index}-{document.source_path}"):
        try:
            result = document_manager.delete_document(document.source_path, document.collection)
        except Exception as exc:
            st.error(f"删除失败：{exc}")
            return
        st.session_state["delete_flash"] = (
            f"已删除 {Path(result.source_path).name}（{result.chroma_deleted} 条 chunk）"
        )
        st.rerun()


def _show_flash(key: str) -> None:
    """展示并清除一次性成功提示。"""
    message = st.session_state.pop(key, None)
    if message:
        st.success(message)


def render() -> None:
    """Streamlit 页面入口。"""
    render_ingestion_manager()
