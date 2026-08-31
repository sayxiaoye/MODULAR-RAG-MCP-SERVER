"""Ingestion Pipeline：串行编排 integrity→load→split→transform→encode→store。

F4 在编排层写入规范阶段名 load / split / transform / embed / upsert。
F5 通过可选 on_progress(stage, current, total) 向外报告同一组阶段进度。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from core.settings import Settings, VectorStoreSettings, resolve_path
from core.trace.trace_context import TraceContext
from core.types import Chunk, Document, ImageMetadata
from ingestion.chunking.document_chunker import DocumentChunker
from ingestion.embedding.batch_processor import BatchProcessor, BatchEncodingResult
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from ingestion.storage.vector_upserter import VectorUpserter
from ingestion.transform.base_transform import BaseTransform
from ingestion.transform.chunk_refiner import ChunkRefiner
from ingestion.transform.image_captioner import ImageCaptioner
from ingestion.transform.metadata_enricher import MetadataEnricher
from libs.loader.base_loader import BaseLoader, LoaderError
from libs.loader.file_integrity import FileIntegrityChecker, SQLiteIntegrityChecker
from libs.loader.pdf_loader import PdfLoader
from libs.vector_store.base_vector_store import BaseVectorStore
from libs.vector_store.vector_store_factory import VectorStoreFactory
from observability.logger import write_trace

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class IngestionResult:
    """单次摄取执行结果摘要。"""

    skipped: bool
    source_path: str
    collection: str
    document_id: str | None = None
    chunk_count: int = 0
    image_count: int = 0
    chunk_ids: list[str] = field(default_factory=list)
    trace_id: str | None = None


class IngestionPipelineError(Exception):
    """Pipeline 某阶段失败时抛出，携带阶段名便于定位。"""

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        super().__init__(f"[{stage}] {message}")


class IngestionPipeline:
    """
    摄取主流程编排器：将 C2~C13 各模块串成可观测的 MVP 链路。

    默认顺序：integrity → load → split → transform → encode → store。
    F4 将摄取链路埋点为 ``load`` / ``split`` / ``transform`` / ``embed`` / ``upsert``。
    F5 进度回调使用同一组规范阶段名（外加 integrity）。
    """

    def __init__(
        self,
        settings: Settings,
        integrity_checker: FileIntegrityChecker | None = None,
        loader: BaseLoader | None = None,
        chunker: DocumentChunker | None = None,
        transforms: Sequence[BaseTransform] | None = None,
        batch_processor: BatchProcessor | None = None,
        vector_store: BaseVectorStore | None = None,
        vector_upserter: VectorUpserter | None = None,
        bm25_indexer: BM25Indexer | None = None,
        image_storage: ImageStorage | None = None,
        images_root: str | Path | None = None,
        bm25_root: str | Path | None = None,
        chroma_persist_directory: str | Path | None = None,
    ) -> None:
        self._base_settings = settings
        self._integrity_checker = integrity_checker or SQLiteIntegrityChecker()
        self._loader = loader
        self._chunker = chunker
        self._transforms = list(transforms) if transforms is not None else None
        self._batch_processor = batch_processor
        self._vector_store = vector_store
        self._vector_upserter = vector_upserter
        self._bm25_indexer = bm25_indexer
        self._image_storage = image_storage
        self._images_root = (
            resolve_path(images_root) if images_root is not None else None
        )
        self._bm25_root = resolve_path(bm25_root) if bm25_root is not None else None
        self._chroma_persist_directory = (
            str(resolve_path(chroma_persist_directory))
            if chroma_persist_directory is not None
            else None
        )

    def run(
        self,
        source_path: str,
        collection: str = "default",
        force: bool = False,
        on_progress: ProgressCallback | None = None,
        trace: TraceContext | None = None,
    ) -> IngestionResult:
        """
        执行完整摄取流程。

        Args:
            source_path: 待摄取文件路径（当前 MVP 支持 PDF）。
            collection: 目标集合名，写入 metadata 与各存储后端。
            force: True 时忽略 integrity 跳过判定。
            on_progress: 可选进度回调 ``(stage, current, total)``。
            trace: 可选 TraceContext；未提供时自动创建 ingestion trace。

        Returns:
            IngestionResult 摘要；跳过时 ``skipped=True`` 且 chunk_count=0。
        """
        if not collection or not collection.strip():
            raise IngestionPipelineError("init", "collection 不能为空")

        collection_name = collection.strip()
        active_trace = trace or TraceContext(trace_type="ingestion")
        resolved_source = str(Path(source_path).resolve())

        try:
            file_hash = self._run_integrity_precheck(
                resolved_source,
                force=force,
                on_progress=on_progress,
            )
            if file_hash is None:
                logger.info("跳过已摄取文件: %s", resolved_source)
                active_trace.record_stage(
                    "skipped",
                    elapsed_ms=0.0,
                    source_path=resolved_source,
                    collection=collection_name,
                )
                active_trace.finish()
                self._persist_trace(active_trace)
                return IngestionResult(
                    skipped=True,
                    source_path=resolved_source,
                    collection=collection_name,
                    trace_id=active_trace.trace_id,
                )

            run_settings = self._settings_for_collection(collection_name)
            document = self._run_load(
                resolved_source,
                run_settings,
                on_progress=on_progress,
                trace=active_trace,
            )
            image_count = self._sync_document_images(document, collection_name, run_settings)

            chunks = self._run_split(document, run_settings, on_progress=on_progress, trace=active_trace)
            chunks = self._annotate_chunks(chunks, collection_name)
            chunks = self._run_transform(chunks, run_settings, on_progress=on_progress, trace=active_trace)

            encoding = self._run_encode(
                chunks,
                run_settings,
                on_progress=on_progress,
                trace=active_trace,
            )
            chunk_ids = self._run_store(
                chunks,
                encoding.dense_vectors,
                encoding.sparse_stats,
                run_settings,
                on_progress=on_progress,
                trace=active_trace,
            )

            self._integrity_checker.mark_success(
                file_hash,
                resolved_source,
                chunk_count=len(chunks),
            )
            active_trace.record_stage(
                "pipeline_complete",
                elapsed_ms=0.0,
                method="pipeline",
                chunk_count=len(chunks),
                image_count=image_count,
                source_path=resolved_source,
                collection=collection_name,
            )
            active_trace.finish()
            self._persist_trace(active_trace)

            logger.info(
                "摄取完成: path=%s collection=%s chunks=%d images=%d",
                resolved_source,
                collection_name,
                len(chunks),
                image_count,
            )
            return IngestionResult(
                skipped=False,
                source_path=resolved_source,
                collection=collection_name,
                document_id=document.id,
                chunk_count=len(chunks),
                image_count=image_count,
                chunk_ids=chunk_ids,
                trace_id=active_trace.trace_id,
            )
        except IngestionPipelineError as exc:
            self._persist_failed_trace(active_trace, exc.stage, str(exc))
            raise
        except Exception as exc:
            self._persist_failed_trace(active_trace, "pipeline", str(exc))
            raise IngestionPipelineError("pipeline", str(exc)) from exc

    def _run_integrity_precheck(
        self,
        source_path: str,
        *,
        force: bool,
        on_progress: ProgressCallback | None,
    ) -> str | None:
        """计算 hash 并判定是否跳过；返回 hash 或 None（表示跳过）。"""
        self._notify_progress(on_progress, "integrity", 1, 1)
        try:
            file_hash = self._integrity_checker.compute_sha256(source_path)
        except Exception as exc:
            raise IngestionPipelineError("integrity", str(exc)) from exc

        if not force and self._integrity_checker.should_skip(file_hash):
            return None
        return file_hash

    def _run_load(
        self,
        source_path: str,
        settings: Settings,
        *,
        on_progress: ProgressCallback | None,
        trace: TraceContext,
    ) -> Document:
        """加载源文件为 Document。"""
        self._notify_progress(on_progress, "load", 1, 1)
        start = time.perf_counter()
        loader = self._loader or self._resolve_loader(source_path, settings)
        try:
            document = loader.load(source_path)
        except LoaderError as exc:
            raise IngestionPipelineError("load", str(exc)) from exc
        except Exception as exc:
            raise IngestionPipelineError("load", str(exc)) from exc

        # F4：load 阶段 method 对齐实际解析器（PDF 走 MarkItDown）
        trace.record_stage(
            "load",
            elapsed_ms=(time.perf_counter() - start) * 1000,
            method=self._resolve_load_method(loader),
            source_path=source_path,
            doc_id=document.id,
        )
        logger.info("阶段 load 完成: doc_id=%s", document.id)
        return document

    def _run_split(
        self,
        document: Document,
        settings: Settings,
        *,
        on_progress: ProgressCallback | None,
        trace: TraceContext,
    ) -> list[Chunk]:
        """Document 切分为 Chunk 列表。"""
        self._notify_progress(on_progress, "split", 1, 1)
        start = time.perf_counter()
        chunker = self._chunker or DocumentChunker(settings)
        try:
            chunks = chunker.split_document(document)
        except Exception as exc:
            raise IngestionPipelineError("split", str(exc)) from exc

        splitter_name = (
            settings.ingestion.splitter if settings.ingestion is not None else "recursive"
        )
        trace.record_stage(
            "split",
            elapsed_ms=(time.perf_counter() - start) * 1000,
            method=splitter_name,
            chunk_count=len(chunks),
        )
        logger.info("阶段 split 完成: chunks=%d", len(chunks))
        return chunks

    def _run_transform(
        self,
        chunks: list[Chunk],
        settings: Settings,
        *,
        on_progress: ProgressCallback | None,
        trace: TraceContext,
    ) -> list[Chunk]:
        """依次执行 Transform 链，整条链记为规范阶段 ``transform``。"""
        transforms = self._transforms or self._default_transforms(settings)
        total = len(transforms) or 1
        current_chunks = chunks
        chain_start = time.perf_counter()
        transform_names: list[str] = []
        for index, transform in enumerate(transforms):
            self._notify_progress(on_progress, "transform", index + 1, total)
            stage_name = transform.__class__.__name__
            transform_names.append(stage_name)
            try:
                current_chunks = transform.transform(current_chunks, trace=trace)
            except Exception as exc:
                raise IngestionPipelineError(stage_name, str(exc)) from exc
            logger.info("阶段 %s 完成: chunks=%d", stage_name, len(current_chunks))

        trace.record_stage(
            "transform",
            elapsed_ms=(time.perf_counter() - chain_start) * 1000,
            method="sequential",
            transforms=transform_names,
            chunk_count=len(current_chunks),
        )
        return current_chunks

    def _run_encode(
        self,
        chunks: list[Chunk],
        settings: Settings,
        *,
        on_progress: ProgressCallback | None,
        trace: TraceContext,
    ) -> BatchEncodingResult:
        """批量 Dense/Sparse 编码。"""
        processor = self._batch_processor or BatchProcessor(settings)
        start = time.perf_counter()
        try:
            result = processor.process(chunks, trace=trace)
        except Exception as exc:
            raise IngestionPipelineError("encode", str(exc)) from exc

        batch_count = result.batch_count or 1
        # F5：编码完成时报告 embed 进度；current/total 对齐实际 batch 数
        self._notify_progress(on_progress, "embed", batch_count, batch_count)
        trace.record_stage(
            "embed",
            elapsed_ms=(time.perf_counter() - start) * 1000,
            method=settings.embedding.provider,
            provider=settings.embedding.provider,
            batch_count=result.batch_count,
            chunk_count=len(chunks),
        )
        logger.info(
            "阶段 encode 完成: batches=%d chunks=%d",
            result.batch_count,
            len(chunks),
        )
        return result

    def _run_store(
        self,
        chunks: list[Chunk],
        dense_vectors: Sequence[Sequence[float]],
        sparse_stats: Sequence[Any],
        settings: Settings,
        *,
        on_progress: ProgressCallback | None,
        trace: TraceContext,
    ) -> list[str]:
        """写入向量库与 BM25 索引。"""
        self._notify_progress(on_progress, "upsert", 1, 1)
        start = time.perf_counter()

        # 已注入 upserter 时不再创建 VectorStore，便于单元测试隔离
        upserter = self._vector_upserter
        if upserter is None:
            vector_store = self._vector_store or VectorStoreFactory.create(settings)
            upserter = VectorUpserter(vector_store)
        bm25 = self._bm25_indexer or BM25Indexer(
            collection=settings.vector_store.collection_name,
            index_root=self._bm25_root,
        )

        try:
            chunk_ids = upserter.upsert(chunks, dense_vectors, trace=trace)
            bm25.add(list(sparse_stats))
            bm25.save()
        except Exception as exc:
            raise IngestionPipelineError("store", str(exc)) from exc

        # F4 规范阶段名为 upsert
        trace.record_stage(
            "upsert",
            elapsed_ms=(time.perf_counter() - start) * 1000,
            method=settings.vector_store.provider,
            provider=settings.vector_store.provider,
            vector_count=len(chunk_ids),
        )
        logger.info("阶段 store 完成: vectors=%d", len(chunk_ids))
        return chunk_ids

    def _sync_document_images(
        self,
        document: Document,
        collection: str,
        settings: Settings,
    ) -> int:
        """将 Document 级图片同步到 ImageStorage 索引（若已配置）。"""
        images = document.metadata.get("images")
        if not images:
            return 0

        storage = self._image_storage or ImageStorage(
            images_root=self._images_root or self._default_images_root(settings),
        )
        doc_hash = str(document.metadata.get("doc_hash", document.id))
        saved = 0

        for item in images:
            meta = self._normalize_image_metadata(item)
            if meta is None:
                continue
            image_path = Path(meta.path)
            if not image_path.is_file():
                continue
            storage.save_image(
                image_id=meta.id,
                data=image_path.read_bytes(),
                collection=collection,
                doc_hash=doc_hash,
                page_num=meta.page,
            )
            saved += 1
        return saved

    def _settings_for_collection(self, collection: str) -> Settings:
        """按运行集合覆盖 vector_store.collection_name 与可选 Chroma 目录。"""
        vector_store = self._base_settings.vector_store
        persist_directory = self._chroma_persist_directory or vector_store.persist_directory
        updated_vector_store = VectorStoreSettings(
            provider=vector_store.provider,
            persist_directory=persist_directory,
            collection_name=collection,
        )
        return Settings(
            llm=self._base_settings.llm,
            embedding=self._base_settings.embedding,
            vector_store=updated_vector_store,
            retrieval=self._base_settings.retrieval,
            rerank=self._base_settings.rerank,
            evaluation=self._base_settings.evaluation,
            observability=self._base_settings.observability,
            ingestion=self._base_settings.ingestion,
            vision_llm=self._base_settings.vision_llm,
        )

    @staticmethod
    def _resolve_load_method(loader: BaseLoader) -> str:
        """推断 load 阶段 method：PDF 默认 MarkItDown，其它用类名。"""
        if isinstance(loader, PdfLoader):
            return "markitdown"
        return loader.__class__.__name__

    def _resolve_loader(self, source_path: str, settings: Settings) -> BaseLoader:
        """按扩展名选择 Loader（MVP 仅 PDF）。"""
        suffix = Path(source_path).suffix.lower()
        if suffix == ".pdf":
            images_root = self._images_root or self._default_images_root(settings)
            return PdfLoader(images_root=images_root)
        raise IngestionPipelineError(
            "load",
            f"暂不支持的文件类型: {suffix}（当前 MVP 仅支持 .pdf）",
        )

    def _default_images_root(self, settings: Settings) -> Path:
        return resolve_path("data/images")

    @staticmethod
    def _default_transforms(settings: Settings) -> list[BaseTransform]:
        return [
            ChunkRefiner(settings),
            MetadataEnricher(settings),
            ImageCaptioner(settings),
        ]

    @staticmethod
    def _annotate_chunks(chunks: Sequence[Chunk], collection: str) -> list[Chunk]:
        """为每个 Chunk metadata 注入 collection 字段。"""
        annotated: list[Chunk] = []
        for chunk in chunks:
            metadata = dict(chunk.metadata)
            metadata["collection"] = collection
            annotated.append(
                Chunk(
                    id=chunk.id,
                    text=chunk.text,
                    metadata=metadata,
                    start_offset=chunk.start_offset,
                    end_offset=chunk.end_offset,
                    source_ref=chunk.source_ref,
                )
            )
        return annotated

    @staticmethod
    def _normalize_image_metadata(item: Any) -> ImageMetadata | None:
        if isinstance(item, ImageMetadata):
            return item
        if isinstance(item, Mapping):
            try:
                return ImageMetadata.from_dict(item)
            except Exception:
                return None
        return None

    def _persist_trace(self, trace: TraceContext) -> None:
        """把已 finish 的 trace 追加到 traces.jsonl；失败不影响摄取主流程。"""
        if not self._base_settings.observability.trace_enabled:
            return
        try:
            write_trace(trace.to_dict())
        except Exception:
            logger.warning("写入 traces.jsonl 失败", exc_info=True)

    def _persist_failed_trace(self, trace: TraceContext, stage: str, message: str) -> None:
        """失败时补记 error 阶段再落盘，供 Dashboard 展示失败状态。"""
        try:
            trace.record_stage("error", elapsed_ms=0.0, method=stage, error=message)
            if not trace.is_finished:
                trace.finish()
            self._persist_trace(trace)
        except Exception:
            logger.warning("写入失败 trace 失败", exc_info=True)

    @staticmethod
    def _notify_progress(
        callback: ProgressCallback | None,
        stage: str,
        current: int,
        total: int,
    ) -> None:
        """
        触发进度回调；未传入时静默跳过。

        Args:
            callback: F5 进度回调，签名 ``(stage_name, current, total)``。
            stage: 规范阶段名。
            current: 当前进度（从 1 计）。
            total: 该阶段总量。
        """
        if callback is not None:
            callback(stage, current, total)
