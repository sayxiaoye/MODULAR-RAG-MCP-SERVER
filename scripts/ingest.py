#!/usr/bin/env python3
"""离线数据摄取 CLI：解析参数并调用 IngestionPipeline。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 支持直接执行 scripts/ingest.py 时能找到 src 包
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from core.settings import SettingsError, load_settings, resolve_path
from ingestion.pipeline import IngestionPipeline, IngestionPipelineError
from ingestion.storage.image_storage import ImageStorage
from libs.loader.file_integrity import SQLiteIntegrityChecker
from observability.logger import get_logger


def build_arg_parser() -> argparse.ArgumentParser:
    """构建 ingest 命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        description="Modular RAG MCP Server — 离线 PDF 摄取入口",
    )
    parser.add_argument(
        "--path",
        required=True,
        help="待摄取 PDF 文件路径，或包含 PDF 的目录",
    )
    parser.add_argument(
        "--collection",
        default="default",
        help="目标集合名（写入 metadata 与各存储后端）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新摄取，忽略文件完整性跳过判定",
    )
    parser.add_argument(
        "--settings",
        default=None,
        help="可选 settings.yaml 路径，默认使用 config/settings.yaml",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="覆盖 data 根目录（默认 data/），测试时可指向临时目录",
    )
    return parser


def collect_pdf_paths(path: str) -> list[str]:
    """
    收集待摄取 PDF 路径列表。

    支持单文件或目录（仅扫描顶层 *.pdf）。
    """
    candidate = Path(path)
    if not candidate.exists():
        raise FileNotFoundError(f"路径不存在: {path}")

    if candidate.is_file():
        if candidate.suffix.lower() != ".pdf":
            raise ValueError(f"仅支持 PDF 文件: {path}")
        return [str(candidate.resolve())]

    if candidate.is_dir():
        pdfs = sorted(candidate.glob("*.pdf"))
        return [str(item.resolve()) for item in pdfs]

    raise FileNotFoundError(f"无法读取路径: {path}")


def _resolve_data_paths(data_root: str | Path | None) -> dict[str, Path]:
    """根据 data 根目录计算各存储子路径。"""
    base = resolve_path(data_root) if data_root is not None else resolve_path("data")
    db_root = base / "db"
    return {
        "chroma": db_root / "chroma",
        "bm25": db_root / "bm25",
        "images": base / "images",
        "integrity_db": db_root / "ingestion_history.db",
        "image_db": db_root / "image_index.db",
    }


def run_ingest(
    path: str,
    collection: str = "default",
    force: bool = False,
    settings_path: str | None = None,
    data_root: str | None = None,
) -> int:
    """
    执行摄取任务。

    Returns:
        进程退出码：0 成功，1 配置或摄取失败。
    """
    logger = get_logger("ingest")

    try:
        settings = load_settings(settings_path)
    except SettingsError as exc:
        logger.error("配置加载失败: %s", exc)
        return 1

    if settings.ingestion is None:
        logger.error("配置缺少 ingestion 块，无法执行摄取")
        return 1

    data_paths = _resolve_data_paths(data_root)
    for directory in (
        data_paths["chroma"],
        data_paths["bm25"],
        data_paths["images"],
        data_paths["integrity_db"].parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    pipeline = IngestionPipeline(
        settings=settings,
        integrity_checker=SQLiteIntegrityChecker(db_path=data_paths["integrity_db"]),
        image_storage=ImageStorage(
            images_root=data_paths["images"],
            db_path=data_paths["image_db"],
        ),
        images_root=data_paths["images"],
        bm25_root=data_paths["bm25"],
        chroma_persist_directory=data_paths["chroma"],
    )

    try:
        pdf_paths = collect_pdf_paths(path)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    if not pdf_paths:
        logger.warning("未在 %s 下找到 PDF 文件", path)
        return 0

    exit_code = 0
    for pdf_path in pdf_paths:
        logger.info(
            "开始摄取: path=%s collection=%s force=%s",
            pdf_path,
            collection,
            force,
        )
        try:
            result = pipeline.run(
                pdf_path,
                collection=collection,
                force=force,
            )
        except IngestionPipelineError as exc:
            logger.error("摄取失败 [%s]: %s — file=%s", exc.stage, exc, pdf_path)
            exit_code = 1
            continue

        if result.skipped:
            logger.info("已跳过（未变更）: %s", pdf_path)
        else:
            logger.info(
                "摄取完成: path=%s chunks=%d images=%d trace_id=%s",
                pdf_path,
                result.chunk_count,
                result.image_count,
                result.trace_id,
            )

    return exit_code


def main(argv: list[str] | None = None) -> int:
    """CLI 入口：解析参数并调用 run_ingest。"""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return run_ingest(
        path=args.path,
        collection=args.collection,
        force=bool(args.force),
        settings_path=args.settings,
        data_root=args.data_root,
    )


if __name__ == "__main__":
    sys.exit(main())
