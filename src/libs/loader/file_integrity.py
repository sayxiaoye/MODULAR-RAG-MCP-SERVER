"""文件完整性检查：SHA256 哈希计算与 SQLite 摄取历史记录。"""

from __future__ import annotations

import hashlib
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path

from core.settings import REPO_ROOT, resolve_path

DEFAULT_INGESTION_HISTORY_DB = REPO_ROOT / "data" / "db" / "ingestion_history.db"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ingestion_history (
    file_hash TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    file_size INTEGER,
    status TEXT NOT NULL CHECK(status IN ('success', 'failed', 'processing')),
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    error_msg TEXT,
    chunk_count INTEGER
);
CREATE INDEX IF NOT EXISTS idx_status ON ingestion_history(status);
CREATE INDEX IF NOT EXISTS idx_processed_at ON ingestion_history(processed_at);
"""


class FileIntegrityError(Exception):
    """文件哈希计算或完整性记录失败时抛出。"""


class FileIntegrityChecker(ABC):
    """文件完整性检查抽象接口，供 Pipeline 增量跳过判定。"""

    @abstractmethod
    def compute_sha256(self, path: str) -> str:
        """计算文件 SHA256 十六进制摘要。"""

    @abstractmethod
    def should_skip(self, file_hash: str) -> bool:
        """若该 hash 已成功处理过，则返回 True（可跳过后续摄取）。"""

    @abstractmethod
    def mark_success(
        self,
        file_hash: str,
        file_path: str,
        chunk_count: int | None = None,
        file_size: int | None = None,
    ) -> None:
        """记录文件成功摄取，供后续增量跳过。"""

    @abstractmethod
    def mark_failed(
        self,
        file_hash: str,
        error_msg: str,
        file_path: str | None = None,
    ) -> None:
        """记录摄取失败原因，不触发 should_skip。"""


class SQLiteIntegrityChecker(FileIntegrityChecker):
    """基于 SQLite 的默认完整性检查器，数据库位于 data/db/ingestion_history.db。"""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            resolved = DEFAULT_INGESTION_HISTORY_DB
        else:
            candidate = Path(db_path)
            resolved = candidate if candidate.is_absolute() else resolve_path(candidate)
        self.db_path = resolved
        # 确保 data/db 目录存在，避免首次摄取时初始化失败
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        """创建连接并启用 WAL，支持并发读写。"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    def compute_sha256(self, path: str) -> str:
        file_path = Path(path)
        if not file_path.is_file():
            raise FileIntegrityError(f"文件不存在或不可读: {path}")

        digest = hashlib.sha256()
        # 分块读取，避免大文件一次性占用过多内存
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def should_skip(self, file_hash: str) -> bool:
        if not file_hash:
            raise FileIntegrityError("file_hash 不能为空")
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM ingestion_history WHERE file_hash = ?",
                (file_hash,),
            ).fetchone()
        # 仅 success 状态视为已处理完成，failed/processing 不跳过
        return row is not None and row[0] == "success"

    def mark_success(
        self,
        file_hash: str,
        file_path: str,
        chunk_count: int | None = None,
        file_size: int | None = None,
    ) -> None:
        if not file_hash:
            raise FileIntegrityError("file_hash 不能为空")
        resolved_size = file_size
        if resolved_size is None and Path(file_path).is_file():
            resolved_size = Path(file_path).stat().st_size

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO ingestion_history
                    (file_hash, file_path, file_size, status, error_msg, chunk_count, processed_at)
                VALUES (?, ?, ?, 'success', NULL, ?, CURRENT_TIMESTAMP)
                """,
                (file_hash, file_path, resolved_size, chunk_count),
            )

    def mark_failed(
        self,
        file_hash: str,
        error_msg: str,
        file_path: str | None = None,
    ) -> None:
        if not file_hash:
            raise FileIntegrityError("file_hash 不能为空")
        if not error_msg:
            raise FileIntegrityError("error_msg 不能为空")

        stored_path = file_path or ""
        file_size = Path(file_path).stat().st_size if file_path and Path(file_path).is_file() else None
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO ingestion_history
                    (file_hash, file_path, file_size, status, error_msg, chunk_count, processed_at)
                VALUES (?, ?, ?, 'failed', ?, NULL, CURRENT_TIMESTAMP)
                """,
                (file_hash, stored_path, file_size, error_msg),
            )
