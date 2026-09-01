"""file_integrity 模块单元测试。"""

from __future__ import annotations

import concurrent.futures
import tempfile
from pathlib import Path

import pytest

from libs.loader.file_integrity import FileIntegrityError, SQLiteIntegrityChecker


@pytest.fixture
def integrity_checker(tmp_path: Path) -> SQLiteIntegrityChecker:
    """创建独立 SQLite 检查器，并在 teardown 时释放 WAL 文件锁。"""
    db_path = tmp_path / "ingestion_history.db"
    checker = SQLiteIntegrityChecker(db_path=db_path)
    yield checker
    # 测试结束后 checkpoint，避免 Windows 上 WAL 文件无法删除
    with checker._connect() as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


@pytest.mark.unit
class TestSQLiteIntegrityChecker:
    """验证 SHA256 计算与摄取历史读写。"""

    def test_compute_sha256_is_stable(self, integrity_checker: SQLiteIntegrityChecker) -> None:
        """同一文件多次计算应得到相同 hash。"""
        with tempfile.NamedTemporaryFile("wb", delete=False) as tmp:
            tmp.write(b"stable-content")
            file_path = tmp.name
        try:
            first = integrity_checker.compute_sha256(file_path)
            second = integrity_checker.compute_sha256(file_path)
            assert first == second
            assert len(first) == 64
        finally:
            Path(file_path).unlink(missing_ok=True)

    def test_should_skip_after_mark_success(self, integrity_checker: SQLiteIntegrityChecker) -> None:
        """标记 success 后应触发跳过。"""
        file_hash = "abc123hash"
        assert integrity_checker.should_skip(file_hash) is False
        integrity_checker.mark_success(file_hash, "docs/sample.pdf", chunk_count=12, file_size=1024)
        assert integrity_checker.should_skip(file_hash) is True

    def test_mark_failed_does_not_skip(self, integrity_checker: SQLiteIntegrityChecker) -> None:
        """失败记录不应导致 should_skip 为 True。"""
        file_hash = "failed-hash"
        integrity_checker.mark_failed(file_hash, "parse error", file_path="docs/bad.pdf")
        assert integrity_checker.should_skip(file_hash) is False

    def test_db_file_created(self, integrity_checker: SQLiteIntegrityChecker, tmp_path: Path) -> None:
        """初始化后应创建数据库文件。"""
        integrity_checker.mark_success("hash-db", "a.pdf")
        assert (tmp_path / "ingestion_history.db").is_file()

    def test_concurrent_writes(self, integrity_checker: SQLiteIntegrityChecker) -> None:
        """并发写入不同 hash 记录应全部成功。"""
        def _mark(index: int) -> None:
            integrity_checker.mark_success(f"hash-{index}", f"doc-{index}.pdf", chunk_count=index)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(_mark, range(8)))

        for index in range(8):
            assert integrity_checker.should_skip(f"hash-{index}") is True

    def test_should_skip_is_scoped_to_collection(
        self, integrity_checker: SQLiteIntegrityChecker
    ) -> None:
        """同一文件只应在已成功摄入的集合上跳过。"""
        file_hash = "same-pdf-hash"
        integrity_checker.mark_success(
            file_hash, "docs/week1.pdf", chunk_count=10, collection="col_a"
        )
        assert integrity_checker.should_skip(file_hash, collection="col_a") is True
        assert integrity_checker.should_skip(file_hash, collection="col_b") is False

    def test_legacy_row_without_collection_does_not_skip_named_target(
        self, integrity_checker: SQLiteIntegrityChecker
    ) -> None:
        """旧表迁移后 collection 为空的记录，不应挡住新集合摄入。"""
        file_hash = "legacy-hash"
        integrity_checker.mark_success(file_hash, "docs/week1.pdf", chunk_count=10)
        assert integrity_checker.should_skip(file_hash) is True
        assert integrity_checker.should_skip(file_hash, collection="col_b") is False

    def test_migrates_pre_collection_schema(self, tmp_path: Path) -> None:
        """旧库只有 file_hash 主键时，打开检查器应迁移且不误跳过新集合。"""
        import sqlite3

        db_path = tmp_path / "legacy_history.db"
        with sqlite3.connect(db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE ingestion_history (
                    file_hash TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_size INTEGER,
                    status TEXT NOT NULL,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    error_msg TEXT,
                    chunk_count INTEGER
                );
                """
            )
            conn.execute(
                """
                INSERT INTO ingestion_history (file_hash, file_path, status, chunk_count)
                VALUES (?, ?, 'success', 10)
                """,
                ("legacy-pdf-hash", "week1.pdf"),
            )
        checker = SQLiteIntegrityChecker(db_path=db_path)
        try:
            assert checker.should_skip("legacy-pdf-hash") is True
            assert checker.should_skip("legacy-pdf-hash", collection="col_b") is False
        finally:
            with checker._connect() as conn:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def test_compute_sha256_missing_file_raises(self, integrity_checker: SQLiteIntegrityChecker) -> None:
        """文件不存在时应抛出 FileIntegrityError。"""
        with pytest.raises(FileIntegrityError, match="不存在"):
            integrity_checker.compute_sha256("not-exists.pdf")
