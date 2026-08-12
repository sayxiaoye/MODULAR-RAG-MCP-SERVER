"""ImageStorage：图片文件落盘与 SQLite image_id→path 索引。"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.settings import REPO_ROOT, resolve_path

DEFAULT_IMAGES_ROOT = REPO_ROOT / "data" / "images"
DEFAULT_IMAGE_INDEX_DB = REPO_ROOT / "data" / "db" / "image_index.db"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS image_index (
    image_id TEXT PRIMARY KEY,
    file_path TEXT NOT NULL,
    collection TEXT,
    doc_hash TEXT,
    page_num INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_collection ON image_index(collection);
CREATE INDEX IF NOT EXISTS idx_doc_hash ON image_index(doc_hash);
"""


class ImageStorageError(Exception):
    """图片保存或索引读写失败时抛出。"""


@dataclass(frozen=True)
class ImageIndexRecord:
    """image_index 表的单行查询结果。"""

    image_id: str
    file_path: str
    collection: str | None
    doc_hash: str | None
    page_num: int | None
    created_at: str | None


class ImageStorage:
    """
    将图片保存到 data/images/{collection}/，并在 SQLite 中维护 image_id 映射。

    供摄取 Pipeline 与 Dashboard DataService 查询图片路径。
    """

    def __init__(
        self,
        images_root: str | Path | None = None,
        db_path: str | Path | None = None,
    ) -> None:
        if images_root is None:
            resolved_root = DEFAULT_IMAGES_ROOT
        else:
            candidate_root = Path(images_root)
            resolved_root = (
                candidate_root if candidate_root.is_absolute() else resolve_path(candidate_root)
            )

        if db_path is None:
            resolved_db = DEFAULT_IMAGE_INDEX_DB
        else:
            candidate_db = Path(db_path)
            resolved_db = (
                candidate_db if candidate_db.is_absolute() else resolve_path(candidate_db)
            )

        self.images_root = resolved_root
        self.db_path = resolved_db
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        """创建连接并启用 WAL，支持并发读写。"""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    def save_image(
        self,
        image_id: str,
        data: bytes,
        collection: str = "default",
        doc_hash: str | None = None,
        page_num: int | None = None,
    ) -> Path:
        """
        保存图片字节并写入索引；同 image_id 重复保存会覆盖文件与索引。

        Args:
            image_id: 全局唯一图片标识。
            data: 图片二进制内容。
            collection: 集合名，决定子目录 data/images/{collection}/。
            doc_hash: 可选来源文档哈希，便于按文档批量查询。
            page_num: 可选页码。

        Returns:
            落盘后的绝对路径。
        """
        if not image_id or not image_id.strip():
            raise ImageStorageError("image_id 不能为空")
        if not data:
            raise ImageStorageError("图片数据不能为空")
        if not collection or not collection.strip():
            raise ImageStorageError("collection 不能为空")

        collection_name = collection.strip()
        output_dir = self.images_root / collection_name
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / f"{image_id.strip()}.png"
        file_path.write_bytes(data)
        resolved_path = str(file_path.resolve())

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO image_index
                    (image_id, file_path, collection, doc_hash, page_num, created_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (image_id.strip(), resolved_path, collection_name, doc_hash, page_num),
            )

        return file_path

    def get_path(self, image_id: str) -> Path | None:
        """根据 image_id 查询本地文件路径；不存在时返回 None。"""
        if not image_id or not image_id.strip():
            raise ImageStorageError("image_id 不能为空")

        with self._connect() as conn:
            row = conn.execute(
                "SELECT file_path FROM image_index WHERE image_id = ?",
                (image_id.strip(),),
            ).fetchone()

        if row is None:
            return None

        path = Path(row["file_path"])
        return path if path.is_file() else None

    def list_images(
        self,
        collection: str | None = None,
        doc_hash: str | None = None,
    ) -> list[ImageIndexRecord]:
        """
        按 collection / doc_hash 过滤查询图片索引。

        两者均为 None 时返回全部记录。
        """
        query = "SELECT image_id, file_path, collection, doc_hash, page_num, created_at FROM image_index"
        clauses: list[str] = []
        params: list[Any] = []

        if collection is not None:
            clauses.append("collection = ?")
            params.append(collection)
        if doc_hash is not None:
            clauses.append("doc_hash = ?")
            params.append(doc_hash)

        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY image_id"

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            ImageIndexRecord(
                image_id=str(row["image_id"]),
                file_path=str(row["file_path"]),
                collection=row["collection"],
                doc_hash=row["doc_hash"],
                page_num=row["page_num"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def delete_images(
        self,
        collection: str,
        doc_hash: str | None = None,
    ) -> int:
        """
        删除指定 collection（及可选 doc_hash）下的图片文件与索引行。

        Returns:
            删除的图片数量。
        """
        if not collection or not collection.strip():
            raise ImageStorageError("collection 不能为空")

        records = self.list_images(collection=collection.strip(), doc_hash=doc_hash)
        deleted = 0
        for record in records:
            path = Path(record.file_path)
            if path.is_file():
                path.unlink()
                deleted += 1

        query = "DELETE FROM image_index WHERE collection = ?"
        params: list[Any] = [collection.strip()]
        if doc_hash is not None:
            query += " AND doc_hash = ?"
            params.append(doc_hash)

        with self._connect() as conn:
            conn.execute(query, params)

        return deleted
