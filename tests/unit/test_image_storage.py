"""ImageStorage 单元测试：落盘、索引查询与持久化。"""

from __future__ import annotations

import concurrent.futures
from pathlib import Path

import pytest

from ingestion.storage.image_storage import ImageStorage, ImageStorageError


@pytest.fixture
def image_storage(tmp_path: Path) -> ImageStorage:
    """创建独立图片根目录与 SQLite 索引，teardown 时释放 WAL 锁。"""
    storage = ImageStorage(
        images_root=tmp_path / "images",
        db_path=tmp_path / "image_index.db",
    )
    yield storage
    with storage._connect() as conn:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


@pytest.mark.unit
class TestImageStorage:
    """验证图片保存、路径查询与索引持久化。"""

    def test_save_image_creates_file(self, image_storage: ImageStorage, tmp_path: Path) -> None:
        """保存后文件应存在于 data/images/{collection}/{image_id}.png。"""
        path = image_storage.save_image(
            image_id="img_001",
            data=b"png-bytes",
            collection="default",
            doc_hash="doc_hash_abc",
            page_num=1,
        )
        expected = tmp_path / "images" / "default" / "img_001.png"
        assert path == expected
        assert expected.is_file()
        assert expected.read_bytes() == b"png-bytes"

    def test_get_path_returns_saved_path(self, image_storage: ImageStorage) -> None:
        """get_path 应返回已保存图片的绝对路径。"""
        saved = image_storage.save_image("img_lookup", b"data", collection="coll_a")
        resolved = image_storage.get_path("img_lookup")
        assert resolved is not None
        assert resolved.resolve() == saved.resolve()

    def test_index_persists_across_instances(self, tmp_path: Path) -> None:
        """新实例加载同一数据库后应能查到映射。"""
        db_path = tmp_path / "image_index.db"
        images_root = tmp_path / "images"
        first = ImageStorage(images_root=images_root, db_path=db_path)
        first.save_image("persist_img", b"persist", collection="default", doc_hash="hash1")

        second = ImageStorage(images_root=images_root, db_path=db_path)
        path = second.get_path("persist_img")
        assert path is not None
        assert path.read_bytes() == b"persist"
        assert db_path.is_file()

    def test_list_images_by_collection(self, image_storage: ImageStorage) -> None:
        """应按 collection 过滤索引记录。"""
        image_storage.save_image("a1", b"a", collection="col_x", doc_hash="h1")
        image_storage.save_image("a2", b"b", collection="col_x", doc_hash="h2")
        image_storage.save_image("b1", b"c", collection="col_y", doc_hash="h1")

        records = image_storage.list_images(collection="col_x")
        assert [record.image_id for record in records] == ["a1", "a2"]

    def test_list_images_by_collection_and_doc_hash(self, image_storage: ImageStorage) -> None:
        """应支持 collection + doc_hash 联合过滤。"""
        image_storage.save_image("x1", b"x", collection="col", doc_hash="doc_a")
        image_storage.save_image("x2", b"y", collection="col", doc_hash="doc_b")

        records = image_storage.list_images(collection="col", doc_hash="doc_a")
        assert len(records) == 1
        assert records[0].image_id == "x1"

    def test_delete_images_removes_files_and_index(self, image_storage: ImageStorage) -> None:
        """delete_images 应删除文件并清理索引。"""
        image_storage.save_image("del1", b"d1", collection="del_col", doc_hash="dh")
        image_storage.save_image("del2", b"d2", collection="del_col", doc_hash="dh")

        deleted = image_storage.delete_images("del_col", doc_hash="dh")
        assert deleted == 2
        assert image_storage.get_path("del1") is None
        assert image_storage.get_path("del2") is None
        assert image_storage.list_images(collection="del_col") == []

    def test_get_path_missing_returns_none(self, image_storage: ImageStorage) -> None:
        """未索引的 image_id 应返回 None。"""
        assert image_storage.get_path("not_exists") is None

    def test_save_image_validation_errors(self, image_storage: ImageStorage) -> None:
        """非法输入应抛出 ImageStorageError。"""
        with pytest.raises(ImageStorageError, match="image_id"):
            image_storage.save_image("", b"x", collection="default")
        with pytest.raises(ImageStorageError, match="图片数据"):
            image_storage.save_image("id", b"", collection="default")

    def test_concurrent_saves(self, image_storage: ImageStorage) -> None:
        """并发保存不同 image_id 应全部成功。"""
        def _save(index: int) -> None:
            image_storage.save_image(
                f"concurrent_{index}",
                f"bytes-{index}".encode(),
                collection="default",
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(_save, range(8)))

        records = image_storage.list_images(collection="default")
        assert len(records) == 8
