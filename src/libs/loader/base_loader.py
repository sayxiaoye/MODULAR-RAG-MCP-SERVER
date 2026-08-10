"""Loader 抽象层：将原始文件解析为 core.types.Document。"""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.types import Document


class LoaderError(Exception):
    """Loader 解析或文件校验失败时抛出。"""


class BaseLoader(ABC):
    """Loader 抽象基类：屏蔽 PDF 等格式的解析差异。"""

    @abstractmethod
    def load(self, path: str) -> Document:
        """
        加载单个文件并产出标准 Document。

        Args:
            path: 待加载文件的本地路径。

        Returns:
            符合 core.types 契约的 Document 对象。
        """
        ...
