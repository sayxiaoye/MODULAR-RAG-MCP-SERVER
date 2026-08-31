"""BM25Indexer：构建倒排索引、计算 IDF 并持久化到 data/db/bm25/。"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from core.settings import REPO_ROOT, resolve_path
from ingestion.embedding.sparse_encoder import SparseChunkStats

DEFAULT_BM25_ROOT = REPO_ROOT / "data" / "db" / "bm25"
_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9\u4e00-\u9fff]+")
BM25_K1 = 1.5
BM25_B = 0.75


class BM25IndexerError(Exception):
    """BM25 索引构建、加载或查询失败时抛出。"""


def _tokenize(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    return _TOKEN_PATTERN.findall(text.lower())


def _compute_idf(document_count: int, document_frequency: int) -> float:
    """按 spec 公式计算 IDF：log((N - df + 0.5) / (df + 0.5))。"""
    if document_count <= 0:
        return 0.0
    return math.log((document_count - document_frequency + 0.5) / (document_frequency + 0.5))


class BM25Indexer:
    """接收 SparseChunkStats，构建可持久化的 BM25 倒排索引。"""

    def __init__(
        self,
        collection: str = "default",
        index_root: str | Path | None = None,
    ) -> None:
        root = Path(index_root) if index_root is not None else DEFAULT_BM25_ROOT
        if not root.is_absolute():
            root = resolve_path(root)
        self.index_root = root
        self.collection = collection
        self._index_path = self.index_root / f"{collection}.json"
        self._document_count = 0
        self._avg_doc_length = 0.0
        self._doc_lengths: dict[str, int] = {}
        # term -> {"idf": float, "postings": [{chunk_id, tf, doc_length}]}
        self._terms: dict[str, dict[str, Any]] = {}

    def build(
        self,
        stats_list: Sequence[SparseChunkStats],
        rebuild: bool = False,
    ) -> None:
        """
        构建或重建索引。

        Args:
            stats_list: SparseEncoder 产出的统计列表。
            rebuild: True 时清空旧索引后写入。
        """
        if rebuild:
            self._clear()
        self.add(stats_list)

    def add(self, stats_list: Sequence[SparseChunkStats]) -> None:
        """增量添加或更新 chunk 统计。"""
        for stat in stats_list:
            if stat.chunk_id in self._doc_lengths:
                self._remove_chunk(stat.chunk_id)
            self._doc_lengths[stat.chunk_id] = stat.doc_length
            for term, tf in stat.term_frequencies.items():
                bucket = self._terms.setdefault(term, {"postings": []})
                bucket["postings"].append(
                    {
                        "chunk_id": stat.chunk_id,
                        "tf": tf,
                        "doc_length": stat.doc_length,
                    }
                )
        self._recompute_metadata()

    def save(self) -> None:
        """将索引序列化到 data/db/bm25/{collection}.json。"""
        self.index_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "collection": self.collection,
            "N": self._document_count,
            "avg_doc_length": self._avg_doc_length,
            "doc_lengths": self._doc_lengths,
            "terms": self._terms,
        }
        self._index_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def load(self) -> None:
        """从磁盘加载索引。"""
        if not self._index_path.is_file():
            raise BM25IndexerError(f"BM25 索引不存在: {self._index_path}")
        payload = json.loads(self._index_path.read_text(encoding="utf-8"))
        self.collection = str(payload.get("collection", self.collection))
        self._document_count = int(payload.get("N", 0))
        self._avg_doc_length = float(payload.get("avg_doc_length", 0.0))
        doc_lengths = payload.get("doc_lengths", {})
        terms = payload.get("terms", {})
        if not isinstance(doc_lengths, Mapping) or not isinstance(terms, Mapping):
            raise BM25IndexerError("BM25 索引格式非法")
        self._doc_lengths = {str(k): int(v) for k, v in doc_lengths.items()}
        self._terms = {str(k): dict(v) for k, v in terms.items()}

    def query(self, query_text: str, top_k: int = 10) -> list[tuple[str, float]]:
        """
        对查询文本执行 BM25 打分，返回按分数降序的 chunk_id 列表。

        Args:
            query_text: 查询字符串。
            top_k: 返回的最大结果数。

        Returns:
            ``(chunk_id, score)`` 列表，分数降序、同分按 chunk_id 排序保证稳定。
        """
        if top_k <= 0:
            return []
        if self._document_count == 0:
            return []

        query_terms = _tokenize(query_text)
        if not query_terms:
            return []

        scores: dict[str, float] = {}
        avg_dl = self._avg_doc_length or 1.0

        for term in query_terms:
            term_data = self._terms.get(term)
            if not term_data:
                continue
            idf = float(term_data.get("idf", 0.0))
            postings = term_data.get("postings", [])
            if not isinstance(postings, list):
                continue
            for posting in postings:
                if not isinstance(posting, Mapping):
                    continue
                chunk_id = str(posting.get("chunk_id", "")).strip()
                if not chunk_id:
                    continue
                tf = int(posting.get("tf", 0))
                doc_length = int(posting.get("doc_length", 0))
                length_norm = 1 - BM25_B + BM25_B * (doc_length / avg_dl)
                denominator = tf + BM25_K1 * length_norm
                if denominator <= 0:
                    continue
                bm25_tf = (tf * (BM25_K1 + 1)) / denominator
                scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * bm25_tf

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:top_k]

    def remove_document(self, source: str, chunk_ids: Sequence[str] | None = None) -> None:
        """
        移除指定文档的倒排条目。

        Args:
            source: 文档 source_path；若未提供 chunk_ids，则删除 chunk_id 等于 source 的条目。
            chunk_ids: 可选，由 DocumentManager 从向量库查出的该文档 chunk 列表。
        """
        if not source or not str(source).strip():
            raise BM25IndexerError("source 不能为空")
        ids = [str(item).strip() for item in (chunk_ids or []) if str(item).strip()]
        if not ids:
            ids = [str(source).strip()]
        for chunk_id in ids:
            self._remove_chunk(chunk_id)
        self._recompute_metadata()

    def get_idf(self, term: str) -> float | None:
        """获取词项 IDF，便于单元测试验证。"""
        term_data = self._terms.get(term.lower())
        if term_data is None:
            return None
        return float(term_data.get("idf", 0.0))

    def _clear(self) -> None:
        self._document_count = 0
        self._avg_doc_length = 0.0
        self._doc_lengths = {}
        self._terms = {}

    def _remove_chunk(self, chunk_id: str) -> None:
        """从倒排表中移除指定 chunk 的所有 posting。"""
        self._doc_lengths.pop(chunk_id, None)
        for term_data in self._terms.values():
            postings = term_data.get("postings", [])
            if not isinstance(postings, list):
                continue
            term_data["postings"] = [
                posting
                for posting in postings
                if isinstance(posting, Mapping) and posting.get("chunk_id") != chunk_id
            ]
        # 清理无 posting 的空 term
        self._terms = {
            term: data for term, data in self._terms.items() if data.get("postings")
        }

    def _recompute_metadata(self) -> None:
        """根据当前文档集重新计算 N、平均长度与各 term 的 IDF。"""
        self._document_count = len(self._doc_lengths)
        if self._document_count == 0:
            self._avg_doc_length = 0.0
            self._terms = {}
            return
        self._avg_doc_length = sum(self._doc_lengths.values()) / self._document_count
        for term, data in self._terms.items():
            postings = data.get("postings", [])
            document_frequency = len(postings) if isinstance(postings, list) else 0
            data["idf"] = _compute_idf(self._document_count, document_frequency)
