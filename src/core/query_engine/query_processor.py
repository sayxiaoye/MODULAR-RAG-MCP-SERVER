"""QueryProcessor：查询预处理，提取关键词并组装 filters。"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from core.trace.trace_context import TraceContext

# 与 SparseEncoder / BM25Indexer 对齐的分词规则
_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9\u4e00-\u9fff]+")

# 规则阶段停用词（MVP：去功能词，保留实体与术语）
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "am",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "what",
        "which",
        "who",
        "whom",
        "where",
        "when",
        "why",
        "how",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "as",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "too",
        "very",
        "just",
        "also",
        "now",
        "的",
        "了",
        "和",
        "是",
        "在",
        "有",
        "我",
        "你",
        "他",
        "她",
        "它",
        "我们",
        "你们",
        "他们",
        "这",
        "那",
        "哪",
        "什么",
        "怎么",
        "如何",
        "为什么",
        "吗",
        "呢",
        "吧",
        "啊",
        "呀",
        "一个",
        "一些",
        "这个",
        "那个",
        "这些",
        "那些",
        "以及",
        "或者",
        "但是",
        "因为",
        "所以",
        "如果",
        "虽然",
        "而",
        "与",
        "及",
        "对",
        "把",
        "被",
        "让",
        "给",
        "从",
        "到",
        "向",
        "为",
        "以",
        "就",
        "都",
        "还",
        "也",
        "很",
        "更",
        "最",
        "非常",
        "比较",
        "已经",
        "可以",
        "可能",
        "应该",
        "需要",
        "进行",
        "使用",
        "关于",
        "什么是",
        "是什么",
        "有没有",
        "能不能",
        "可不可以",
    }
)

# 支持 query 内嵌 ``field:value`` 约束（MVP 可空扩展）
_FILTER_PATTERN = re.compile(
    r"(?<!\w)(collection|doc_type|language|access_level):([^\s]+)",
    re.IGNORECASE,
)


class QueryProcessorError(Exception):
    """查询预处理失败时抛出。"""


@dataclass(frozen=True)
class ProcessedQuery:
    """
    QueryProcessor 输出契约，供 Dense/Sparse 检索与 HybridSearch 消费。

    Attributes:
        original_query: 用户原始查询。
        dense_query: 稠密检索用 query（MVP 保持原句，轻度 trim）。
        keywords: 稀疏检索用词项列表（去停用词后）。
        filters: 结构化过滤条件（collection / doc_type 等）。
    """

    original_query: str
    dense_query: str
    keywords: list[str]
    filters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "dense_query": self.dense_query,
            "keywords": list(self.keywords),
            "filters": dict(self.filters),
        }


class QueryProcessor:
    """规则分词 + 停用词过滤，解析可选 filters（MVP 可空实现）。"""

    def process(
        self,
        query: str,
        filters: Mapping[str, Any] | None = None,
        trace: Any | None = None,
    ) -> ProcessedQuery:
        """
        将自然语言 query 转为检索可用的结构化表示。

        Args:
            query: 已消歧的独立查询文本。
            filters: 调用方传入的硬过滤条件，将与 query 内嵌约束合并。
            trace: 可选 TraceContext，记录预处理耗时。

        Returns:
            ProcessedQuery，keywords 非空（停用词全过滤时回退为原始词项）。
        """
        if not isinstance(query, str):
            raise QueryProcessorError("query 必须是字符串")

        stripped = query.strip()
        if not stripped:
            raise QueryProcessorError("query 不能为空")

        parsed_filters = self._parse_inline_filters(stripped)
        merged_filters = self._merge_filters(filters, parsed_filters)
        query_for_tokenize = self._strip_inline_filters(stripped)

        keywords = self._extract_keywords(query_for_tokenize)
        dense_query = query_for_tokenize.strip() or stripped

        if isinstance(trace, TraceContext):
            trace.record_stage(
                "query_processor",
                keyword_count=len(keywords),
                filter_keys=sorted(merged_filters.keys()),
            )

        return ProcessedQuery(
            original_query=stripped,
            dense_query=dense_query,
            keywords=keywords,
            filters=merged_filters,
        )

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        if not text:
            return []
        return _TOKEN_PATTERN.findall(text.lower())

    def _extract_keywords(self, text: str) -> list[str]:
        """分词并去停用词；若全部被过滤则回退为原始词项以保证非空。"""
        tokens = self._tokenize(text)
        if not tokens:
            return []

        keywords = [
            token
            for token in tokens
            if token not in _STOPWORDS and len(token) > 1
        ]
        if keywords:
            return keywords
        return tokens

    @staticmethod
    def _parse_inline_filters(query: str) -> dict[str, Any]:
        """从 query 中解析 ``collection:xxx`` 等内嵌约束（MVP 轻量实现）。"""
        filters: dict[str, Any] = {}
        for match in _FILTER_PATTERN.finditer(query):
            field = match.group(1).lower()
            value = match.group(2).strip()
            if value:
                filters[field] = value
        return filters

    @staticmethod
    def _strip_inline_filters(query: str) -> str:
        """移除已解析的 inline filter 片段，避免污染关键词。"""
        cleaned = _FILTER_PATTERN.sub("", query)
        return re.sub(r"\s+", " ", cleaned).strip()

    @staticmethod
    def _merge_filters(
        explicit: Mapping[str, Any] | None,
        parsed: Mapping[str, Any],
    ) -> dict[str, Any]:
        """显式 filters 优先于 query 内嵌解析结果。"""
        merged: dict[str, Any] = dict(parsed)
        if explicit:
            for key, value in explicit.items():
                if value is not None:
                    merged[str(key)] = value
        return merged
