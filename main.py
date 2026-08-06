"""Modular RAG MCP Server — 应用入口，启动时加载并校验配置。"""

from __future__ import annotations

import sys

from core.settings import load_settings, SettingsError
from observability.logger import get_logger


def main() -> None:
    """加载 settings.yaml，校验通过后打印就绪信息；配置错误则 fail-fast 退出。"""
    logger = get_logger("main")
    try:
        settings = load_settings()
    except SettingsError as exc:
        logger.error("配置加载失败: %s", exc)
        sys.exit(1)

    logger.info(
        "配置加载成功 — LLM=%s, Embedding=%s, VectorStore=%s",
        settings.llm.provider,
        settings.embedding.provider,
        settings.vector_store.provider,
    )
    print("Modular RAG MCP Server — ready")


if __name__ == "__main__":
    main()
