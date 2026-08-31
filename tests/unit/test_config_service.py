"""ConfigService 单元测试：验证组件卡片格式化。"""

from __future__ import annotations

import pytest

from core.settings import load_settings
from observability.dashboard.services.config_service import ConfigService


@pytest.mark.unit
class TestConfigService:
    """验证 Settings 被格式化为总览页所需的组件卡片。"""

    def test_component_cards_cover_five_plugins(self) -> None:
        """应输出 LLM/Embedding/Splitter/Reranker/Evaluator 五张卡片。"""
        service = ConfigService(load_settings())
        cards = {card.title: card for card in service.component_cards()}

        assert set(cards) == {"LLM", "Embedding", "Splitter", "Reranker", "Evaluator"}
        assert cards["LLM"].provider
        assert "/" in cards["LLM"].summary
        assert "d" in cards["Embedding"].summary
        assert "size=" in cards["Splitter"].summary
        assert "enabled" in cards["Reranker"].extras
        assert isinstance(cards["Evaluator"].extras["metrics"], list)

    def test_disabled_rerank_uses_none_provider(self) -> None:
        """Rerank 未启用时 provider 应展示 none。"""
        settings = load_settings()
        assert settings.rerank.enabled is False
        service = ConfigService(settings)
        rerank = next(card for card in service.component_cards() if card.title == "Reranker")
        assert rerank.provider == "none"
        assert rerank.extras["enabled"] is False
