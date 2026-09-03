"""README 章节完整性：I3 验收要求的标题与 MCP 配置文件名必须出现。"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"


@pytest.mark.unit
class TestReadmeRequiredSections:
    """新用户 10 分钟上手所需章节应写在 README 里。"""

    def test_required_headings_present(self) -> None:
        """验收：快速开始、配置、MCP、Dashboard、测试、FAQ 标题齐全。"""
        text = README.read_text(encoding="utf-8")
        required = (
            "## 快速开始",
            "## 配置说明",
            "## MCP 配置示例",
            "## Dashboard 使用指南",
            "## 运行测试",
            "## 常见问题",
        )
        missing = [title for title in required if title not in text]
        assert not missing, f"README 缺少章节: {missing}"

    def test_mentions_mcp_client_config_files(self) -> None:
        """验收：同时给出 Copilot mcp.json 与 Claude Desktop 配置文件名。"""
        text = README.read_text(encoding="utf-8")
        assert "mcp.json" in text
        assert "claude_desktop_config.json" in text
        assert "query_knowledge_hub" in text

    def test_mentions_install_ingest_and_settings(self) -> None:
        """快速开始应覆盖安装、配置与首次摄取。"""
        text = README.read_text(encoding="utf-8")
        assert "pip install" in text
        assert "config/settings.yaml" in text
        assert "scripts/ingest.py" in text
        assert "scripts/start_dashboard.py" in text
        assert "pytest" in text
