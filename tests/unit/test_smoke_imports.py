"""pytest 冒烟测试：校验关键顶层包与子模块均可正常 import。"""

import importlib
import pytest
from pathlib import Path

# 架构分层对应的顶层包，与 A1 验收标准一致
TOP_LEVEL_PACKAGES = [
    "mcp_server",
    "core",
    "ingestion",
    "libs",
    "observability",
]

# 各层主要子模块，确保目录骨架完整且可被测试发现
SUB_PACKAGES = [
    "mcp_server.tools",
    "core.query_engine",
    "core.response",
    "core.trace",
    "ingestion.chunking",
    "ingestion.transform",
    "ingestion.embedding",
    "ingestion.storage",
    "libs.loader",
    "libs.llm",
    "libs.embedding",
    "libs.splitter",
    "libs.vector_store",
    "libs.reranker",
    "libs.evaluator",
    "observability.dashboard",
    "observability.dashboard.pages",
    "observability.dashboard.services",
    "observability.evaluation",
]


@pytest.mark.unit
class TestSmokeImports:
    """验证 src 布局下所有关键包在 editable install 后可被 import。"""

    @pytest.mark.parametrize("package", TOP_LEVEL_PACKAGES)
    def test_top_level_package_importable(self, package: str) -> None:
        """顶层包（mcp_server/core/ingestion/libs/observability）应能成功导入。"""
        module = importlib.import_module(package)
        assert module is not None

    @pytest.mark.parametrize("package", SUB_PACKAGES)
    def test_sub_package_importable(self, package: str) -> None:
        """各层子模块应存在且可导入，防止目录骨架缺失。"""
        module = importlib.import_module(package)
        assert module is not None


@pytest.mark.unit
class TestFixturesLayout:
    """验证测试夹具目录约定已就绪。"""

    def test_sample_document_fixture_exists(self) -> None:
        """fixtures/sample_documents 下应有最小样例文档供后续摄取测试使用。"""
        sample = Path("tests/fixtures/sample_documents/sample.txt")
        assert sample.is_file()
        assert sample.stat().st_size > 0

    def test_test_directory_structure(self) -> None:
        """unit/integration/e2e/fixtures 四层测试目录均应存在。"""
        for name in ("unit", "integration", "e2e", "fixtures"):
            assert Path("tests", name).is_dir()
