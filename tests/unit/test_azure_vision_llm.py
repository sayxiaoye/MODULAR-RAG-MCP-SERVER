"""Azure Vision LLM 单元测试（mock HTTP，不走真实 API）。"""

from __future__ import annotations

from io import BytesIO
from unittest.mock import MagicMock, patch

import httpx
import pytest
from PIL import Image

from core.settings import Settings, VisionLLMSettings, load_settings
from libs.llm.base_vision_llm import VisionLLMError
from libs.llm.azure_vision_llm import AzureVisionLLM, _resize_image_bytes
from libs.llm.llm_factory import LLMFactory


def _vision_settings(**overrides: object) -> VisionLLMSettings:
    base = {
        "enabled": True,
        "provider": "azure",
        "model": "gpt-4o",
        "max_image_size": 256,
        "api_key": "azure-vision-key",
        "api_version": "2024-02-15-preview",
        "azure_endpoint": "https://my.openai.azure.com",
        "deployment_name": "gpt-4o-vision",
    }
    base.update(overrides)
    return VisionLLMSettings(**base)


def _make_large_png_bytes(width: int = 800, height: int = 600) -> bytes:
    """生成测试用大尺寸 PNG 字节。"""
    buffer = BytesIO()
    Image.new("RGB", (width, height), color=(10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.mark.unit
class TestAzureVisionLLMImageProcessing:
    """验证图片压缩与输入方式。"""

    def test_resize_large_image(self) -> None:
        """超长边应被压缩到 max_image_size 以内。"""
        resized, mime = _resize_image_bytes(_make_large_png_bytes(800, 600), max_size=256)
        with Image.open(BytesIO(resized)) as image:
            assert max(image.size) <= 256
        assert mime == "image/png"

    @patch("httpx.post")
    def test_chat_with_image_bytes(self, mock_post: MagicMock) -> None:
        """bytes 输入应成功构造 base64 image_url 并解析响应。"""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = ""
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "图中是一只猫"}}],
            "model": "gpt-4o",
        }
        mock_post.return_value = mock_resp

        vision = AzureVisionLLM(_vision_settings())
        response = vision.chat_with_image("描述图片", _make_large_png_bytes(400, 300))
        assert response.content == "图中是一只猫"

        payload = mock_post.call_args.kwargs["json"]
        image_part = payload["messages"][0]["content"][1]
        assert image_part["type"] == "image_url"
        assert image_part["image_url"]["url"].startswith("data:image/png;base64,")
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["api-key"] == "azure-vision-key"
        url = mock_post.call_args.args[0]
        assert "deployments/gpt-4o-vision/chat/completions" in url

    @patch("httpx.post")
    def test_chat_with_image_path(self, mock_post: MagicMock, tmp_path) -> None:
        """文件路径输入应可读入并发送请求。"""
        image_path = tmp_path / "sample.png"
        image_path.write_bytes(_make_large_png_bytes(120, 80))

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = ""
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "path caption"}}],
            "model": "gpt-4o",
        }
        mock_post.return_value = mock_resp

        vision = AzureVisionLLM(_vision_settings())
        response = vision.chat_with_image("describe", str(image_path))
        assert response.content == "path caption"


@pytest.mark.unit
class TestAzureVisionLLMErrors:
    """验证 Azure Vision 错误处理。"""

    @patch("httpx.post")
    def test_auth_failure_contains_status_code(self, mock_post: MagicMock) -> None:
        """401 应包含 provider 与 HTTP 状态码。"""
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_resp.text = "Unauthorized"
        mock_post.return_value = mock_resp

        vision = AzureVisionLLM(_vision_settings())
        sample = _make_large_png_bytes(64, 64)
        with pytest.raises(VisionLLMError, match=r"\[azure-vision\].*HTTP 401"):
            vision.chat_with_image("q", sample)

    @patch("httpx.post", side_effect=httpx.TimeoutException("timeout"))
    def test_network_timeout_readable(self, mock_post: MagicMock) -> None:
        """超时应包含异常类型信息。"""
        vision = AzureVisionLLM(_vision_settings())
        sample = _make_large_png_bytes(64, 64)
        with pytest.raises(VisionLLMError, match=r"\[azure-vision\].*TimeoutException"):
            vision.chat_with_image("q", sample)


@pytest.mark.unit
class TestAzureVisionLLMFactoryRouting:
    """验证 LLMFactory 能创建 AzureVisionLLM。"""

    def test_factory_creates_azure_vision(self) -> None:
        """provider=azure 且 vision_llm 启用时应返回 AzureVisionLLM。"""
        base = load_settings()
        settings = Settings(
            llm=base.llm,
            embedding=base.embedding,
            vector_store=base.vector_store,
            retrieval=base.retrieval,
            rerank=base.rerank,
            evaluation=base.evaluation,
            observability=base.observability,
            ingestion=base.ingestion,
            vision_llm=_vision_settings(),
        )
        vision = LLMFactory.create_vision_llm(settings)
        assert isinstance(vision, AzureVisionLLM)
