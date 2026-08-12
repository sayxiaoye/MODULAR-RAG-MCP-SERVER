"""Azure OpenAI Vision LLM 实现（GPT-4o / GPT-4-Vision）。"""

from __future__ import annotations

import base64
import os
from io import BytesIO
from typing import Any

import httpx
from PIL import Image

from core.settings import VisionLLMSettings
from libs.llm.base_llm import ChatResponse
from libs.llm.base_vision_llm import BaseVisionLLM, VisionLLMError

_MIME_BY_FORMAT = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "GIF": "image/gif",
    "WEBP": "image/webp",
}


def _resize_image_bytes(image_bytes: bytes, max_size: int) -> tuple[bytes, str]:
    """将图片最长边缩放至 max_size 以内，返回编码后的字节与 MIME 类型。"""
    if max_size <= 0:
        raise VisionLLMError("max_image_size 必须大于 0")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            original_format = (image.format or "JPEG").upper()
            if original_format not in _MIME_BY_FORMAT:
                original_format = "JPEG"
            width, height = image.size
            longest = max(width, height)
            if longest > max_size:
                scale = max_size / float(longest)
                resized = image.resize(
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    Image.Resampling.LANCZOS,
                )
            else:
                resized = image.copy()

            if original_format == "JPEG":
                resized = resized.convert("RGB")

            buffer = BytesIO()
            save_kwargs: dict[str, Any] = {"format": original_format}
            if original_format == "JPEG":
                save_kwargs["quality"] = 85
            resized.save(buffer, **save_kwargs)
            mime = _MIME_BY_FORMAT[original_format]
            return buffer.getvalue(), mime
    except Exception as exc:
        raise VisionLLMError(f"无法解析或处理图片数据: {exc}") from exc


class AzureVisionLLM(BaseVisionLLM):
    """通过 Azure OpenAI 部署调用多模态 Chat 模型理解图像。"""

    def __init__(self, settings: VisionLLMSettings) -> None:
        super().__init__(settings)

    def _resolve_api_key(self) -> str:
        if self.settings.api_key:
            return self.settings.api_key
        env_key = os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not env_key:
            raise VisionLLMError(
                "[azure-vision] 缺少 API Key：请配置 vision_llm.api_key 或环境变量 AZURE_OPENAI_API_KEY"
            )
        return env_key

    def _chat_url(self) -> str:
        endpoint = (self.settings.azure_endpoint or "").rstrip("/")
        deployment = self.settings.deployment_name or self.settings.model
        api_version = self.settings.api_version or "2024-02-15-preview"
        if not endpoint:
            raise VisionLLMError("[azure-vision] 缺少必填配置 vision_llm.azure_endpoint")
        if not deployment:
            raise VisionLLMError(
                "[azure-vision] 缺少必填配置 vision_llm.deployment_name 或 vision_llm.model"
            )
        return (
            f"{endpoint}/openai/deployments/{deployment}/chat/completions"
            f"?api-version={api_version}"
        )

    def _preprocess_image(self, image_bytes: bytes) -> bytes:
        """按 max_image_size 压缩图片，避免超出 Azure Vision 上下文限制。"""
        processed, _ = _resize_image_bytes(image_bytes, self.settings.max_image_size)
        return processed

    def _image_payload(self, image_bytes: bytes) -> dict[str, Any]:
        """构造 OpenAI 兼容的多模态 image_url 字段。"""
        processed_bytes, mime = _resize_image_bytes(image_bytes, self.settings.max_image_size)
        encoded = base64.b64encode(processed_bytes).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{encoded}"},
        }

    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: Any | None = None,
    ) -> ChatResponse:
        validated_text = self._validate_text(text)
        image_bytes = self.prepare_image_bytes(image_path)
        url = self._chat_url()
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": validated_text},
                        self._image_payload(image_bytes),
                    ],
                }
            ],
        }
        headers = {
            "api-key": self._resolve_api_key(),
            "Content-Type": "application/json",
        }

        try:
            response = httpx.post(url, json=payload, headers=headers, timeout=120.0)
        except httpx.HTTPError as exc:
            raise VisionLLMError(
                f"[azure-vision] 网络请求失败 ({type(exc).__name__}): {exc}"
            ) from exc

        if response.status_code >= 400:
            raise VisionLLMError(
                f"[azure-vision] API 错误 HTTP {response.status_code}: {response.text[:300]}"
            )

        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise VisionLLMError(
                "[azure-vision] 响应格式异常，无法解析 choices.message.content"
            ) from exc

        return ChatResponse(
            content=str(content),
            model=str(data.get("model", self.settings.deployment_name or self.settings.model)),
            usage=dict(data.get("usage") or {}),
        )
