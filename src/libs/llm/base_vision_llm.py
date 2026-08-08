"""Vision LLM 抽象层：定义文本+图像多模态接口，供 ImageCaptioner 等模块复用。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from core.settings import VisionLLMSettings
from libs.llm.base_llm import ChatResponse, LLMError


class VisionLLMError(LLMError):
    """Vision LLM 输入校验或图像处理失败时抛出。"""


class BaseVisionLLM(ABC):
    """Vision LLM 抽象基类：屏蔽各 Provider 的多模态请求格式差异。"""

    def __init__(self, settings: VisionLLMSettings) -> None:
        self.settings = settings

    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: Any | None = None,
    ) -> ChatResponse:
        """
        发送文本+图像的多模态请求并返回模型响应。

        Args:
            text: 用户提示词或上下文描述。
            image_path: 图片本地路径，或已编码的原始图像字节。
            trace: 可选追踪上下文（F 阶段 TraceContext 注入）。

        Returns:
            ChatResponse，content 通常为图像描述或视觉理解结果。
        """

    def _validate_text(self, text: str) -> str:
        """校验提示文本非空。"""
        if not isinstance(text, str) or not text.strip():
            raise VisionLLMError("text 必须是非空字符串")
        return text.strip()

    def prepare_image_bytes(self, image: str | bytes) -> bytes:
        """
        将路径或 bytes 规范化为图像字节，供 Provider 构造多模态 payload。

        子类可覆盖 ``_preprocess_image`` 实现压缩、格式转换等预处理。
        """
        if isinstance(image, bytes):
            raw_bytes = image
        elif isinstance(image, str):
            path = Path(image)
            if not path.is_file():
                raise VisionLLMError(f"图片路径不存在或不可读: {image}")
            raw_bytes = path.read_bytes()
        else:
            raise VisionLLMError("image_path 必须是文件路径字符串或 bytes")

        if not raw_bytes:
            raise VisionLLMError("图片内容不能为空")
        return self._preprocess_image(raw_bytes)

    def _preprocess_image(self, image_bytes: bytes) -> bytes:
        """
        图片预处理扩展点：默认原样返回，子类可实现缩放/压缩。

        B9 等具体 Provider 可在此根据 ``settings.max_image_size`` 限制尺寸。
        """
        return image_bytes
