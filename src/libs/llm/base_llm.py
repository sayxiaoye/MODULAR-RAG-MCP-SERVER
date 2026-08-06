"""LLM 抽象层：定义统一对话接口与响应结构，供工厂与各 Provider 实现复用。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

# 合法的角色取值，与主流 Chat API 对齐
_VALID_ROLES = frozenset({"system", "user", "assistant"})


class LLMError(Exception):
    """LLM 调用或消息校验失败时抛出。"""


@dataclass(frozen=True)
class ChatMessage:
    """单条对话消息，role + content 与 OpenAI Chat 格式一致。"""

    role: str
    content: str

    def __post_init__(self) -> None:
        if self.role not in _VALID_ROLES:
            raise LLMError(f"无效的消息角色: {self.role!r}，允许: {sorted(_VALID_ROLES)}")
        if not isinstance(self.content, str) or not self.content.strip():
            raise LLMError("消息 content 必须是非空字符串")


@dataclass
class ChatResponse:
    """LLM 对话响应，content 为模型生成文本。"""

    content: str
    model: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)


def normalize_messages(messages: Sequence[ChatMessage | Mapping[str, Any]]) -> list[ChatMessage]:
    """将 dict 或 ChatMessage 序列规范化为 ChatMessage 列表，并校验 shape。"""
    if not messages:
        raise LLMError("messages 不能为空")

    normalized: list[ChatMessage] = []
    for index, item in enumerate(messages):
        if isinstance(item, ChatMessage):
            normalized.append(item)
            continue
        if not isinstance(item, Mapping):
            raise LLMError(f"messages[{index}] 必须是 ChatMessage 或 dict")
        if "role" not in item or "content" not in item:
            raise LLMError(f"messages[{index}] 必须包含 role 与 content 字段")
        normalized.append(ChatMessage(role=str(item["role"]), content=str(item["content"])))
    return normalized


class BaseLLM(ABC):
    """LLM 抽象基类：屏蔽各 Provider 的请求格式与认证差异。"""

    @abstractmethod
    def chat(
        self,
        messages: Sequence[ChatMessage | Mapping[str, Any]],
        trace: Any | None = None,
    ) -> ChatResponse:
        """
        发送对话请求并返回模型响应。

        Args:
            messages: 对话历史，元素为 ChatMessage 或含 role/content 的 dict。
            trace: 可选追踪上下文（F 阶段 TraceContext 注入，用于打点）。

        Returns:
            ChatResponse，其中 content 为模型生成文本。
        """
