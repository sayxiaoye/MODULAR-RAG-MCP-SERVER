"""把项目 BaseLLM / BaseEmbedding 接到 Ragas Judge（走工厂，不直连 OpenAI）。"""

from __future__ import annotations

from typing import Any

from libs.evaluator.base_evaluator import EvaluatorError

_ROLE_MAP = {
    "human": "user",
    "ai": "assistant",
    "system": "system",
    "chat": "user",
}


def wrap_project_llm_for_ragas(llm: Any) -> Any:
    """
    把 ``BaseLLM.chat()`` 包成 Ragas 可调用的 Judge。

    必须经过项目 LLM（例如 LlamaCppLLM 的按需启停），不能直接打 OpenAI。
    若已是 Ragas wrapper，原样返回。
    """
    if llm is None:
        raise EvaluatorError(
            "Ragas Judge 未注入 LLM。请配置 settings.llm（llamacpp / openai 等），"
            "不要依赖 ragas 默认的 OPENAI_API_KEY。"
        )
    if hasattr(llm, "generate_text") and hasattr(llm, "agenerate_text"):
        return llm
    try:
        from ragas.llms import LangchainLLMWrapper
    except ImportError as exc:
        raise ImportError(
            "未安装 Ragas。请执行: python -m pip install '.[evaluation]'"
        ) from exc

    _enable_structured_json_output(llm)
    if hasattr(llm, "chat") and not hasattr(llm, "generate_prompt"):
        return LangchainLLMWrapper(_build_chat_model(llm))
    return LangchainLLMWrapper(llm)


def wrap_project_embeddings_for_ragas(embeddings: Any) -> Any:
    """
    把 ``BaseEmbedding.embed()`` 包成 Ragas Embedding。

    Answer Relevancy 需要向量；不注入时 ragas 会回退 OpenAI embedding。
    """
    if embeddings is None:
        raise EvaluatorError(
            "Ragas 未注入 Embedding（answer_relevancy 需要向量）。请配置 settings.embedding。"
        )
    name = type(embeddings).__name__
    if name in {"LangchainEmbeddingsWrapper", "HuggingfaceEmbeddings", "LlamaIndexEmbeddingsWrapper"}:
        return embeddings
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper
    except ImportError as exc:
        raise ImportError(
            "未安装 Ragas。请执行: python -m pip install '.[evaluation]'"
        ) from exc

    inner = embeddings
    if hasattr(embeddings, "embed") and not hasattr(embeddings, "embed_documents"):
        inner = _build_embeddings(embeddings)
    return LangchainEmbeddingsWrapper(inner)


def _build_chat_model(client: Any) -> Any:
    """构造 LangChain BaseChatModel，把 generate 转到 BaseLLM.chat()。"""
    try:
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage, BaseMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
    except ImportError as exc:
        raise ImportError(
            "Ragas Judge 需要 langchain-core。请执行: python -m pip install '.[evaluation]'"
        ) from exc

    class ProjectChatModel(BaseChatModel):
        """LangChain Chat 外壳：调用转到项目 BaseLLM.chat()。"""

        client: Any
        temperature: float = 0.0

        @property
        def _llm_type(self) -> str:
            return "modular_rag_judge"

        @property
        def _identifying_params(self) -> dict[str, Any]:
            return {"backend": "project_llm"}

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> ChatResult:
            converted = _langchain_messages_to_chat(messages)
            response = self.client.chat(converted)
            text = (getattr(response, "content", None) or str(response) or "").strip() or " "
            from observability.evaluation.cjk_text import sanitize_judge_output

            text = sanitize_judge_output(text).strip() or " "
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    return ProjectChatModel(client=client)


def _build_embeddings(inner: Any) -> Any:
    """构造 LangChain Embeddings，把向量化转到 BaseEmbedding.embed()。"""
    try:
        from langchain_core.embeddings import Embeddings
    except ImportError as exc:
        raise ImportError(
            "Ragas Embedding 需要 langchain-core。请执行: python -m pip install '.[evaluation]'"
        ) from exc

    class ProjectEmbeddings(Embeddings):
        """LangChain Embeddings 外壳：调用转到项目 BaseEmbedding.embed()。"""

        def __init__(self, backend: Any) -> None:
            super().__init__()
            self._inner = backend

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            if not texts:
                return []
            sanitized = [item if isinstance(item, str) and item.strip() else " " for item in texts]
            return self._inner.embed(sanitized)

        def embed_query(self, text: str) -> list[float]:
            cleaned = text if isinstance(text, str) and text.strip() else " "
            return self._inner.embed([cleaned])[0]

    return ProjectEmbeddings(inner)


def _enable_structured_json_output(llm: Any) -> None:
    """本地 llama.cpp 用 GBNF 锁 JSON；其它 OpenAI 兼容端用 response_format。"""
    extra: dict[str, Any]
    provider = str(getattr(llm, "provider_name", "") or "").lower()
    type_name = type(llm).__name__.lower()
    if provider == "llamacpp" or type_name == "llamacppllm":
        from observability.evaluation.json_grammar import JSON_OBJECT_GBNF

        extra = {"grammar": JSON_OBJECT_GBNF}
    else:
        extra = {"response_format": {"type": "json_object"}}
    try:
        llm.extra_chat_payload = extra
    except Exception:
        return


def _langchain_messages_to_chat(messages: list[Any]) -> list[dict[str, str]]:
    """把 LangChain BaseMessage 转成项目 chat() 所需的 role/content。"""
    converted: list[dict[str, str]] = []
    for msg in messages:
        role = _ROLE_MAP.get(getattr(msg, "type", "human"), "user")
        raw = getattr(msg, "content", "")
        content = raw if isinstance(raw, str) else str(raw)
        if not content.strip():
            content = " "
        converted.append({"role": role, "content": content})
    if not converted:
        converted.append({"role": "user", "content": " "})
    return converted
