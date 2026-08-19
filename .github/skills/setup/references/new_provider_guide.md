# New Provider Scaffolding Guide

When a user selects a provider that is not yet implemented, follow this procedure to auto-scaffold the code. The project uses a plugin architecture — adding a new provider is mechanical.

## Quick Check: Is the Provider Built-in?

Built-in LLM providers: `openai`, `azure`, `deepseek`, `ollama` (legacy), `llamacpp` 
Built-in Embedding providers: `openai`, `azure`, `ollama` (legacy), `llamacpp` 
Built-in Vision providers: `openai`, `azure`

**Local inference**: prefer **`llamacpp`** (llama.cpp `llama-server`, OpenAI-compatible `/v1/*`). Use `ollama` only for legacy setups. Config examples: [provider_profiles.md](provider_profiles.md).

If the provider is NOT in these lists, proceed with scaffolding below.

## Key Insight: OpenAI-Compatible Providers

Many modern providers (Qwen/DashScope, Gemini, Groq, Mistral, Together AI, etc.) expose an **OpenAI-compatible API**. For these, the implementation is trivial — subclass the existing `OpenAILLM` / `OpenAIEmbedding` and override the base URL + auth.

## Step 1: Create LLM Provider

Create `src/libs/llm/{name}_llm.py`:

```python
"""<ProviderName> LLM implementation (OpenAI-compatible)."""
from __future__ import annotations
import os
from typing import Any, Optional
from src.libs.llm.openai_llm import OpenAILLM


class <ProviderName>LLMError(RuntimeError):
    """Raised when <ProviderName> API call fails."""


class <ProviderName>LLM(OpenAILLM):
    """<ProviderName> LLM provider — OpenAI-compatible endpoint.

    Inherits all chat logic from OpenAILLM; overrides base URL and
    API key resolution.
    """

    DEFAULT_BASE_URL = "<PROVIDER_BASE_URL>"

    def __init__(self, settings: Any, api_key: Optional[str] = None,
                 base_url: Optional[str] = None, **kwargs: Any) -> None:
        # Allow base_url from settings.llm.base_url
        resolved_base = (
            base_url
            or getattr(settings.llm, 'base_url', None)
            or self.DEFAULT_BASE_URL
        )
        # Allow api_key from settings or env
        resolved_key = (
            api_key
            or getattr(settings.llm, 'api_key', None)
            or os.environ.get("<ENV_VAR_NAME>")
        )
        if not resolved_key:
            raise ValueError(
                "<ProviderName> API key not provided. Set in settings.yaml "
                "(llm.api_key) or <ENV_VAR_NAME> environment variable."
            )
        super().__init__(settings, api_key=resolved_key, base_url=resolved_base, **kwargs)
```

For providers that are NOT OpenAI-compatible, subclass `BaseLLM` directly and implement `chat()` — refer to `deepseek_llm.py` or `ollama_llm.py` as examples.

## Planned: LlamaCpp Provider (local llama.cpp)

Do **not** generic-scaffold as a cloud OpenAI clone. Follow this section + `DEV_SPEC.md` B7.9 / B7.10.

**Why a dedicated provider**: llama.cpp `llama-server` is OpenAI-compatible (`/v1/chat/completions`, `/v1/embeddings`). Ollama uses a different native API (`/api/chat`). Do not reuse `ollama` or set `provider: openai` against localhost.

**Files to add**

| File | Notes |
|------|--------|
| `src/libs/llm/llamacpp_llm.py` | Subclass `OpenAICompatibleLLM` (same pattern as `DeepSeekLLM`) |
| `src/libs/embedding/llamacpp_embedding.py` | Reuse `request_compatible_embeddings` |
| `tests/unit/test_llamacpp_llm.py` | mock HTTP: chat / connect fail / HTTP error |
| `tests/unit/test_llamacpp_embedding.py` | mock HTTP: embed / dim mismatch / errors |

**Files to update**

- `llm_factory.py` / `embedding_factory.py`: `register_*_provider("llamacpp", ...)`
- `config/settings.yaml`: default `provider: llamacpp` after implementation
- `tests/unit/test_config_loading.py`: assert default provider `llamacpp`
- `tests/integration/test_chunk_refiner_llm.py`: probe `/v1/models` instead of Ollama `/api/tags`

**LLM sketch**

```python
DEFAULT_LLAMACPP_BASE_URL = "http://localhost:8080/v1"

class LlamaCppLLM(OpenAICompatibleLLM):
    def __init__(self, settings: LLMSettings) -> None:
        super().__init__(
            settings=settings,
            provider_name="llamacpp",
            base_url=settings.base_url or DEFAULT_LLAMACPP_BASE_URL,
            api_key_env="LLAMACPP_API_KEY",
        )

    def _resolve_api_key(self) -> str:
        if self.settings.api_key:
            return self.settings.api_key
        return os.environ.get("LLAMACPP_API_KEY", "not-needed")
```

Optional: timeout `120.0` (local inference is slower); errors like `[llamacpp] 请确认 llama-server 已启动：llama-server -m <model.gguf> --port 8080`.

**Embedding notes**

- Default `base_url`: `http://localhost:8081/v1` (separate process from chat)
- Same optional API key placeholder
- If llama-server returns 400 on `dimensions`, omit that field in `LlamaCppEmbedding.embed()`

**Out of scope**: LlamaCpp Vision; deleting Ollama; in-process `llama-cpp-python` (HTTP `llama-server` only).

## Step 2: Create Embedding Provider (if needed)

Create `src/libs/embedding/{name}_embedding.py`:

```python
"""<ProviderName> Embedding implementation (OpenAI-compatible)."""
from __future__ import annotations
import os
from typing import Any, Optional
from src.libs.embedding.openai_embedding import OpenAIEmbedding


class <ProviderName>EmbeddingError(RuntimeError):
    """Raised when <ProviderName> Embedding API call fails."""


class <ProviderName>Embedding(OpenAIEmbedding):
    """<ProviderName> Embedding provider — OpenAI-compatible endpoint."""

    DEFAULT_BASE_URL = "<PROVIDER_BASE_URL>"

    def __init__(self, settings: Any, api_key: Optional[str] = None,
                 base_url: Optional[str] = None, **kwargs: Any) -> None:
        resolved_base = (
            base_url
            or getattr(settings.embedding, 'base_url', None)
            or self.DEFAULT_BASE_URL
        )
        resolved_key = (
            api_key
            or getattr(settings.embedding, 'api_key', None)
            or os.environ.get("<ENV_VAR_NAME>")
        )
        if not resolved_key:
            raise ValueError(
                "<ProviderName> API key not provided. Set in settings.yaml "
                "(embedding.api_key) or <ENV_VAR_NAME> environment variable."
            )
        super().__init__(settings, api_key=resolved_key, base_url=resolved_base, **kwargs)
```

## Step 3: Create Vision LLM Provider (if needed)

Create `src/libs/llm/{name}_vision_llm.py` — subclass `OpenAIVisionLLM` with same pattern as Step 1.

## Step 4: Register Providers

### LLM — append to `src/libs/llm/__init__.py`:

```python
from src.libs.llm.{name}_llm import <ProviderName>LLM, <ProviderName>LLMError
LLMFactory.register_provider("{name}", <ProviderName>LLM)
```

### Embedding — append to `src/libs/embedding/embedding_factory.py` inside `_register_builtin_providers()`:

```python
try:
    from src.libs.embedding.{name}_embedding import <ProviderName>Embedding
    EmbeddingFactory.register_provider("{name}", <ProviderName>Embedding)
except ImportError:
    pass
```

Also update `src/libs/embedding/__init__.py` imports and `__all__`.

### Vision — append to `src/libs/llm/llm_factory.py` inside `_register_vision_providers()`:

```python
try:
    from src.libs.llm.{name}_vision_llm import <ProviderName>VisionLLM
    LLMFactory.register_vision_provider("{name}", <ProviderName>VisionLLM)
except ImportError:
    pass
```

## Step 5: Verify `base_url` Field Exists in Settings

Check `src/core/settings.py` — `LLMSettings` and `EmbeddingSettings` already have `base_url: Optional[str] = None`. No change needed.

## Step 6: Install SDK (if any)

Most OpenAI-compatible providers only need `python -m pip install openai` (already installed).
Provider-specific SDKs if NOT OpenAI-compatible:
- Qwen: `python -m pip install dashscope` (alternative, but OpenAI-compat mode recommended)
- Gemini: `python -m pip install google-generativeai` (only if NOT using OpenAI-compat mode)

## Provider-Specific Reference

| Provider | Base URL (LLM) | Base URL (Embedding) | API Key Env Var | OpenAI-compat |
|----------|----------------|----------------------|-----------------|---------------|
| Qwen     | `https://dashscope.aliyuncs.com/compatible-mode/v1` | same | `DASHSCOPE_API_KEY` | Yes |
| Gemini   | `https://generativelanguage.googleapis.com/v1beta/openai/` | same | `GEMINI_API_KEY` | Yes |
| Groq     | `https://api.groq.com/openai/v1` | N/A (no embedding) | `GROQ_API_KEY` | Yes |
| Mistral  | `https://api.mistral.ai/v1` | same | `MISTRAL_API_KEY` | Yes |
| Together | `https://api.together.xyz/v1` | same | `TOGETHER_API_KEY` | Yes |
| LlamaCpp (local) | `http://localhost:8080/v1` | `http://localhost:8081/v1` | optional (`not-needed`) | Yes (pending) |

## Validation

After scaffolding, run quick validation:

```python
python -c "
from src.libs.llm import LLMFactory
from src.libs.embedding import EmbeddingFactory
print('LLM providers:', LLMFactory.list_providers())
print('Embedding providers:', EmbeddingFactory.list_providers())
print('Vision providers:', LLMFactory.list_vision_providers())
"
```

The new provider should appear in the list. Then proceed with setup Step 3 (Generate Config).
