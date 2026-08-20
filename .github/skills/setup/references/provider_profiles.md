# Provider Profiles Reference

Quick reference for supported provider configurations.

## LLM Providers

### OpenAI
```yaml
llm:
  provider: "openai"
  model: "gpt-4o"           # or gpt-4.1, gpt-4o-mini, gpt-5 (newer but less universally available)
  api_key: "<OPENAI_API_KEY>"
  temperature: 0.0
  max_tokens: 4096
```
Required fields: `provider`, `model`, `api_key`
Remove/leave empty: `azure_endpoint`, `deployment_name`, `api_version`

> **Model selection**: `gpt-5` for best quality, `gpt-5-mini` for cost/performance, `gpt-4o` for maximum stability.

### Azure OpenAI
```yaml
llm:
  provider: "azure"
  model: "gpt-4o"          # or gpt-4o-mini, gpt-5 (must match your deployment)
  deployment_name: "<YOUR_DEPLOYMENT>"
  azure_endpoint: "https://<RESOURCE>.openai.azure.com/"
  api_version: "2025-06-01"
  api_key: "<AZURE_API_KEY>"
  temperature: 0.0
  max_tokens: 4096
```
Required fields: all shown above

### DeepSeek
```yaml
llm:
  provider: "deepseek"
  model: "deepseek-v4-flash"   # or deepseek-v4-pro (higher quality, slower)
  api_key: "<DEEPSEEK_API_KEY>"
  temperature: 0.0
  max_tokens: 4096
```

> ⚠️ **Migration notice** (2026-07-24): DeepSeek has retired `deepseek-chat` and `deepseek-reasoner`.
> Replace `deepseek-chat` → `deepseek-v4-flash`. For reasoning, use `deepseek-v4-flash`
> with `thinking: {type: enabled}` or switch to `deepseek-v4-pro`.
> Both V4 models support 1M context (8× the old V3.2 limit).

### LlamaCpp (local, recommended)

> **Status**: Provider code **pending** (tasks B7.9 / B7.10 in `DEV_SPEC.md`).
> Uses **llama-server** OpenAI-compatible API (`/v1/chat/completions`), not Ollama protocol.
> Implementation notes: [new_provider_guide.md](new_provider_guide.md) § Planned: LlamaCpp.

```yaml
llm:
  provider: "llamacpp"
  model: "qwen2.5-7b-instruct"     # must match GGUF loaded in llama-server / --alias
  base_url: "http://localhost:8080/v1"
  temperature: 0.0
  max_tokens: 4096
  # api_key optional — llama-server typically does not validate keys
```

Start chat server:

```bash
llama-server -m /path/to/chat-model.gguf --port 8080
```

Verify: `curl http://localhost:8080/v1/models`

`model` must match the id returned by `/v1/models` (GGUF filename or `--alias`).
No API key required once implemented (`api_key` may be omitted).

**vs Ollama**: Ollama uses native `/api/chat` on port 11434; LlamaCpp uses OpenAI-compat `/v1/*`. Prefer `llamacpp` for new local deployments; keep `ollama` as legacy.

### Ollama (local, legacy)

> **Note**: Retained for compatibility. **New local deployments should prefer `llamacpp`.**

```yaml
llm:
  provider: "ollama"
  model: "llama3.1"          # or llama4:scout, qwen3:8b, gemma4:12b
  base_url: "http://localhost:11434"
  temperature: 0.0
  max_tokens: 4096
```
No API key required. Pull model first: `ollama pull llama4:scout`

## Embedding Providers

### OpenAI
```yaml
embedding:
  provider: "openai"
  model: "text-embedding-3-small"     # or text-embedding-3-large (3072d), text-embedding-ada-002 (legacy)
  dimensions: 1536                     # small: 512-1536, large: 256-3072 (Matryoshka)
  api_key: "<OPENAI_API_KEY>"
```

### Azure OpenAI
```yaml
embedding:
  provider: "azure"
  model: "text-embedding-3-small"
  dimensions: 1536
  deployment_name: "<YOUR_EMBEDDING_DEPLOYMENT>"
  azure_endpoint: "https://<RESOURCE>.openai.azure.com/"
  api_version: "2025-06-01"
  api_key: "<AZURE_API_KEY>"
```

### LlamaCpp (local, recommended)

> **Status**: Provider code **pending** (B7.10). Chat and embedding should use **separate llama-server instances** (different ports / GGUF files).

```yaml
embedding:
  provider: "llamacpp"
  model: "bge-m3"
  dimensions: 1024
  base_url: "http://localhost:8081/v1"
```

Start embedding server:

```bash
llama-server -m /path/to/embed-model.gguf --port 8081 --embedding
```

Verify: `curl http://localhost:8081/v1/embeddings -H "Content-Type: application/json" -d '{"model":"<embed-model>","input":["hello"]}'`

> After changing embedding model or `dimensions`, **clear Chroma and re-run ingestion** — old vectors are not compatible.

### Ollama (legacy)
```yaml
embedding:
  provider: "ollama"
  model: "nomic-embed-text"
  dimensions: 768
  base_url: "http://localhost:11434"
```

## Unimplemented Providers (auto-scaffold on selection)

> These providers use OpenAI-compatible APIs. When selected, the setup skill
> auto-scaffolds the provider code by subclassing `OpenAILLM` / `OpenAIEmbedding`.

### Qwen (Alibaba Cloud DashScope)
```yaml
llm:
  provider: "qwen"
  model: "qwen3.7-flash"      # or qwen3.7-max (best reasoning), qwen3.6-plus (multimodal)
  api_key: "<DASHSCOPE_API_KEY>"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  temperature: 0.0
  max_tokens: 4096
```
Embedding:
```yaml
embedding:
  provider: "qwen"
  model: "text-embedding-v3"
  dimensions: 1024
  api_key: "<DASHSCOPE_API_KEY>"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
```
Vision model: `qwen3-vl-32b-thinking` (flagship), `qwen3.7-plus` (multimodal text+image+video)
SDK: `pip install openai` (uses OpenAI-compatible protocol)
International endpoint also available: `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`

### Gemini (Google AI Studio)
```yaml
llm:
  provider: "gemini"
  model: "gemini-3.6-flash"   # or gemini-3.5-flash, gemini-3.5-pro (heavier reasoning)
  api_key: "<GEMINI_API_KEY>"
  base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"
  temperature: 0.0
  max_tokens: 4096
```
Embedding:
```yaml
embedding:
  provider: "gemini"
  model: "text-embedding-004"
  dimensions: 768
  api_key: "<GEMINI_API_KEY>"
  base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"
```
Vision models: all Gemini 3.x models are natively multimodal. `gemini-3.6-flash` recommended.
SDK: `pip install openai` (uses OpenAI-compatible protocol)

### Groq
```yaml
llm:
  provider: "groq"
  model: "llama-3.3-70b"      # or mixtral-8x7b, gemma2-9b-it (see Groq docs for latest)
  api_key: "<GROQ_API_KEY>"
  base_url: "https://api.groq.com/openai/v1"
  temperature: 0.0
  max_tokens: 4096
```
No embedding endpoint. Vision: `llama-3.2-90b-vision-preview`.
SDK: `pip install openai` (uses OpenAI-compatible protocol)

### Mistral
```yaml
llm:
  provider: "mistral"
  model: "mistral-large-latest"  # or mistral-small-latest, codestral-latest
  api_key: "<MISTRAL_API_KEY>"
  base_url: "https://api.mistral.ai/v1"
  temperature: 0.0
  max_tokens: 4096
```
Embedding:
```yaml
embedding:
  provider: "mistral"
  model: "mistral-embed"
  dimensions: 1024
  api_key: "<MISTRAL_API_KEY>"
  base_url: "https://api.mistral.ai/v1"
```
SDK: `pip install openai` (uses OpenAI-compatible protocol)

### Together AI
```yaml
llm:
  provider: "together"
  model: "meta-llama/Llama-3.3-70B-Instruct-Turbo"  # see Together docs for full catalog
  api_key: "<TOGETHER_API_KEY>"
  base_url: "https://api.together.xyz/v1"
  temperature: 0.0
  max_tokens: 4096
```
Embedding:
```yaml
embedding:
  provider: "together"
  model: "togethercomputer/m2-bert-80M-8k-retrieval"
  dimensions: 768
  api_key: "<TOGETHER_API_KEY>"
  base_url: "https://api.together.xyz/v1"
```
SDK: `pip install openai` (uses OpenAI-compatible protocol)

## Model → Dimensions Lookup

| Model                           | Dimensions | Notes |
|---------------------------------|------------|-------|
| text-embedding-3-small          | 1536 (512–1536) | Matryoshka — variable dims supported |
| text-embedding-3-large          | 3072 (256–3072) | Matryoshka — 256d still beats ada-002 |
| text-embedding-ada-002 (legacy) | 1536       | Not recommended for new projects |
| bge-m3 (LlamaCpp) | 1024     | via llama-server `--embedding`; confirm with `/v1/embeddings` |
| nomic-embed-text (Ollama, legacy) | 768      | |
| mxbai-embed-large (Ollama, legacy) | 1024    | |
| text-embedding-v3 (Qwen)        | 1024       | |
| text-embedding-004 (Gemini)     | 768        | |
| mistral-embed (Mistral)         | 1024       | |

## Vision LLM Providers

Vision uses a **separate config section** (`vision_llm`) with its own `provider`, `api_key`, etc.

### Vision Model Reference Table

| Provider | Model | Quality | Speed | Cost | Notes |
|----------|-------|---------|-------|------|-------|
| OpenAI | `gpt-5` | ⭐⭐⭐⭐⭐ | Fast | $$ | 推荐，最新旗舰多模态 |
| OpenAI | `gpt-5-mini` | ⭐⭐⭐⭐ | Very Fast | $ | 性价比高，适合简单图片 |
| OpenAI | `gpt-4o` | ⭐⭐⭐⭐⭐ | Fast | $$ | 稳定可靠，综合优秀 |
| Azure | `gpt-5` | ⭐⭐⭐⭐⭐ | Fast | $$ | 同 OpenAI，需 Azure 部署 |
| Azure | `gpt-4o` | ⭐⭐⭐⭐⭐ | Fast | $$ | 稳定版本 |
| Ollama | `llama4:scout` | ⭐⭐⭐⭐⭐ | Medium | Free | 原生多模态 MoE，10M 上下文 |
| Ollama | `qwen3-vl:8b` | ⭐⭐⭐⭐ | Fast | Free | 最佳开源视觉，擅截图/UI |
| Ollama | `gemma4:12b` | ⭐⭐⭐⭐ | Fast | Free | Google 开源多模态 |
| Ollama | `llava:13b` | ⭐⭐⭐⭐ | Slow | Free | 经典选择，需更多显存 |
| Ollama | `moondream` | ⭐⭐ | Very Fast | Free | 最轻量（1.6B），资源占用极少 |
| Qwen | `qwen3-vl-32b-thinking` | ⭐⭐⭐⭐⭐ | Medium | $$ | 通义千问视觉推理旗舰 |
| Qwen | `qwen3.7-plus` | ⭐⭐⭐⭐ | Fast | $ | 多模态（text+image+video），性价比高 |
| Gemini | `gemini-3.6-flash` | ⭐⭐⭐⭐⭐ | Very Fast | $ | 推荐，最新最高效 |
| Gemini | `gemini-3.5-flash` | ⭐⭐⭐⭐⭐ | Fast | $ | 均衡之选 |
| Gemini | `gemini-3.5-pro` | ⭐⭐⭐⭐⭐ | Slow | $$$ | 最高质量，适合复杂图片 |
| Gemini | `gemini-3.5-flash-lite` | ⭐⭐⭐ | Very Fast | ¢ | 极低成本 |
| DeepSeek | — | — | — | — | ❌ 不支持 Vision，需选其他 provider |

### OpenAI Vision
```yaml
vision_llm:
  enabled: true
  provider: "openai"
  model: "gpt-5"  # or: gpt-5-mini, gpt-4o
  api_key: "<OPENAI_API_KEY>"
  max_image_size: 2048
```

### Azure Vision
```yaml
vision_llm:
  enabled: true
  provider: "azure"
  model: "gpt-5"  # or: gpt-4o (must match your deployment)
  deployment_name: "<YOUR_VISION_DEPLOYMENT>"
  azure_endpoint: "https://<RESOURCE>.openai.azure.com/"
  api_version: "2025-06-01"
  api_key: "<AZURE_API_KEY>"
  max_image_size: 2048
```

### Ollama Vision
```yaml
vision_llm:
  enabled: true
  provider: "ollama"
  model: "llama4:scout"  # or: qwen3-vl:8b, gemma4:12b, llava:13b
  base_url: "http://localhost:11434"
  max_image_size: 2048
```

### Qwen Vision
```yaml
vision_llm:
  enabled: true
  provider: "qwen"
  model: "qwen3.7-plus"  # or: qwen3-vl-32b-thinking, qwen3.6-plus
  api_key: "<DASHSCOPE_API_KEY>"
  base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1"
  max_image_size: 2048
```

### Gemini Vision
```yaml
vision_llm:
  enabled: true
  provider: "gemini"
  model: "gemini-3.6-flash"  # or: gemini-3.5-flash, gemini-3.5-pro
  api_key: "<GEMINI_API_KEY>"
  base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"
  max_image_size: 2048
```

### Vision Disabled
```yaml
vision_llm:
  enabled: false
  provider: "openai"
  model: "gpt-5"
  api_key: "<GEMINI_API_KEY>"
  base_url: "https://generativelanguage.googleapis.com/v1beta/openai/"
  max_image_size: 2048
```

### Vision Disabled
```yaml
vision_llm:
  enabled: false
  provider: "openai"
  model: "gpt-4o"
  max_image_size: 2048
```

## Rerank Providers

### None (disabled)
```yaml
rerank:
  enabled: false
  provider: "none"
  model: ""
  top_k: 5
```

### Cross-Encoder

> ⚠️ **Note**: Cross-Encoder 仅完成了本地代码实现，尚未经过充分测试，可能存在兼容性问题。建议优先选择「No (disabled)」或「LLM-based」。

```yaml
rerank:
  enabled: true
  provider: "cross_encoder"
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
  top_k: 5
```

### LLM-based
```yaml
rerank:
  enabled: true
  provider: "llm"
  model: ""  # uses the configured LLM
  top_k: 5
```
