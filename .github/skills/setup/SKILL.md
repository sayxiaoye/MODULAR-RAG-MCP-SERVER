---
name: setup
description: "Interactive project setup wizard. From a clean codebase, guides user through provider selection (OpenAI/Azure/DeepSeek/LlamaCpp/Ollama/Qwen/Gemini/etc.), API key configuration, dependency installation, config generation, and launches the dashboard. If user selects an unimplemented provider, auto-scaffolds the provider code following the plugin architecture. Auto-diagnoses and fixes startup failures with up to 3 retry rounds. Use when user says 'setup', 'set up', 'configure', 'init project', '初始化', '环境配置', '项目配置', 'first run', 'get started', 'quick start', or wants to configure and launch the project from scratch."
---

# Setup

Interactive wizard: configure providers → install deps → generate config → launch dashboard → auto-fix issues.

---

## Pipeline

```
Preflight → Ask User → Generate Config → Install Deps → Validate → Launch → Usage Guide
```

> Auto-fix loop: if any step fails, diagnose → fix → retry (≤3 rounds).

### Operating Defaults

- Always use the active Python interpreter consistently via `python -m pip` and `python -c`.
- Prefer environment variables for secrets whenever possible; only write credentials into `config/settings.yaml` if the user explicitly allows it.
- Make repeated runs idempotent: back up existing config before overwriting and reuse prior values when available.
- Handle Windows/macOS/Linux activation separately and verify the interpreter path before installing packages.

---

## Step 1: Preflight Checks

Verify prerequisites before asking the user anything:

### 1.1 Check Python Version

```powershell
python --version          # Require >=3.10
```

If Python < 3.10, stop and inform user to install a supported version.

### 1.2 Check & Create Virtual Environment

Check if `.venv` directory already exists. If it does, skip creation and just activate it. If it does NOT exist, create it and activate it.

**Important:** Use `--without-pip` to avoid the slow `ensurepip` step that can hang on Windows, then bootstrap pip manually with `ensurepip` after activation.

```powershell
# Step 1: Check if .venv exists
Test-Path ".venv"

# Step 2: If .venv does NOT exist, create it (fast, no pip bundling)
python -m venv .venv --without-pip

# Step 3: Activate the virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Windows (cmd):
# .\.venv\Scripts\activate.bat
# macOS/Linux:
# source .venv/bin/activate

# Step 4: Bootstrap pip inside the venv (only needed after --without-pip)
python -m ensurepip --upgrade

# Step 5: Verify
python -m pip --version  # Should show pip path inside .venv
python -c "import sys; print(sys.executable)"  # Should point inside .venv
```

If `.venv` already exists, only run Step 3 (activate) and Step 5 (verify).

---

## Step 2: Ask User for Configuration

Use the `ask_questions` tool to gather provider choices. Ask in batches (max 4 questions per call).

### Batch 1: Core Providers

Ask these questions together:

1. **LLM Provider** — Which LLM provider?
   - Options: `LlamaCpp (local, recommended)`, `OpenAI`, `Azure OpenAI`, `DeepSeek`, `Ollama (local, legacy)`, `Qwen (Alibaba Cloud)`, `Gemini (Google)`, `Mistral`, `Groq`, `Together AI`
   - Recommended (local/offline): `LlamaCpp` — config in [references/provider_profiles.md](references/provider_profiles.md); implement via [references/new_provider_guide.md](references/new_provider_guide.md) (B7.9)
   - Recommended (cloud): `OpenAI`
   - Built-in: OpenAI, Azure, DeepSeek, Ollama (legacy). **LlamaCpp**: spec ready, code pending (B7.9). Others require auto-scaffolding (see Step 2.5).

2. **Embedding Provider** — Which embedding provider?
   - Options: `LlamaCpp (local)`, `OpenAI`, `Azure OpenAI`, `Ollama (local, legacy)`, `Qwen (Alibaba Cloud)`, `Gemini (Google)`, `Mistral`, `Together AI`
   - Recommended: match LLM deployment style (LlamaCpp local or same cloud vendor as LLM)
   - Built-in: OpenAI, Azure, Ollama (legacy). **LlamaCpp embedding**: pending (B7.10). Others require auto-scaffolding (see Step 2.5).

3. **Vision** — Enable vision/image captioning?
   - Options: `Yes`, `No`
   - Recommended: `Yes`

4. **Rerank** — Enable reranking?
   - Options: `No (fastest)`, `Cross-Encoder (local model)`, `LLM-based`
   - Recommended: `No (fastest)`
   - ⚠️ Note: Cross-Encoder 仅完成了本地代码实现，尚未经过充分测试，可能存在兼容性问题。建议优先选择「不启用」或「LLM 重排序」。

### Batch 2: Credentials (based on Batch 1 answers)

Ask for credentials based on selected providers. Refer to [references/provider_profiles.md](references/provider_profiles.md) for required fields per provider.

For each secret, prefer asking whether to store it in an environment variable or in `config/settings.yaml`; recommend environment variables for API keys and keep YAML minimal.

**If OpenAI selected:**
- Ask: OpenAI API Key
- Ask: LLM model (default: `gpt-5`; `gpt-4o` for maximum stability)
- Ask: Embedding model (default: `text-embedding-3-small`)

**If Azure OpenAI selected:**
- Ask: Azure API Key
- Ask: Azure Endpoint URL
- Ask: LLM deployment name (default: `gpt-4o`)
- Ask: Embedding deployment name (default: `text-embedding-ada-002`)

**If DeepSeek selected:**
- Ask: DeepSeek API Key
- Ask: Embedding provider separately (DeepSeek has no embeddings — must use OpenAI/LlamaCpp/Ollama)

**If LlamaCpp selected:**
- Ask: Chat base URL (default: `http://localhost:8080/v1`)
- Ask: LLM model name (must match `llama-server` loaded GGUF / `--alias`)
- Ask: Embedding base URL (default: `http://localhost:8081/v1` — separate instance recommended)
- Ask: Embedding model name (default: `bge-m3`)
- Verify llama-server: `curl http://localhost:8080/v1/models` (chat) and embedding port if used
- Refer to [references/provider_profiles.md](references/provider_profiles.md) and [references/new_provider_guide.md](references/new_provider_guide.md)
- If provider code not yet implemented: inform user and offer to implement B7.9/B7.10

**If Ollama selected (legacy):**
- Ask: Ollama base URL (default: `http://localhost:11434`)
- Ask: LLM model name (default: `llama4:scout`)
- Ask: Embedding model name (default: `nomic-embed-text`)
- Verify Ollama is running: `curl http://localhost:11434/api/tags` or equivalent

**If Qwen selected:**
- Ask: Qwen API Key (DashScope)
- Ask: LLM model (default: `qwen3.7-flash`)
- Ask: Embedding model (default: `text-embedding-v3`) — if Qwen also chosen for embedding
- Base URL: `https://dashscope.aliyuncs.com/compatible-mode/v1` (OpenAI-compatible)

**If Gemini selected:**
- Ask: Gemini API Key (Google AI Studio)
- Ask: LLM model (default: `gemini-3.6-flash`)
- Ask: Embedding model (default: `text-embedding-004`) — if Gemini also chosen for embedding
- Base URL: `https://generativelanguage.googleapis.com/v1beta/openai/` (OpenAI-compatible)

### Batch 3: Vision Credentials (if vision enabled)

Vision LLM has its **own independent config section** (`vision_llm`) with separate `provider`, `api_key`, `azure_endpoint`, etc. Do NOT assume it shares credentials with the main LLM.

Ask up to 2 questions:

1. **Vision Provider** — Which provider for vision/image captioning?
   - Options: same as LLM provider list, but default to user's LLM choice
   - Built-in vision: OpenAI, Azure. Others require auto-scaffolding.

2. **Vision credentials** — based on provider:
   - If vision provider == LLM provider: ask "Reuse the same API key/endpoint for vision?" (default: Yes)
     - If Yes: copy LLM credentials to vision config
     - If No: ask for separate vision API key / endpoint
   - If vision provider != LLM provider: ask for vision-specific API key / endpoint / model

Vision models per provider (recommended model listed first):

| Provider | Recommended | Other Options | Notes |
|----------|------------|---------------|-------|
| OpenAI | `gpt-5` | `gpt-5-mini`, `gpt-4o` | gpt-5-mini 更便宜，gpt-4o 最稳定 |
| Azure | `gpt-5` | `gpt-4o` | 需要 deployment_name + azure_endpoint |
| LlamaCpp | `qwen2.5-7b-instruct` | `qwen2.5-14b-instruct`, etc. | llama-server + GGUF; OpenAI-compat `/v1/*` |
| Ollama (legacy) | `llama4:scout` | `qwen3-vl:8b`, `gemma4:12b`, `llava:13b` | llama4:scout 原生多模态 10M 上下文，qwen3-vl 最佳开源视觉 |
| Qwen | `qwen3.7-plus` | `qwen3-vl-32b-thinking` | qwen3.7-plus 多模态性价比高，vl-32b 视觉推理旗舰 |
| Gemini | `gemini-3.6-flash` | `gemini-3.5-flash`, `gemini-3.5-pro` | 3.6-flash 最新最高效，3.5-pro 质量最高但较慢 |
| DeepSeek | ❌ 无 Vision 模型 | — | 需选择其他 provider 作为 Vision LLM |

---

## Step 2.5: Scaffold Unimplemented Providers (if needed)

If the user selected a provider not yet built-in (e.g., Qwen, Gemini, or any custom provider), auto-scaffold the implementation before proceeding to config generation.

Refer to [references/new_provider_guide.md](references/new_provider_guide.md) for the complete scaffolding procedure.

Summary:
1. Create LLM class in `src/libs/llm/{name}_llm.py` (extend `BaseLLM`)
2. Create Embedding class in `src/libs/embedding/{name}_embedding.py` (extend `BaseEmbedding`) — if needed
3. Create Vision LLM class in `src/libs/llm/{name}_vision_llm.py` (extend `BaseVisionLLM`) — if needed
4. Register in `src/libs/llm/__init__.py` and `src/libs/embedding/__init__.py`
5. Add `base_url` field to `LLMSettings` / `EmbeddingSettings` if the provider uses a custom endpoint
6. Install provider SDK: `pip install <sdk>` if needed
7. Add provider profile to `references/provider_profiles.md`

Many providers (Qwen, Gemini, Groq, Mistral, etc.) are OpenAI-compatible — simply subclass `OpenAILLM` / `OpenAIEmbedding` and override `DEFAULT_BASE_URL` + auth logic.

---

## Step 3: Generate Config

Read the template from [references/settings_template.yaml](references/settings_template.yaml) and fill in values based on user answers.

Key rules:
- Look up `dimensions` from the model→dimensions table in [references/provider_profiles.md](references/provider_profiles.md)
- For LlamaCpp: set `base_url` (chat `8080/v1`, embed `8081/v1`), omit `api_key`; see [provider_profiles.md](references/provider_profiles.md)
- For Ollama (legacy): set `base_url`, leave `api_key`/`azure_endpoint`/`deployment_name` empty
- For OpenAI: leave `azure_endpoint`/`deployment_name`/`api_version` empty
- If vision disabled: set `vision_llm.enabled: false`
- For rerank: set `enabled`, `provider`, and `model` accordingly

Write the generated config to `config/settings.yaml`.

Before writing, if `config/settings.yaml` already exists, copy it to `config/settings.yaml.bak` first. Then generate a merged config rather than blindly overwriting the previous file.

Also ensure required directories exist:

```powershell
python -c "from pathlib import Path; [Path(d).mkdir(parents=True, exist_ok=True) for d in ['data/db/chroma', 'data/images/default', 'logs', 'config/prompts']]"
```

---

## Step 4: Install Dependencies

```powershell
python -m pip install -e ".[dev]"
```

If specific providers need extra packages:
- **Cross-Encoder rerank**: `python -m pip install sentence-transformers`
- **Streamlit dashboard**: `python -m pip install streamlit`
- **OpenAI**: `python -m pip install openai`

Verify critical imports:

```powershell
python -c "import chromadb; import mcp; import yaml; print('Core deps OK')"
python -c "import streamlit; print('Streamlit OK')"
python -c "import openai; print('OpenAI SDK OK')"
```

---

## Step 5: Validate Configuration

Test that the config loads correctly:

```powershell
python -c "from src.core.settings import load_settings; s = load_settings(); print(f'Config OK: LLM={s.llm.provider}/{s.llm.model}, Embed={s.embedding.provider}/{s.embedding.model}')"
```

If the config loads, perform a lightweight provider initialization check when credentials are present. If available, attempt to instantiate the selected provider class or run a minimal request probe; if that fails, treat it as a credential/network issue and continue to the auto-fix loop.

If this fails, enter **auto-fix loop**:

### Auto-Fix Loop (≤3 rounds)

```
Round 0..2:
  Read error message
  Diagnose root cause (missing field, wrong type, bad provider name, etc.)
  Fix config/settings.yaml or install missing dependency
  Re-validate
  If pass → continue to Step 6
  If fail → next round
```

Common fixes:
- `SettingsError: Missing required field` → add the field to settings.yaml
- `ModuleNotFoundError` → `pip install <package>`
- `Connection refused` (LlamaCpp) → inform user to start `llama-server` (check port 8080/8081)
- `Connection refused` (Ollama, legacy) → inform user to start Ollama service
- `未知的 LLM provider: 'llamacpp'` → implement B7.9/B7.10 per [new_provider_guide.md](references/new_provider_guide.md)
- Wrong `dimensions` value → look up correct value from provider_profiles.md

If 3 rounds fail, report the issue to the user with diagnosis and ask for help.

---

## Step 6: Launch Dashboard

```powershell
python scripts/start_dashboard.py --port 8501
```

Run this as a **background process**. Wait a few seconds, then verify it's accessible:

```powershell
python -c "
import urllib.request
try:
    r = urllib.request.urlopen('http://localhost:8501/_stcore/health')
    print('Dashboard is running!' if r.status == 200 else f'Status: {r.status}')
except Exception as e:
    print(f'Dashboard not yet ready: {e}')
"
```

If the dashboard fails to start, enter auto-fix loop:
- Read the error output from the background terminal
- Common issues: missing `streamlit`, port already in use, import errors
- Fix and retry

---

## Step 7: Usage Guide

After successful launch, present this to the user:

```
🎉 Setup Complete!

Dashboard: http://localhost:8501

Quick Start:
  1. Ingest documents:  python scripts/ingest.py <path-to-pdf-or-folder>
  2. Query:             python scripts/query.py "your question here"
  3. Dashboard:         python scripts/start_dashboard.py
  4. MCP Server:        python main.py

Configuration: config/settings.yaml
Logs:          logs/traces.jsonl

Provider: {provider} / Model: {model}
```

Adapt the message based on the user's chosen providers and language (Chinese if user communicates in Chinese).
