# Modular RAG MCP Server

可插拔的检索增强生成（RAG）服务：Hybrid Search（Dense + BM25）+ MCP Tools + Streamlit Dashboard。  
默认走 **llama.cpp 本地推理**（按需启停 `llama-server`）；也可换成 OpenAI / Azure / DeepSeek。

目标：约 10 分钟内完成安装 → 配置 → 首次摄取 → 查询 → Dashboard → 测试，并能接到 GitHub Copilot / Claude Desktop。

---

## 目录

- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [MCP 配置示例](#mcp-配置示例)
- [Dashboard 使用指南](#dashboard-使用指南)
- [运行测试](#运行测试)
- [常见问题](#常见问题)

---

## 快速开始

### 1. 安装依赖

需要 **Python 3.10+**。在仓库根目录：

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS / Linux
# source .venv/bin/activate

pip install -e ".[dev,dashboard,ingestion]"
```

可选 extra：

| Extra | 用途 |
|-------|------|
| `ingestion` | PDF 解析（MarkItDown） |
| `dashboard` | Streamlit 管理台 |
| `evaluation` | Ragas 评估 |
| `rerank` | Cross-Encoder 精排 |
| `dev` | pytest |

评估面板要用 Ragas 时再装：`pip install -e ".[evaluation]"`。

### 2. 配置 API Key / 本地模型

编辑 `config/settings.yaml`。

**方案 A — 本地 llama.cpp（仓库默认）**

1. 填写 `llamacpp.server_bin`（`llama-server` 可执行文件）。
2. 填写 `llm.model_path`、`embedding.model_path`（GGUF 路径）。
3. 保持 `llm.provider` / `embedding.provider` 为 `llamacpp`。
4. 先关掉已占用 8080/8081 的 `llama-server`。系统会在调用时按需拉起，空闲后释放 GPU。

本地路径不需要云端 API Key。

**方案 B — 云端 API**

在 `llm` / `embedding` 中设置 `provider` 与 `api_key`，或改用环境变量：

| Provider | 环境变量 |
|----------|----------|
| OpenAI | `OPENAI_API_KEY` |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` |
| DeepSeek | `DEEPSEEK_API_KEY` |

`settings.yaml` 里的 `api_key` 优先于环境变量。不要把真实 Key 提交进 Git。

### 3. 首次摄取

当前摄取入口只接受 **PDF**（单文件或目录顶层 `*.pdf`）：

```bash
python scripts/ingest.py --path tests/fixtures/sample_documents/sample.pdf --collection knowledge_hub --force
```

`--collection` 建议与 `vector_store.collection_name` 一致（默认 yaml 为 `knowledge_hub`）。CLI 若省略 `--collection` 会写成 `default`，查询时对不上。

成功后会在 `data/db/chroma`、`data/db/bm25/` 写入索引。

### 4. 查询与 Dashboard

```bash
python scripts/query.py --query "这份文档讲了什么？" --collection knowledge_hub --verbose
python scripts/start_dashboard.py
```

浏览器打开提示的地址（默认 `http://localhost:8501`）。

配置校验（不启动 MCP 循环）：

```bash
python main.py
```

应打印 `Modular RAG MCP Server — ready`。

---

## 配置说明

主文件：`config/settings.yaml`。全项目通过 `load_settings()` 读取，业务代码不要硬编码路径。

| 块 | 关键字段 | 含义 |
|----|----------|------|
| `llm` | `provider`, `model`, `base_url`, `temperature`, `max_tokens`, `model_path`, `api_key` | 生成 / 评估 Judge 用的聊天模型。`provider`：`llamacpp` / `openai` / `azure` / `deepseek` / `ollama` |
| `embedding` | `provider`, `model`, `base_url`, `dimensions`, `model_path` | 查询与入库向量。默认 bge-m3、1024 维 |
| `llamacpp` | `auto_manage`, `server_bin`, `exclusive_gpu`, `idle_timeout`, `startup_timeout` | 共享 llama-server 运行时。`exclusive_gpu: true` 时 LLM 与 Embedding 互斥占 GPU |
| `vector_store` | `provider`, `persist_directory`, `collection_name` | 目前 `chroma`；集合名须与 ingest/query `--collection` 对齐 |
| `retrieval` | `dense_top_k`, `sparse_top_k`, `fusion_top_k`, `rrf_k` | Dense / BM25 召回条数与 RRF 融合 |
| `rerank` | `enabled`, `provider`, `model`, `top_k` | 精排。`enabled: false` 时跳过 |
| `evaluation` | `enabled`, `provider`, `metrics`, `backends`（可选） | `provider`/`backends`：`custom` / `ragas`。多项 backends 会组合成 CompositeEvaluator |
| `observability` | `log_level`, `trace_enabled`, `trace_file`, `structured_logging` | 日志与 `traces.jsonl` 追踪 |
| `ingestion` | `chunk_size`, `chunk_overlap`, `splitter`, `batch_size`, `chunk_refiner`, `metadata_enricher` | 切分与可选 LLM 精炼。默认 `chunk_size: 400`、`overlap: 80` |

切换云端示例（LLM）：

```yaml
llm:
  provider: openai
  model: gpt-4o-mini
  api_key: "YOUR_API_KEY_HERE"
  temperature: 0.0
  max_tokens: 4096
```

更完整的 Provider 样例见 `.github/skills/setup/references/provider_profiles.md`。

---

## MCP 配置示例

MCP Server 走 **Stdio**：stdout 只输出 JSON-RPC，日志在 stderr。安装 editable 包后用 **venv 里的 python** 启动：

```bash
python -m mcp_server.server
```

对外 Tools：`query_knowledge_hub`、`list_collections`、`get_document_summary`。

把下面的 `<REPO>` 换成仓库绝对路径，`<PYTHON>` 换成 `.venv` 中的解释器。

### GitHub Copilot（`mcp.json`）

VS Code / Copilot 常见位置：项目 `.vscode/mcp.json` 或用户级 MCP 配置。

```json
{
  "servers": {
    "modular-rag": {
      "type": "stdio",
      "command": "<PYTHON>",
      "args": ["-m", "mcp_server.server"],
      "cwd": "<REPO>"
    }
  }
}
```

Windows 示例：

```json
{
  "servers": {
    "modular-rag": {
      "type": "stdio",
      "command": "D:\\JDIT\\MODULAR-RAG-MCP-SERVER\\.venv\\Scripts\\python.exe",
      "args": ["-m", "mcp_server.server"],
      "cwd": "D:\\JDIT\\MODULAR-RAG-MCP-SERVER"
    }
  }
}
```

在 Copilot Chat 中应能看到 `query_knowledge_hub`；调用时传入 `query`，可选 `top_k`、`collection`。

### Claude Desktop（`claude_desktop_config.json`）

常见路径：

- Windows：`%APPDATA%\Claude\claude_desktop_config.json`
- macOS：`~/Library/Application Support/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "modular-rag": {
      "command": "<PYTHON>",
      "args": ["-m", "mcp_server.server"],
      "cwd": "<REPO>"
    }
  }
}
```

改完后重启 Claude Desktop。若 Client 找不到 `mcp_server` 包，确认已 `pip install -e .`，且 `command` 指向该 venv 的 Python。

---

## Dashboard 使用指南

### 启动

```bash
pip install -e ".[dashboard]"
python scripts/start_dashboard.py
```

默认端口 **8501**。脚本会把 `src/` 加入 `PYTHONPATH` 并以 headless 方式拉起 Streamlit。

### 页面示意（截图示例）

启动后左侧为分组导航，主区为当前页。结构如下（可用本机界面对照）：

```
┌─────────────────────────────────────────────────────────┐
│ Modular RAG Dashboard                                   │
├────────────┬────────────────────────────────────────────┤
│ 观测       │  系统总览                                   │
│  · 系统总览 │  [LLM] [Embedding] [Splitter] [Rerank] …  │
│  · 数据浏览 │  集合 / Chunk 数 / 文档数                    │
│            │                                            │
│ 摄取       │                                            │
│  · 管理    │                                            │
│  · 追踪    │                                            │
│            │                                            │
│ 查询与评估 │                                            │
│  · Query   │                                            │
│  · 评估面板 │                                            │
└────────────┴────────────────────────────────────────────┘
```

| 页面 | 做什么 |
|------|--------|
| **系统总览** | 当前 LLM / Embedding / Splitter / Reranker / Evaluator 与知识库资产统计 |
| **数据浏览器** | 按集合浏览文档与 Chunk，可按 source 关键词筛选 |
| **Ingestion 管理** | 上传或填写 PDF 路径摄取，进度条，删除已摄入文档 |
| **Ingestion 追踪** | `traces.jsonl` 里 `ingestion` 记录：阶段瀑布图、成功/跳过/失败 |
| **Query 追踪** | 查询历史、Dense vs Sparse、Rerank 排名变化 |
| **评估面板** | 选集合 / Custom·Ragas·All / 黄金集，运行评估；可从已摄入 chunk 生成黄金集 |

摄取或查询前请打开 `observability.trace_enabled`（默认 true），否则追踪页没有数据。

评估：Custom 看 hit_rate / mrr；Ragas / All 会先用项目 LLM 根据检索上下文生成答案，再用同一套 LLM/Embedding 做 Judge（需 `.[evaluation]`）。

---

## 运行测试

先激活 `.venv`。

```bash
# 单元测试
pytest -q tests/unit

# 集成测试
pytest -q tests/integration

# E2E（MCP Client 模拟、Dashboard 冒烟、召回回归、摄取 CLI）
pytest -q tests/e2e

# 全量
pytest -q
```

按标记：

```bash
pytest -q -m unit
pytest -q -m integration
pytest -q -m e2e
```

常用单文件：

```bash
pytest -q tests/e2e/test_mcp_client.py
pytest -q tests/e2e/test_dashboard_smoke.py
pytest -q tests/e2e/test_recall.py
```

评估 CLI（需已有索引；黄金集默认 `tests/fixtures/golden_test_set.json`）：

```bash
python scripts/evaluate.py --collection knowledge_hub
python scripts/evaluate.py --json
```

---

## 常见问题

**Q: 提示缺少 API Key？**  
A: 云端 Provider 需配置 `llm.api_key` / `embedding.api_key`，或对应环境变量（`OPENAI_API_KEY`、`AZURE_OPENAI_API_KEY`、`DEEPSEEK_API_KEY`）。llamacpp 一般不需要真实 Key。

**Q: `pip install` 失败 / 找不到 `mcp_server`？**  
A: 确认在仓库根执行 `pip install -e .`，并用 **同一个 venv** 跑脚本与 MCP。MCP 的 `command` 不要指向系统 Python。

**Q: 摄取成功但查询为空？**  
A: 核对 ingest 与 query/MCP 的 `--collection` 是否一致；CLI 默认集合是 `default`，yaml 默认是 `knowledge_hub`。

**Q: llama-server 端口占用 / 启动超时？**  
A: 关掉手动启动的 `llama-server`。增大 `llamacpp.startup_timeout`。确认 `server_bin` 与 `model_path` 存在。日志：`logs/llamacpp-embedding.log` 等。

**Q: Dashboard 起不来？**  
A: `pip install -e ".[dashboard]"`，再 `python scripts/start_dashboard.py`。不要直接 `streamlit run` 却漏掉 `src` 的 PYTHONPATH（启动脚本已处理）。

**Q: PDF 解析失败？**  
A: 安装 `.[ingestion]`。当前 MVP Loader 只支持 `.pdf`。

**Q: Ragas 指标全空或超时？**  
A: 安装 `.[evaluation]`。Judge 走 `settings.llm` / `settings.embedding`，不会回退官方 OpenAI。本地模型首次加载可能要数分钟。

**Q: MCP Client 连上但工具报错？**  
A: 看 Claude/Copilot 日志里的 stderr。stdout 必须是纯 JSON-RPC，不要把 `print` 打到 stdout。

---

## 项目结构（简表）

- `src/core` — 配置、查询流水线、契约类型  
- `src/ingestion` — 离线摄取 Pipeline  
- `src/libs` — 可插拔 LLM / Embedding / VectorStore / Reranker / Evaluator  
- `src/mcp_server` — Stdio MCP  
- `src/observability` — 日志、Trace、Dashboard、评估  
- `config/settings.yaml` — 主配置  
- `scripts/` — `ingest.py` / `query.py` / `evaluate.py` / `start_dashboard.py`  
- `tests/` — unit / integration / e2e  

完整设计见 `DEV_SPEC.md`。
