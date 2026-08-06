# Modular RAG MCP Server

Pluggable RAG (Retrieval-Augmented Generation) server with MCP tools, hybrid retrieval, and Streamlit dashboard.

## Quick start

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

pip install -e .
python main.py
```

## Project layout

- `src/` — application code (`core`, `ingestion`, `libs`, `mcp_server`, `observability`)
- `config/settings.yaml` — main configuration
- `tests/` — unit, integration, and e2e tests

See `DEV_SPEC.md` for full architecture and development schedule.
