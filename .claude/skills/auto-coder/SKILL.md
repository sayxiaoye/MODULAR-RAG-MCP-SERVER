---
name: auto-coder
description: Autonomous spec-driven development agent. Syncs DEV_SPEC.md into chapter-based reference files, identifies the next pending task from the schedule, implements code following spec architecture and patterns (with Chinese comments for readability), runs tests with up to 3 auto-fix rounds, and persists progress with atomic commits. Use when user says "auto code", "自动开发", "自动写代码", "auto dev", "一键开发", "autopilot", or wants fully automated spec-to-code workflow.
---

# Auto Coder

One trigger completes **read spec → find task → code → test → persist progress**.

Optional modifiers: append a task ID (e.g. `auto code B2`) to target a specific task, or `--no-commit` to skip git commit.

---

## Pipeline

```
Sync Spec → Find Task → Implement → Test (≤3 fix rounds) → Persist
```

Pause only at the end for commit confirmation. Run everything else autonomously.

> **⚠️ CRITICAL: Activate `.venv` before ANY `python`/`pytest` command (idempotent, re-run if unsure).**
> - **Windows**: `.\.venv\Scripts\Activate.ps1`
> - **macOS/Linux**: `source .venv/bin/activate`

## Reference Map

All files under `{{SKILL_ROOT}}/references/`:

| File | Content | When to Read |
|------|---------|-------------|
| `01-overview.md` | Project overview & goals | First task or when needing project context |
| `02-features.md` | Feature specifications | When implementing feature-related tasks |
| `03-tech-stack.md` | Tech stack & dependencies | When choosing libraries or patterns |
| `04-testing.md` | Testing conventions | When writing tests |
| `05-architecture.md` | Architecture & module design | When creating/modifying modules |
| `06-schedule.md` | Task schedule & status | Every cycle (Sync Spec step) |
| `07-future.md` | Future roadmap | When planning or assessing scope |

---

### 1. Sync Spec

```powershell
python {{SKILL_ROOT}}/scripts/sync_spec.py
```

Then read the schedule file to get task statuses:
- Read `{{SKILL_ROOT}}/references/06-schedule.md`

Task markers:

| Marker | Status |
|--------|--------|
| `[ ]` / `⬜` | Not started |
| `[~]` / `🔶` / `(进行中)` | In progress |
| `[x]` / `✅` / `(已完成)` | Completed |

---

### 2. Find Task

Pick the first `IN_PROGRESS` task, then the first `NOT_STARTED`. If user specified a task ID, use that directly.

Quick-check predecessor artifacts exist (file-level only). On mismatch, log a warning and continue — only stop if the target task itself is blocked.

---

### 3. Implement

1. **Read relevant spec** from `{{SKILL_ROOT}}/references/`:
   - Architecture: `05-architecture.md`
   - Tech details: `03-tech-stack.md`
   - Testing conventions: `04-testing.md`

2. **Extract** from spec: inputs/outputs, design principles (Pluggable? Config-driven? Factory?), file list, acceptance criteria.

3. **Plan** files to create/modify before writing any code.

4. **Code** — project-specific rules:
   - Treat spec as single source of truth
   - Use `config/settings.yaml` values, never hardcode
   - Match existing codebase patterns and style
   - **Add Chinese comments** so readers can quickly grasp intent (see below)

5. **Write tests** alongside code:
   - Place in `tests/unit/` or `tests/integration/` per spec
   - Mock external deps in unit tests

6. **Self-review** before running tests: verify all planned files exist, tests import correctly, and new/changed code includes Chinese comments per the rules below.

#### Chinese Comment Rules (生成代码时必须遵守)

Generated **production code** and **tests** must include **Chinese** explanations. Goal: a reader can understand *what* each unit does and *why* non-obvious logic exists, without reading the spec.

| Scope | Requirement |
|-------|-------------|
| **Module / file** | Top-of-file docstring (or module comment): 模块职责、在架构中的位置、主要对外接口 |
| **Class** | Class docstring: 类的用途、核心属性/方法、与 spec 中哪块功能对应 |
| **Public function / method** | Docstring: 功能说明、`Args` / `Returns` / `Raises`（如有）、关键副作用 |
| **Non-obvious logic** | Inline `#` comments: 分支原因、算法步骤、边界处理、与配置/工厂的关联 |
| **Tests** | 每个测试类/用例上方简短中文说明：测什么场景、期望行为 |

**Do:**
- 用中文写「为什么」和「做什么」，与 spec 术语保持一致（如 Retriever、Chunk、Provider）
- 在复杂流程前用 1–2 行中文总述（例如 `# 按配置选择 Embedding 实现并缓存单例`）
- 新建文件从第一行就带模块级中文说明

**Don't:**
- 不要为显而易见的语句写注释（如 `# 返回 result`）
- 不要大段复述 spec 原文；注释应贴近代码、简洁可读
- 不要只用英文注释；**中文为主**，标识符与 API 名保持英文

**Minimal example:**

```python
"""配置加载与校验：从 settings.yaml 读取并构造 Settings 单例。"""

class SettingsLoader:
    """将 YAML 配置解析为强类型 Settings，供全项目注入使用。"""

    def load(self, path: Path) -> Settings:
        # 先读文件再校验，避免部分字段缺失导致后续模块初始化失败
        raw = self._read_yaml(path)
        return self._validate(raw)
```

---

### 4. Test & Auto-Fix

```

Round 0..2:
  Run pytest on relevant test file
  If pass → go to step 5
  If fail → analyze error, apply fix, re-run

Round 3 still failing → STOP, show failure report to user
```

---

### 5. Persist

1. **Update `DEV_SPEC.md`** (global file): change task marker `[ ]` → `[x]`
2. **Re-sync**: `python {{SKILL_ROOT}}/scripts/sync_spec.py --force`
3. **Show summary & ask**:

```
✅ [A3] 配置加载与校验 — done
   Files: src/core/settings.py, tests/unit/test_settings.py
   Tests: 8/8 passed
   Commit: feat(config): [A3] implement config loader

   "commit" → git add + commit
   "skip"   → end
   "next"   → commit + start next task
```

On "next", loop back to step 1 and start the next task.
