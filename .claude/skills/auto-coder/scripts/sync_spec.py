#!/usr/bin/env python3
"""
Spec Sync — splits DEV_SPEC.md into chapter files under auto-coder/references/.

Usage:
    python {{SKILL_ROOT}}/scripts/sync_spec.py [--force]
"""

import hashlib
import re
import sys
from pathlib import Path
from typing import List, Tuple, NamedTuple


class Chapter(NamedTuple):
    number: int
    cn_title: str
    filename: str
    start_line: int
    end_line: int
    line_count: int


# Chapter number -> English slug (encoding-independent)
NUMBER_SLUG_MAP = {
    1: "overview",
    2: "features",
    3: "tech-stack",
    4: "testing",
    5: "architecture",
    6: "schedule",
    7: "future",
}


def _slug(chapter_num: int, title: str) -> str:
    if chapter_num in NUMBER_SLUG_MAP:
        return NUMBER_SLUG_MAP[chapter_num]
    # Fallback: sanitize whatever title text we have
    clean = re.sub(r'[^\w]+', '-', title, flags=re.ASCII).strip('-').lower()
    return clean or f"chapter-{chapter_num}"


def detect_chapters(content: str) -> List[Chapter]:
    lines = content.split('\n')
    starts: List[Tuple[int, str, int]] = []
    for i, line in enumerate(lines):
        m = re.match(r'^## (\d+)\.\s+(.+)$', line)
        if m:
            starts.append((int(m.group(1)), m.group(2).strip(), i))
    if not starts:
        raise ValueError("No chapters found. Expected '## N. Title'")
    chapters = []
    for idx, (num, title, start) in enumerate(starts):
        end = starts[idx + 1][2] if idx + 1 < len(starts) else len(lines)
        chapters.append(Chapter(num, title, f"{num:02d}-{_slug(num, title)}.md", start, end, end - start))
    return chapters


TASK_ID_RE = re.compile(r'^[A-I]\d+(?:\.\d+)?$')
STATUS_DONE = {"[x]", "✅"}
OVERALL_TABLE_RE = re.compile(
    r"### 📈 总体进度\n\n"
    r"\| 阶段 \| 总任务数 \| 已完成 \| 进度 \|\n"
    r"\|[-| ]+\|\n"
    r"(?:\|.*\|\n)+",
)


def recount_overall_progress(content: str) -> str:
    """Rebuild the 📈 总体进度 table from 📊 进度跟踪表 [x] counts.

    The summary table is derived data. Persist only flips per-task markers;
    this function keeps the rollup in sync so references/06-schedule.md
    does not stay at 0/0 after a successful task cycle.
    """
    tracking_start = content.find("### 📊 进度跟踪表")
    overall_start = content.find("### 📈 总体进度")
    if tracking_start < 0 or overall_start < 0 or overall_start <= tracking_start:
        print("WARNING: progress tables not found; skip recount")
        return content

    phases = {letter: {"total": 0, "done": 0} for letter in "ABCDEFGHI"}
    for line in content[tracking_start:overall_start].split("\n"):
        cols = [col.strip() for col in line.split("|")]
        # markdown row: ['', task_id, name, status, date, note, '']
        if len(cols) < 5 or not TASK_ID_RE.fullmatch(cols[1]):
            continue
        phase = cols[1][0]
        phases[phase]["total"] += 1
        if cols[3] in STATUS_DONE:
            phases[phase]["done"] += 1

    rows = [
        "| 阶段 | 总任务数 | 已完成 | 进度 |",
        "|------|---------|--------|------|",
    ]
    grand_total = grand_done = 0
    for letter in "ABCDEFGHI":
        total, done = phases[letter]["total"], phases[letter]["done"]
        grand_total += total
        grand_done += done
        pct = 0 if total == 0 else round(done / total * 100)
        rows.append(f"| 阶段 {letter} | {total} | {done} | {pct}% |")
    grand_pct = 0 if grand_total == 0 else round(grand_done / grand_total * 100)
    rows.append(f"| **总计** | **{grand_total}** | **{grand_done}** | **{grand_pct}%** |")
    new_table = "### 📈 总体进度\n\n" + "\n".join(rows) + "\n"

    updated, n = OVERALL_TABLE_RE.subn(new_table, content, count=1)
    if n == 0:
        print("WARNING: overall progress table not replaced; skip recount")
        return content
    print(f"recounted progress: {grand_done}/{grand_total} ({grand_pct}%)")
    return updated


def sync(force: bool = False):
    skill_dir = Path(__file__).parent.parent          # auto-coder/
    repo_root = skill_dir.parent.parent.parent        # project root
    dev_spec  = repo_root / "DEV_SPEC.md"
    specs_dir = skill_dir / "references"
    hash_file = skill_dir / ".spec_hash"

    if not dev_spec.exists():
        print(f"ERROR: {dev_spec} not found"); sys.exit(1)

    original = dev_spec.read_text(encoding='utf-8')
    content = recount_overall_progress(original)
    if content != original:
        # Write derived rollup back to the source of truth before hashing.
        dev_spec.write_text(content, encoding='utf-8')

    # Hash check
    current_hash = hashlib.sha256(dev_spec.read_bytes()).hexdigest()
    if not force and hash_file.exists() and hash_file.read_text().strip() == current_hash:
        print("specs up-to-date"); return

    chapters = detect_chapters(content)
    lines = content.split('\n')

    specs_dir.mkdir(parents=True, exist_ok=True)

    # Clean orphans
    old = {f.name for f in specs_dir.glob("*.md")}
    new = {ch.filename for ch in chapters}
    for f in old - new:
        (specs_dir / f).unlink()

    # Write chapters
    for ch in chapters:
        (specs_dir / ch.filename).write_text('\n'.join(lines[ch.start_line:ch.end_line]), encoding='utf-8')

    hash_file.write_text(current_hash)
    print(f"synced {len(chapters)} chapters")


if __name__ == "__main__":
    sync(force="--force" in sys.argv)
