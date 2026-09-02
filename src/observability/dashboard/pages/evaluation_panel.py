"""评估面板：选择后端与黄金测试集、运行评估、展示指标与历史趋势（H4）。"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import streamlit as st

from core.settings import REPO_ROOT, Settings, load_settings, resolve_path
from observability.evaluation.eval_runner import EvalReport, EvalRunner

# 与 spec 页面 6 一致：Ragas / Custom / All
BACKEND_LABELS = ("Custom", "Ragas", "All")
_BACKEND_MAP: dict[str, list[str]] = {
    "Custom": ["custom"],
    "Ragas": ["ragas"],
    "All": ["ragas", "custom"],
}

RunEvalFn = Callable[[str, str], EvalReport]


def backends_for_label(label: str) -> list[str]:
    """把面板上的后端选项映射为 EvaluatorFactory 的 backends 列表。"""
    key = (label or "").strip()
    if key not in _BACKEND_MAP:
        raise ValueError(f"未知评估后端: {label!r}")
    return list(_BACKEND_MAP[key])


def default_golden_sets() -> list[Path]:
    """发现可用的黄金测试集：默认 fixture + fixtures 目录中含 test_cases 的 JSON。"""
    fixture_dir = REPO_ROOT / "tests" / "fixtures"
    found: list[Path] = []
    seen: set[Path] = set()
    preferred = fixture_dir / "golden_test_set.json"
    if preferred.is_file():
        found.append(preferred)
        seen.add(preferred.resolve())
    if fixture_dir.is_dir():
        for path in sorted(fixture_dir.glob("*.json")):
            resolved = path.resolve()
            if resolved in seen:
                continue
            if _looks_like_golden_set(path):
                found.append(path)
                seen.add(resolved)
    return found


def default_history_path(settings: Settings | None = None) -> Path:
    """历史评估落在 traces.jsonl 同目录的 eval_history.jsonl。"""
    resolved = settings or load_settings()
    trace_file = resolve_path(resolved.observability.trace_file)
    return trace_file.parent / "eval_history.jsonl"


def load_eval_history(path: str | Path) -> list[dict[str, Any]]:
    """读取 JSONL 历史；缺文件返回空列表。"""
    target = Path(path)
    if not target.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in target.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            item = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            records.append(item)
    return records


def append_eval_history(path: str | Path, record: Mapping[str, Any]) -> None:
    """追加一条评估摘要，供历史趋势图使用。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def settings_with_backends(settings: Settings, backends: Sequence[str]) -> Settings:
    """覆盖 evaluation.backends，供工厂组合成 Custom / Ragas / Composite。"""
    names = [str(item).strip() for item in backends if str(item).strip()]
    evaluation = replace(
        settings.evaluation,
        backends=names or None,
        provider=names[0] if names else settings.evaluation.provider,
    )
    return replace(settings, evaluation=evaluation)


def render_evaluation_panel(
    run_eval: RunEvalFn | None = None,
    golden_sets: Sequence[Path | str] | None = None,
    history_records: Sequence[Mapping[str, Any]] | None = None,
    history_path: Path | str | None = None,
    *,
    load_deps: bool = True,
) -> None:
    """
    渲染评估面板。

    Args:
        run_eval: ``(backend_label, test_set_path) -> EvalReport``；测试注入 Fake。
        golden_sets: 可选测试集路径列表；缺省扫描 fixtures。
        history_records: 注入的历史记录（优先于读文件）。
        history_path: 历史 JSONL 路径；测试可指向临时文件。
        load_deps: 为 False 时不访问真实检索栈与默认 traces 目录。
    """
    st.header("评估面板")
    st.caption("选择评估后端与黄金测试集，运行后查看 hit_rate / mrr 及历史对比")

    sets = [Path(item) for item in golden_sets] if golden_sets is not None else []
    if not sets and load_deps:
        sets = default_golden_sets()
    if not sets:
        st.warning("未找到黄金测试集。请准备 tests/fixtures/golden_test_set.json。")
        return

    backend = st.selectbox("评估后端", list(BACKEND_LABELS))
    labels = [path.name for path in sets]
    selected_index = st.selectbox(
        "Golden Test Set",
        list(range(len(sets))),
        format_func=lambda index: labels[index],
    )
    test_set_path = sets[int(selected_index)]

    resolved_history_path = Path(history_path) if history_path is not None else None
    if resolved_history_path is None and load_deps:
        resolved_history_path = default_history_path()

    if st.button("运行评估"):
        runner = run_eval if run_eval is not None else _default_run_eval
        try:
            with st.spinner("正在运行评估…"):
                report = runner(str(backend), str(test_set_path))
        except Exception as exc:
            st.error(f"评估失败: {exc}")
        else:
            st.session_state["eval_last_report"] = report
            record = _history_record(str(backend), str(test_set_path), report)
            st.session_state.setdefault("eval_history_extra", []).append(record)
            if resolved_history_path is not None:
                append_eval_history(resolved_history_path, record)
            st.success(
                f"评估完成：{report.case_count} 条用例 · "
                f"hit_rate={report.hit_rate:.4f} · mrr={report.mrr:.4f}"
            )

    report = st.session_state.get("eval_last_report")
    if isinstance(report, EvalReport):
        _render_report(report)
    else:
        st.info("点击「运行评估」后将展示 hit_rate、mrr 与各 query 明细。")

    history = _merge_history(history_records, resolved_history_path)
    extras = st.session_state.get("eval_history_extra") or []
    if extras:
        history = list(history) + [item for item in extras if item not in history]
    _render_history(history)


def render() -> None:
    """Streamlit 页面入口。"""
    render_evaluation_panel()


def _looks_like_golden_set(path: Path) -> bool:
    """快速判断 JSON 是否含 test_cases，避免把无关 fixture 放进下拉框。"""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(payload, Mapping) and isinstance(payload.get("test_cases"), list)


def _default_run_eval(backend_label: str, test_set_path: str) -> EvalReport:
    """用当前 Settings + HybridSearch + 工厂评估器跑黄金集。"""
    from core.query_engine.dense_retriever import DenseRetriever
    from core.query_engine.hybrid_search import HybridSearch
    from core.query_engine.query_pipeline import settings_for_query
    from core.query_engine.sparse_retriever import SparseRetriever
    from ingestion.storage.bm25_indexer import BM25Indexer, BM25IndexerError
    from libs.evaluator.evaluator_factory import EvaluatorFactory
    from libs.vector_store.vector_store_factory import VectorStoreFactory

    base = load_settings()
    names = backends_for_label(backend_label)
    tuned = settings_with_backends(base, names)
    settings, bm25_root = settings_for_query(tuned, None, None)
    vector_store = VectorStoreFactory.create(settings)
    indexer = BM25Indexer(
        collection=settings.vector_store.collection_name,
        index_root=bm25_root,
    )
    try:
        indexer.load()
    except BM25IndexerError:
        pass
    hybrid_search = HybridSearch(
        settings,
        dense_retriever=DenseRetriever(settings, vector_store=vector_store),
        sparse_retriever=SparseRetriever(
            settings,
            bm25_indexer=indexer,
            vector_store=vector_store,
        ),
    )
    evaluator = EvaluatorFactory.create(settings)
    return EvalRunner(settings, hybrid_search, evaluator).run(test_set_path)


def _history_record(backend: str, test_set_path: str, report: EvalReport) -> dict[str, Any]:
    """把本次报告压成历史趋势可用的摘要。"""
    return {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "test_set": test_set_path,
        "hit_rate": report.hit_rate,
        "mrr": report.mrr,
        "case_count": report.case_count,
        "metrics": dict(report.metrics),
    }


def _merge_history(
    injected: Sequence[Mapping[str, Any]] | None,
    history_path: Path | None,
) -> list[dict[str, Any]]:
    """注入记录优先；否则读 JSONL。"""
    if injected is not None:
        return [dict(item) for item in injected]
    if history_path is None:
        return []
    return load_eval_history(history_path)


def _render_report(report: EvalReport) -> None:
    """展示宏观指标与各 query 明细表。"""
    st.subheader("本次结果")
    metric_items: list[tuple[str, float]] = [
        ("hit_rate", report.hit_rate),
        ("mrr", report.mrr),
    ]
    for key, value in sorted(report.metrics.items()):
        if key in {"hit_rate", "mrr"}:
            continue
        metric_items.append((key, float(value)))
    columns = st.columns(len(metric_items) or 1)
    for column, (name, value) in zip(columns, metric_items):
        column.metric(name, f"{value:.4f}")

    st.markdown("**各 query 明细**")
    st.dataframe(_case_rows(report), width="stretch", hide_index=True)


def _case_rows(report: EvalReport) -> list[dict[str, Any]]:
    """把 EvalCaseResult 转成表格行。"""
    rows: list[dict[str, Any]] = []
    for item in report.cases:
        rows.append(
            {
                "query": item.query,
                "hit_rate": item.metrics.get("hit_rate"),
                "mrr": item.metrics.get("mrr"),
                "retrieved": ", ".join(item.retrieved_ids),
                "golden": ", ".join(item.golden_ids),
                "error": item.error or "",
            }
        )
    return rows


def _render_history(records: Sequence[Mapping[str, Any]]) -> None:
    """两条及以上历史时画 hit_rate / mrr 趋势。"""
    st.subheader("历史趋势")
    if len(records) < 2:
        st.caption("再运行一次评估后可对比历史趋势。")
        return
    chart = {
        "hit_rate": [float(item.get("hit_rate") or 0.0) for item in records],
        "mrr": [float(item.get("mrr") or 0.0) for item in records],
    }
    st.line_chart(chart)
    st.dataframe(
        [
            {
                "时间": str(item.get("ran_at") or "—"),
                "后端": str(item.get("backend") or "—"),
                "hit_rate": item.get("hit_rate"),
                "mrr": item.get("mrr"),
            }
            for item in records
        ],
        width="stretch",
        hide_index=True,
    )
