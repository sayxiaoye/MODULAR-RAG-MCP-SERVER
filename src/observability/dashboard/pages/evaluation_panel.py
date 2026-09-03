"""评估面板：选择后端与黄金测试集、运行评估、展示指标与历史趋势（H4）。"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import streamlit as st

from core.settings import REPO_ROOT, Settings, load_settings, resolve_path
from observability.evaluation.eval_runner import (
    EvalCaseResult,
    EvalReport,
    EvalRunner,
    load_golden_test_set,
)
from observability.evaluation.golden_generator import (
    DEFAULT_CASE_COUNT,
    MAX_CASE_COUNT,
    GoldenGeneratorError,
    generate_golden_test_set,
    generated_golden_dir,
)

# 与 spec 页面 6 一致：Ragas / Custom / All
BACKEND_LABELS = ("Custom", "Ragas", "All")
_BACKEND_MAP: dict[str, list[str]] = {
    "Custom": ["custom"],
    "Ragas": ["ragas"],
    "All": ["ragas", "custom"],
}

# 历史表固定展示的指标列：Custom 两项 + Ragas 三项
HISTORY_METRIC_KEYS = (
    "hit_rate",
    "mrr",
    "faithfulness",
    "answer_relevancy",
    "context_precision",
)

# (backend_label, test_set_path, collection) -> EvalReport
RunEvalFn = Callable[[str, str, str], EvalReport]
# (collection, count) -> 落盘路径
GenerateGoldenFn = Callable[[str, int], Path]


def backends_for_label(label: str) -> list[str]:
    """把面板上的后端选项映射为 EvaluatorFactory 的 backends 列表。"""
    key = (label or "").strip()
    if key not in _BACKEND_MAP:
        raise ValueError(f"未知评估后端: {label!r}")
    return list(_BACKEND_MAP[key])


def discover_eval_collections(settings: Settings | None = None) -> list[str]:
    """列出可供评估的逻辑集合：yaml 默认在前，其余为 Chroma 已有名称。"""
    resolved = settings or load_settings()
    preferred = (resolved.vector_store.collection_name or "").strip() or "knowledge_hub"
    names: list[str] = []
    try:
        from observability.dashboard.services.data_service import DataService

        names = [
            str(item).strip()
            for item in DataService.from_settings(resolved).list_collections()
            if str(item).strip()
        ]
    except Exception:
        names = []
    ordered: list[str] = []
    seen: set[str] = set()
    for name in [preferred, *names]:
        if name and name not in seen:
            seen.add(name)
            ordered.append(name)
    return ordered or [preferred]


def default_golden_sets() -> list[Path]:
    """发现可用的黄金测试集：仓库 fixture + data/eval 下生成的 JSON。"""
    found: list[Path] = []
    seen: set[Path] = set()
    fixture_dir = REPO_ROOT / "tests" / "fixtures"
    preferred = fixture_dir / "golden_test_set.json"
    if preferred.is_file():
        found.append(preferred)
        seen.add(preferred.resolve())
    for directory in (fixture_dir, generated_golden_dir()):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
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
    collections: Sequence[str] | None = None,
    generate_golden: GenerateGoldenFn | None = None,
    *,
    load_deps: bool = True,
) -> None:
    """
    渲染评估面板。

    Args:
        run_eval: ``(backend_label, test_set_path, collection) -> EvalReport``；测试注入 Fake。
        golden_sets: 可选测试集路径列表；缺省扫描 fixtures 与 data/eval。
        history_records: 注入的历史记录（优先于读文件）。
        history_path: 历史 JSONL 路径；测试可指向临时文件。
        collections: 集合下拉选项；缺省且 ``load_deps`` 时从 Chroma 发现。
        generate_golden: ``(collection, count) -> Path``；缺省调用 generate_golden_test_set。
        load_deps: 为 False 时不访问真实检索栈与默认 traces 目录。
    """
    st.header("评估面板")
    st.caption(
        "Custom 看 hit_rate / mrr；Ragas 会先用项目 LLM 生成答案，再用同一套 LLM/Embedding 做 Judge"
    )

    sets = _merge_golden_sets(golden_sets, load_deps=load_deps)

    collection_names = [str(item).strip() for item in collections or () if str(item).strip()]
    if not collection_names and load_deps:
        collection_names = discover_eval_collections()
    if not collection_names:
        collection_names = ["knowledge_hub"]

    selected_collection = str(st.selectbox("集合", collection_names) or "").strip()
    backend = st.selectbox("评估后端", list(BACKEND_LABELS))

    st.subheader("黄金测试集")
    case_count = int(
        st.number_input(
            "生成条数",
            min_value=1,
            max_value=MAX_CASE_COUNT,
            value=DEFAULT_CASE_COUNT,
            step=1,
        )
    )
    if st.button("生成黄金集"):
        _handle_generate_golden(
            selected_collection,
            case_count,
            generate_golden=generate_golden,
            load_deps=load_deps,
        )
        sets = _merge_golden_sets(golden_sets, load_deps=load_deps)

    if not sets:
        st.warning("还没有黄金测试集。请选择集合后点击「生成黄金集」，或放入 JSON。")
        test_set_path: Path | None = None
    else:
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
        if not selected_collection:
            st.warning("请选择要评估的集合。")
        elif test_set_path is None:
            st.warning("请先生成或选择黄金测试集。")
        else:
            runner = run_eval if run_eval is not None else _default_run_eval
            try:
                with st.spinner("正在运行评估…"):
                    report = runner(str(backend), str(test_set_path), selected_collection)
            except Exception as exc:
                st.error(f"评估失败: {exc}")
            else:
                st.session_state["eval_last_report"] = report
                record = _history_record(
                    str(backend), str(test_set_path), report, collection=selected_collection
                )
                st.session_state.setdefault("eval_history_extra", []).append(record)
                if resolved_history_path is not None:
                    append_eval_history(resolved_history_path, record)
                st.success(
                    f"评估完成：集合 {selected_collection} · {report.case_count} 条用例 · "
                    + _format_success_metrics(report)
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


def _merge_golden_sets(
    golden_sets: Sequence[Path | str] | None,
    *,
    load_deps: bool,
) -> list[Path]:
    """合并扫描结果、注入列表与本次会话新生成的路径。"""
    found: list[Path] = []
    seen: set[Path] = set()

    def _add(path: Path) -> None:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        if resolved in seen:
            return
        if path.is_file() and _looks_like_golden_set(path):
            found.append(path)
            seen.add(resolved)

    if golden_sets is not None:
        for item in golden_sets:
            _add(Path(item))
    elif load_deps:
        for item in default_golden_sets():
            _add(item)

    extras = st.session_state.get("eval_generated_sets") or []
    for item in extras:
        _add(Path(item))
    return found


def _handle_generate_golden(
    collection: str,
    count: int,
    *,
    generate_golden: GenerateGoldenFn | None,
    load_deps: bool,
) -> None:
    """从所选集合生成 N 条黄金用例并登记到下拉列表。"""
    if not collection:
        st.warning("请选择要生成黄金集的集合。")
        return
    runner = generate_golden
    if runner is None:
        if not load_deps:
            st.warning("测试未注入生成器。")
            return
        runner = _default_generate_golden
    try:
        with st.spinner(f"正在从集合 {collection} 生成 {count} 条黄金用例…"):
            path = runner(collection, count)
    except GoldenGeneratorError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error(f"生成黄金集失败: {exc}")
        return
    extras = list(st.session_state.get("eval_generated_sets") or [])
    extras.append(str(path))
    st.session_state["eval_generated_sets"] = extras
    try:
        written = len(load_golden_test_set(path))
    except Exception:
        written = count
    st.success(f"已生成 {written} 条并写入 {Path(path).name}，可在下方下拉框选择。")


def _default_generate_golden(collection: str, count: int) -> Path:
    """面板默认生成入口：读当前 Settings 下的 Chroma 集合。"""
    return generate_golden_test_set(collection, count)


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


def _default_run_eval(backend_label: str, test_set_path: str, collection: str) -> EvalReport:
    """用所选集合组装 HybridSearch，与 CLI ``--collection`` 对齐。"""
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
    target = (collection or "").strip() or None
    settings, bm25_root = settings_for_query(tuned, target, None)
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


def _format_success_metrics(report: EvalReport) -> str:
    """成功提示里带上实际产出的指标，避免 Ragas 只显示 hit_rate=0。"""
    case_keys: set[str] = set()
    for item in report.cases:
        case_keys.update(item.metrics.keys())
    parts: list[str] = []
    for key in HISTORY_METRIC_KEYS:
        if key not in case_keys and key not in report.metrics:
            continue
        if key in {"hit_rate", "mrr"} and key not in case_keys:
            continue
        value = report.metrics.get(key)
        if key == "hit_rate" and value is None:
            value = report.hit_rate
        if key == "mrr" and value is None:
            value = report.mrr
        number = _finite_number(value)
        if number is None:
            continue
        parts.append(f"{key}={number:.4f}")
    return " · ".join(parts) if parts else "无有效指标"


def _finite_number(value: Any) -> float | None:
    """把可展示的有限浮点数取出。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _history_record(
    backend: str,
    test_set_path: str,
    report: EvalReport,
    *,
    collection: str = "",
) -> dict[str, Any]:
    """把本次报告写入历史：宏观指标 + 各 query 明细，供事后点开查看。"""
    metric_names = sorted(
        {key for item in report.cases for key in item.metrics.keys()}
    )
    payload: dict[str, Any] = {
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "backend": backend,
        "test_set": test_set_path,
        "collection": collection,
        "hit_rate": report.hit_rate,
        "mrr": report.mrr,
        "case_count": report.case_count,
        "metrics": dict(report.metrics),
        "metric_names": metric_names,
        "cases": [item.to_dict() for item in report.cases],
    }
    for key in HISTORY_METRIC_KEYS:
        number = _finite_number(report.metrics.get(key))
        if number is not None and key in metric_names:
            payload[key] = number
    return payload


def metric_from_history_record(record: Mapping[str, Any], key: str) -> float | None:
    """从历史行读取一项指标；旧记录没有 Ragas 字段时返回 None。"""
    names = record.get("metric_names")
    if isinstance(names, list) and names and key not in names:
        return None
    backend = str(record.get("backend") or "").strip()
    if not names:
        if key in {"hit_rate", "mrr"} and backend == "Ragas":
            return None
        if key in {"faithfulness", "answer_relevancy", "context_precision"} and backend == "Custom":
            return None
    metrics = record.get("metrics") if isinstance(record.get("metrics"), Mapping) else {}
    value = record.get(key)
    if value is None:
        value = metrics.get(key) if isinstance(metrics, Mapping) else None
    return _finite_number(value)


def report_from_history_record(record: Mapping[str, Any]) -> EvalReport:
    """把历史 JSON 还原成 EvalReport，供点开查看明细。"""
    cases: list[EvalCaseResult] = []
    raw_cases = record.get("cases")
    if isinstance(raw_cases, list):
        for item in raw_cases:
            if isinstance(item, Mapping):
                cases.append(_case_from_mapping(item))
    metrics = dict(record["metrics"]) if isinstance(record.get("metrics"), Mapping) else {}
    for key in HISTORY_METRIC_KEYS:
        number = metric_from_history_record(record, key)
        if number is not None:
            metrics.setdefault(key, number)
    hit = metric_from_history_record(record, "hit_rate")
    mrr = metric_from_history_record(record, "mrr")
    return EvalReport(
        hit_rate=hit if hit is not None else float(record.get("hit_rate") or 0.0),
        mrr=mrr if mrr is not None else float(record.get("mrr") or 0.0),
        case_count=int(record.get("case_count") or len(cases)),
        cases=cases,
        metrics=metrics,
    )


def _case_from_mapping(item: Mapping[str, Any]) -> EvalCaseResult:
    """还原单条 query 明细。"""
    metrics_raw = item.get("metrics") if isinstance(item.get("metrics"), Mapping) else {}
    metrics: dict[str, float] = {}
    if isinstance(metrics_raw, Mapping):
        for key, value in metrics_raw.items():
            number = _finite_number(value)
            if number is not None:
                metrics[str(key)] = number
    error = item.get("error")
    answer = item.get("generated_answer") or item.get("answer")
    return EvalCaseResult(
        query=str(item.get("query") or ""),
        retrieved_ids=[str(x) for x in item.get("retrieved_ids") or []],
        golden_ids=[str(x) for x in item.get("golden_ids") or []],
        expected_sources=[str(x) for x in item.get("expected_sources") or []],
        retrieved_sources=[str(x) for x in item.get("retrieved_sources") or []],
        metrics=metrics,
        error=str(error) if error else None,
        generated_answer=str(answer) if answer else None,
    )


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


def _render_report(report: EvalReport, *, title: str = "本次结果") -> None:
    """展示宏观指标与各 query 明细表。"""
    st.subheader(title)
    case_keys: set[str] = set()
    for item in report.cases:
        case_keys.update(item.metrics.keys())
    metric_items: list[tuple[str, float]] = []
    if "hit_rate" in case_keys:
        metric_items.append(("hit_rate", report.hit_rate))
    if "mrr" in case_keys:
        metric_items.append(("mrr", report.mrr))
    for key, value in sorted(report.metrics.items()):
        if key in {"hit_rate", "mrr"}:
            continue
        number = _finite_number(value)
        if number is None:
            continue
        metric_items.append((key, number))
    columns = st.columns(len(metric_items) or 1)
    for column, (name, value) in zip(columns, metric_items):
        column.metric(name, f"{value:.4f}")
    if not metric_items:
        st.warning("没有得到有效指标。若选了 Ragas，请查看明细表的 error 列或终端日志。")

    st.markdown("**各 query 明细**")
    if not report.cases:
        st.info("这条记录没有保存各 query 明细。重新运行评估后可点开查看。")
        return
    st.dataframe(_case_rows(report), width="stretch", hide_index=True)


def _case_rows(report: EvalReport) -> list[dict[str, Any]]:
    """把 EvalCaseResult 转成表格行；没有的指标不占列，避免整列 None。"""
    extra_keys: list[str] = []
    has_hit = False
    has_mrr = False
    for item in report.cases:
        if "hit_rate" in item.metrics:
            has_hit = True
        if "mrr" in item.metrics:
            has_mrr = True
        for key in item.metrics:
            if key not in {"hit_rate", "mrr"} and key not in extra_keys:
                extra_keys.append(key)
    rows: list[dict[str, Any]] = []
    for item in report.cases:
        row: dict[str, Any] = {"query": item.query}
        if has_hit:
            row["hit_rate"] = item.metrics.get("hit_rate")
        if has_mrr:
            row["mrr"] = item.metrics.get("mrr")
        for key in extra_keys:
            row[key] = item.metrics.get(key)
        row["retrieved"] = ", ".join(item.retrieved_ids)
        row["golden"] = ", ".join(item.golden_ids)
        row["answer"] = _truncate_text(item.generated_answer)
        row["error"] = item.error or ""
        rows.append(row)
    return rows


def _truncate_text(text: str | None, limit: int = 80) -> str:
    """明细表里压缩生成答案，避免撑爆列宽。"""
    if not text:
        return ""
    stripped = " ".join(str(text).split())
    if len(stripped) <= limit:
        return stripped
    return stripped[: limit - 1] + "…"


def _render_history(records: Sequence[Mapping[str, Any]]) -> None:
    """历史列表展示 Custom + Ragas 指标，并可选择一条查看明细。"""
    st.subheader("历史记录")
    if not records:
        st.caption("运行评估后将在此保存记录，可点开查看当时的各 query 明细。")
        return

    ordered = list(records)
    st.dataframe(_history_table_rows(ordered), width="stretch", hide_index=True)

    labels = ["（选择一条查看明细）"]
    for index, item in enumerate(ordered):
        when = str(item.get("ran_at") or "—")
        backend = str(item.get("backend") or "—")
        collection = str(item.get("collection") or "—")
        labels.append(f"{index + 1}. {when} · {backend} · {collection}")
    selected = st.selectbox("查看历史明细", list(range(len(labels))), format_func=lambda i: labels[i])
    if int(selected) > 0:
        picked = ordered[int(selected) - 1]
        _render_report(report_from_history_record(picked), title="历史详情")

    chart_keys = [
        key
        for key in HISTORY_METRIC_KEYS
        if any(metric_from_history_record(item, key) is not None for item in ordered)
    ]
    if len(ordered) >= 2 and chart_keys:
        st.subheader("历史趋势")
        chart = {
            key: [
                metric_from_history_record(item, key) or 0.0
                for item in ordered
            ]
            for key in chart_keys
        }
        st.line_chart(chart)
    elif len(ordered) < 2:
        st.caption("再运行一次评估后可对比历史趋势。")


def _history_table_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """历史表：时间/集合/后端 + Custom 与 Ragas 指标。"""
    rows: list[dict[str, Any]] = []
    for item in records:
        row: dict[str, Any] = {
            "时间": str(item.get("ran_at") or "—"),
            "集合": str(item.get("collection") or "—"),
            "后端": str(item.get("backend") or "—"),
            "条数": item.get("case_count") or "",
        }
        for key in HISTORY_METRIC_KEYS:
            number = metric_from_history_record(item, key)
            row[key] = f"{number:.4f}" if number is not None else "—"
        rows.append(row)
    return rows
