"""Dashboard 评估面板渲染测试：后端选择、运行结果与历史趋势。"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

from observability.dashboard.pages.evaluation_panel import (
    append_eval_history,
    backends_for_label,
    discover_eval_collections,
    load_eval_history,
)


@pytest.mark.unit
class TestEvaluationBackends:
    """验证面板后端选项与工厂 backends 的映射。"""

    def test_backends_for_label(self) -> None:
        """Custom / Ragas / All 应对应工厂组合列表。"""
        assert backends_for_label("Custom") == ["custom"]
        assert backends_for_label("Ragas") == ["ragas"]
        assert backends_for_label("All") == ["ragas", "custom"]


@pytest.mark.unit
class TestDiscoverEvalCollections:
    """验证评估面板集合列表：yaml 默认在前，并合并 Chroma 已有名称。"""

    def test_yaml_default_first_then_chroma_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """knowledge_hub 应排在最前，其余集合去重后跟上。"""

        class FakeService:
            def list_collections(self) -> list[str]:
                return ["col_b", "knowledge_hub", "col_a"]

        monkeypatch.setattr(
            "observability.dashboard.services.data_service.DataService.from_settings",
            classmethod(lambda cls, settings=None: FakeService()),
        )
        names = discover_eval_collections()
        assert names[0] == "knowledge_hub"
        assert "col_a" in names
        assert "col_b" in names
        assert names.count("knowledge_hub") == 1


@pytest.mark.unit
class TestEvalHistoryStore:
    """验证历史 JSONL 读写。"""

    def test_append_and_load_roundtrip(self, tmp_path: Path) -> None:
        """追加两条记录后应能按顺序读回。"""
        path = tmp_path / "eval_history.jsonl"
        append_eval_history(path, {"hit_rate": 0.2, "mrr": 0.1, "backend": "Custom"})
        append_eval_history(path, {"hit_rate": 0.8, "mrr": 0.7, "backend": "All"})
        records = load_eval_history(path)
        assert len(records) == 2
        assert records[0]["hit_rate"] == 0.2
        assert records[1]["backend"] == "All"


@pytest.mark.unit
class TestEvaluationPanelPage:
    """验证评估面板控件、运行结果与历史趋势。"""

    def test_renders_controls_without_placeholder(self) -> None:
        """应展示后端/测试集选择与运行按钮，不再显示阶段 H 占位提示。"""

        def page_script() -> None:
            # AppTest.from_function 抽取源码，须在函数内完成全部导入与路径
            from core.settings import REPO_ROOT
            from observability.dashboard.pages.evaluation_panel import render_evaluation_panel

            render_evaluation_panel(
                golden_sets=[REPO_ROOT / "tests" / "fixtures" / "golden_test_set.json"],
                history_records=[],
                collections=["knowledge_hub", "col_a"],
                load_deps=False,
            )

        app = AppTest.from_function(page_script, default_timeout=30)
        app.run()
        assert not app.exception
        labels = [item.label for item in app.selectbox]
        assert "集合" in labels
        assert "评估后端" in labels
        assert "Golden Test Set" in labels
        assert "运行评估" in [item.label for item in app.button]
        assert "生成黄金集" in [item.label for item in app.button]
        number_labels = [item.label for item in app.number_input]
        assert "生成条数" in number_labels
        infos = [str(item.value) for item in app.info]
        assert not any("评估模块尚未启用" in text for text in infos)

    def test_run_shows_metrics_and_query_details(self) -> None:
        """点击运行评估后应展示 hit_rate / mrr 与 query 明细。"""

        def page_script() -> None:
            from pathlib import Path
            import tempfile

            from core.settings import REPO_ROOT
            from observability.dashboard.pages.evaluation_panel import render_evaluation_panel
            from observability.evaluation.eval_runner import EvalCaseResult, EvalReport

            def fake_run(backend: str, test_set: str, collection: str) -> EvalReport:
                return EvalReport(
                    hit_rate=0.5,
                    mrr=0.5,
                    case_count=2,
                    metrics={"hit_rate": 0.5, "mrr": 0.5},
                    cases=[
                        EvalCaseResult(
                            query="如何配置 Azure OpenAI？",
                            retrieved_ids=["chunk_abc_001"],
                            golden_ids=["chunk_abc_001"],
                            expected_sources=["config_guide.pdf"],
                            retrieved_sources=["config_guide.pdf"],
                            metrics={"hit_rate": 1.0, "mrr": 1.0},
                        ),
                        EvalCaseResult(
                            query="未命中示例",
                            retrieved_ids=["other"],
                            golden_ids=["gold-2"],
                            expected_sources=[],
                            retrieved_sources=[],
                            metrics={"hit_rate": 0.0, "mrr": 0.0},
                        ),
                    ],
                )

            render_evaluation_panel(
                run_eval=fake_run,
                golden_sets=[REPO_ROOT / "tests" / "fixtures" / "golden_test_set.json"],
                history_path=Path(tempfile.gettempdir()) / "eval_panel_test_history.jsonl",
                history_records=[],
                collections=["col_a"],
                load_deps=False,
            )

        app = AppTest.from_function(page_script, default_timeout=30)
        app.run()
        assert not app.exception
        run = next(item for item in app.button if item.label == "运行评估")
        run.click().run()
        assert not app.exception
        metric_labels = [item.label for item in app.metric]
        assert "hit_rate" in metric_labels
        assert "mrr" in metric_labels
        successes = [str(item.value) for item in app.success]
        assert any("评估完成" in text for text in successes)
        assert any("集合 col_a" in text for text in successes)
        body = " ".join(str(item.value) for item in app.markdown)
        assert "各 query 明细" in body

    def test_run_uses_selected_collection(self) -> None:
        """下拉选中的集合应传入 run_eval。"""

        def page_script() -> None:
            from pathlib import Path
            import tempfile

            from core.settings import REPO_ROOT
            from observability.dashboard.pages.evaluation_panel import render_evaluation_panel
            from observability.evaluation.eval_runner import EvalReport

            def fake_run(backend: str, test_set: str, collection: str) -> EvalReport:
                return EvalReport(
                    hit_rate=0.0,
                    mrr=0.0,
                    case_count=0,
                    metrics={"hit_rate": 0.0, "mrr": 0.0},
                    cases=[],
                )

            render_evaluation_panel(
                run_eval=fake_run,
                golden_sets=[REPO_ROOT / "tests" / "fixtures" / "golden_test_set.json"],
                history_path=Path(tempfile.gettempdir()) / "eval_panel_collection_history.jsonl",
                history_records=[],
                collections=["col_a", "col_b"],
                load_deps=False,
            )

        app = AppTest.from_function(page_script, default_timeout=30)
        app.run()
        assert not app.exception
        picker = next(item for item in app.selectbox if item.label == "集合")
        picker.select("col_b").run()
        run = next(item for item in app.button if item.label == "运行评估")
        run.click().run()
        assert not app.exception
        successes = [str(item.value) for item in app.success]
        assert any("集合 col_b" in text for text in successes)

    def test_generate_adds_set_to_dropdown(self) -> None:
        """生成黄金集后应写入 JSON，并出现在 Golden Test Set 下拉框。"""

        def page_script() -> None:
            from pathlib import Path
            import tempfile

            from core.settings import REPO_ROOT
            from observability.dashboard.pages.evaluation_panel import render_evaluation_panel
            from observability.evaluation.golden_generator import write_golden_test_set

            out = Path(tempfile.gettempdir()) / "golden_gen_panel_ui.json"

            def fake_generate(collection: str, count: int) -> Path:
                write_golden_test_set(
                    [
                        {
                            "query": "生成的问题？",
                            "expected_chunk_ids": ["real-id-1"],
                            "expected_sources": ["doc.pdf"],
                        }
                    ],
                    out,
                    collection=collection,
                )
                return out

            render_evaluation_panel(
                golden_sets=[REPO_ROOT / "tests" / "fixtures" / "golden_test_set.json"],
                history_records=[],
                collections=["col_a"],
                generate_golden=fake_generate,
                load_deps=False,
            )

        app = AppTest.from_function(page_script, default_timeout=30)
        app.run()
        assert not app.exception
        gen = next(item for item in app.button if item.label == "生成黄金集")
        gen.click().run()
        assert not app.exception
        successes = [str(item.value) for item in app.success]
        assert any("已生成" in text and "golden_gen_panel_ui.json" in text for text in successes)
        golden_box = next(item for item in app.selectbox if item.label == "Golden Test Set")
        option_names = [golden_box.format_func(0), golden_box.format_func(1)]
        assert any("golden_gen_panel_ui.json" in str(name) for name in option_names)

    def test_history_trend_when_two_records(self) -> None:
        """注入两条历史时应展示历史趋势标题，而非再跑一次的提示。"""

        def page_script() -> None:
            from core.settings import REPO_ROOT
            from observability.dashboard.pages.evaluation_panel import render_evaluation_panel

            render_evaluation_panel(
                golden_sets=[REPO_ROOT / "tests" / "fixtures" / "golden_test_set.json"],
                history_records=[
                    {
                        "ran_at": "2026-09-01T00:00:00+00:00",
                        "backend": "Custom",
                        "hit_rate": 0.4,
                        "mrr": 0.3,
                    },
                    {
                        "ran_at": "2026-09-02T00:00:00+00:00",
                        "backend": "All",
                        "hit_rate": 0.7,
                        "mrr": 0.6,
                    },
                ],
                load_deps=False,
            )

        app = AppTest.from_function(page_script, default_timeout=30)
        app.run()
        assert not app.exception
        subheaders = [str(item.value) for item in app.subheader]
        assert any("历史趋势" in text for text in subheaders)
        captions = [str(item.value) for item in app.caption]
        assert not any("再运行一次评估后可对比历史趋势" in text for text in captions)
