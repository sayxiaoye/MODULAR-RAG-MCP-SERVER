"""JSON Lines logger 单元测试：JSONFormatter、write_trace、get_trace_logger。"""

from __future__ import annotations

import json
import logging

import pytest

from core.trace.trace_context import TraceContext
from observability.logger import (
    JSONFormatter,
    get_trace_logger,
    reset_trace_logger,
    write_trace,
)


@pytest.mark.unit
class TestJSONFormatter:
    """验证日志记录被格式化为单行合法 JSON。"""

    def test_format_produces_json_object(self) -> None:
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="modular_rag.trace",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        payload = json.loads(formatter.format(record))
        assert payload["message"] == "hello"
        assert payload["level"] == "INFO"
        assert payload["logger"] == "modular_rag.trace"


@pytest.mark.unit
class TestWriteTrace:
    """验证 write_trace 向 jsonl 追加一行且含 trace_type。"""

    def test_write_trace_appends_json_line_with_trace_type(self, tmp_path) -> None:
        """写入一条 query trace 后文件应新增一行合法 JSON。"""
        trace_file = tmp_path / "traces.jsonl"
        trace = TraceContext(trace_type="query")
        trace.record_stage("dense", elapsed_ms=1.5)
        trace.finish()

        write_trace(trace.to_dict(), path=trace_file)

        lines = trace_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        payload = json.loads(lines[0])
        assert payload["trace_type"] == "query"
        assert payload["trace_id"] == trace.trace_id
        assert payload["stages"][0]["name"] == "dense"

        write_trace({"trace_type": "ingestion", "trace_id": "second"}, path=trace_file)
        lines = trace_file.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert json.loads(lines[1])["trace_type"] == "ingestion"


@pytest.mark.unit
class TestGetTraceLogger:
    """验证 get_trace_logger 使用 JSONFormatter 写入文件。"""

    def test_get_trace_logger_writes_json_line(self, tmp_path) -> None:
        reset_trace_logger()
        try:
            log_file = tmp_path / "app.jsonl"
            logger = get_trace_logger(trace_file=log_file)
            logger.info("trace logger ready", extra={"trace_type": "query"})

            lines = [line for line in log_file.read_text(encoding="utf-8").splitlines() if line]
            assert len(lines) == 1
            payload = json.loads(lines[0])
            assert payload["message"] == "trace logger ready"
            assert payload["trace_type"] == "query"
        finally:
            reset_trace_logger()
