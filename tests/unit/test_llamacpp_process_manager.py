"""llama-server 按需启停进程管理器的单元测试（不启动真实进程）。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.settings import EmbeddingSettings, LLMSettings
from libs.llamacpp.process_manager import (
    LlamaCppLaunchConfig,
    LlamaCppProcessError,
    LlamaCppProcessManager,
    ROLE_EMBEDDING,
    ROLE_LLM,
    build_llama_server_command,
    launch_config_from_embedding,
    launch_config_from_llm,
    reset_process_manager,
)


@pytest.fixture(autouse=True)
def _reset_manager() -> None:
    """每个用例前后丢弃单例，避免泄漏自管进程状态。"""
    reset_process_manager()
    yield
    reset_process_manager()


def _make_bin_and_model(tmp_path: Path) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    server_bin = tmp_path / "llama-server.exe"
    model_path = tmp_path / "model.gguf"
    server_bin.write_text("fake", encoding="utf-8")
    model_path.write_text("fake", encoding="utf-8")
    return server_bin, model_path


def _config(tmp_path: Path, role: str = ROLE_LLM, **overrides: object) -> LlamaCppLaunchConfig:
    server_bin, model_path = _make_bin_and_model(tmp_path)
    port = 8080 if role == ROLE_LLM else 8081
    payload: dict[str, object] = {
        "role": role,
        "server_bin": str(server_bin),
        "model_path": str(model_path),
        "host": "127.0.0.1",
        "port": port,
        "alias": "test-model",
        "extra_args": (),
        "idle_timeout": 0.0,
        "startup_timeout": 5.0,
        "exclusive_gpu": True,
        "health_url": f"http://127.0.0.1:{port}/v1/models",
    }
    payload.update(overrides)
    return LlamaCppLaunchConfig(**payload)  # type: ignore[arg-type]


class FakeProcess:
    """模拟 llama-server 子进程：默认未启动，popen 后存活，wait/kill 后退出。"""

    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid
        self.running = False

    def poll(self) -> int | None:
        return None if self.running else 0

    def wait(self, timeout: float | None = None) -> int:
        self.running = False
        return 0

    def terminate(self) -> None:
        self.running = False

    def kill(self) -> None:
        self.running = False


@pytest.mark.unit
class TestBuildCommand:
    """验证命令行拼装，尤其 Embedding 自动加 --embedding。"""

    def test_llm_command_includes_alias(self, tmp_path: Path) -> None:
        """LLM 命令应包含模型路径、端口与 alias。"""
        config = _config(tmp_path, ROLE_LLM)
        cmd = build_llama_server_command(config)
        assert config.server_bin in cmd
        assert "-m" in cmd
        assert config.model_path in cmd
        assert "--port" in cmd
        assert "8080" in cmd
        assert "--alias" in cmd
        assert "test-model" in cmd
        assert "--embedding" not in cmd

    def test_embedding_command_adds_embedding_flag(self, tmp_path: Path) -> None:
        """Embedding 角色即使 extra_args 为空也应带 --embedding。"""
        config = _config(tmp_path, ROLE_EMBEDDING)
        cmd = build_llama_server_command(config)
        assert "--embedding" in cmd
        assert "8081" in cmd


@pytest.mark.unit
class TestLaunchConfig:
    """验证从 Settings 推导启动参数与开关。"""

    def test_disabled_without_paths(self) -> None:
        """未配置路径时不应启用按需启停。"""
        llm = LLMSettings(provider="llamacpp", model="qwen", temperature=0.0, max_tokens=16)
        assert launch_config_from_llm(llm) is None
        emb = EmbeddingSettings(provider="llamacpp", model="bge", dimensions=4)
        assert launch_config_from_embedding(emb) is None

    def test_explicit_auto_manage_false(self, tmp_path: Path) -> None:
        """即使有路径，auto_manage=false 也应关闭按需启停。"""
        server_bin, model_path = _make_bin_and_model(tmp_path)
        llm = LLMSettings(
            provider="llamacpp",
            model="qwen",
            temperature=0.0,
            max_tokens=16,
            server_bin=str(server_bin),
            model_path=str(model_path),
            auto_manage=False,
        )
        assert launch_config_from_llm(llm) is None

    def test_auto_manage_true_requires_paths(self) -> None:
        """显式开启但缺少路径时应报错。"""
        llm = LLMSettings(
            provider="llamacpp",
            model="qwen",
            temperature=0.0,
            max_tokens=16,
            auto_manage=True,
        )
        with pytest.raises(LlamaCppProcessError, match="缺少配置"):
            launch_config_from_llm(llm)

    def test_from_llm_settings(self, tmp_path: Path) -> None:
        """有路径时应解析 host/port/alias。"""
        server_bin, model_path = _make_bin_and_model(tmp_path)
        llm = LLMSettings(
            provider="llamacpp",
            model="qwen2.5-7b-instruct",
            temperature=0.0,
            max_tokens=16,
            base_url="http://localhost:8080/v1",
            server_bin=str(server_bin),
            model_path=str(model_path),
            idle_timeout=0,
        )
        config = launch_config_from_llm(llm)
        assert config is not None
        assert config.role == ROLE_LLM
        assert config.host == "127.0.0.1"
        assert config.port == 8080
        assert config.alias == "qwen2.5-7b-instruct"
        assert config.idle_timeout == 0.0


@pytest.mark.unit
class TestProcessManagerLifecycle:
    """验证启动、复用外部进程、立即释放与 GPU 互斥。"""

    @patch("libs.llamacpp.process_manager.subprocess.run")
    def test_starts_then_stops_on_release(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """无外部服务时应 Popen 启动，idle_timeout=0 时 release 立即关闭。"""
        fake = FakeProcess()
        health_ok = {"value": False}

        def probe(_url: str, _timeout: float) -> bool:
            return health_ok["value"] and fake.running

        def sleep(_seconds: float) -> None:
            health_ok["value"] = True

        def popen(cmd: list[str], **kwargs: object) -> FakeProcess:
            fake.running = True
            health_ok["value"] = False
            return fake

        manager = LlamaCppProcessManager(
            popen=popen,
            health_probe=probe,
            sleep=sleep,
            port_probe=lambda host, port: False,
            log_dir=tmp_path,
        )
        config = _config(tmp_path, ROLE_LLM, idle_timeout=0.0)
        manager.ensure_ready(config)
        assert fake.poll() is None
        manager.release(config)
        assert fake.poll() == 0
        mock_run.assert_called()

    def test_reuses_healthy_external_server(self, tmp_path: Path) -> None:
        """端口已有健康服务时不应 Popen，release 也不杀外部进程。"""
        popen = MagicMock(side_effect=AssertionError("不应启动新进程"))
        manager = LlamaCppProcessManager(
            popen=popen,
            health_probe=lambda url, timeout: True,
            sleep=lambda s: None,
            port_probe=lambda host, port: True,
            log_dir=tmp_path,
        )
        config = _config(tmp_path, ROLE_LLM, idle_timeout=0.0)
        manager.ensure_ready(config)
        manager.release(config)
        popen.assert_not_called()

    @patch("libs.llamacpp.process_manager.subprocess.run")
    def test_exclusive_gpu_stops_other_role(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """启动 LLM 时应立刻关掉仍在运行的 Embedding 自管进程。"""
        processes = {"llm": FakeProcess(pid=1), "embedding": FakeProcess(pid=2)}
        started: list[str] = []
        health_ok = {"value": False}

        def popen(cmd: list[str], **kwargs: object) -> FakeProcess:
            role = ROLE_EMBEDDING if "--embedding" in cmd else ROLE_LLM
            started.append(role)
            proc = processes[role]
            proc.running = True
            health_ok["value"] = False
            return proc

        def probe(url: str, _timeout: float) -> bool:
            if not health_ok["value"]:
                return False
            role = ROLE_EMBEDDING if ":8081" in url else ROLE_LLM
            return processes[role].poll() is None

        def sleep(_seconds: float) -> None:
            health_ok["value"] = True

        manager = LlamaCppProcessManager(
            popen=popen,
            health_probe=probe,
            sleep=sleep,
            port_probe=lambda host, port: False,
            log_dir=tmp_path,
        )
        embed_cfg = _config(tmp_path / "e", ROLE_EMBEDDING, idle_timeout=0.0)
        # 为 embedding 单独准备可执行文件/模型，避免与 llm 共用被覆盖
        llm_cfg = _config(tmp_path / "l", ROLE_LLM, idle_timeout=0.0)

        manager.ensure_ready(embed_cfg)
        assert processes["embedding"].poll() is None
        manager.release(embed_cfg)
        # idle=0 会立刻关掉 embedding；再 ensure llm 验证互斥路径：先重新拉起 embedding 但不 release
        manager.ensure_ready(embed_cfg)
        manager.ensure_ready(llm_cfg)
        assert processes["embedding"].poll() == 0
        assert processes["llm"].poll() is None
        assert ROLE_LLM in started
        manager.release(llm_cfg)


@pytest.mark.unit
class TestLlamaCppProvidersUseManager:
    """验证 LLM / Embedding 在配置了路径时走进程管理包装。"""

    @patch("httpx.post")
    @patch("libs.llm.llamacpp_llm.run_with_server")
    def test_llm_chat_uses_run_with_server(
        self,
        mock_run: MagicMock,
        mock_post: MagicMock,
        tmp_path: Path,
    ) -> None:
        """配置了 server_bin 时 chat 应通过 run_with_server 包装。"""
        from libs.llm.llamacpp_llm import LlamaCppLLM

        mock_run.side_effect = lambda config, fn: fn()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": "ok"}}],
            "model": "qwen",
            "usage": {},
        }
        mock_post.return_value = mock_resp
        server_bin, model_path = _make_bin_and_model(tmp_path)
        llm = LlamaCppLLM(
            LLMSettings(
                provider="llamacpp",
                model="qwen",
                temperature=0.0,
                max_tokens=16,
                server_bin=str(server_bin),
                model_path=str(model_path),
            )
        )
        from libs.llm.base_llm import ChatMessage

        result = llm.chat([ChatMessage(role="user", content="hi")])
        assert result.content == "ok"
        mock_run.assert_called_once()
        launched = mock_run.call_args.args[0]
        assert launched is not None
        assert launched.role == ROLE_LLM

    @patch("httpx.post")
    @patch("libs.embedding.llamacpp_embedding.run_with_server")
    def test_embedding_uses_run_with_server(
        self,
        mock_run: MagicMock,
        mock_post: MagicMock,
        tmp_path: Path,
    ) -> None:
        """配置了路径时 embed 应通过 run_with_server 包装。"""
        from libs.embedding.llamacpp_embedding import LlamaCppEmbedding

        mock_run.side_effect = lambda config, fn: fn()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [{"index": 0, "embedding": [0.1, 0.2]}],
        }
        mock_post.return_value = mock_resp
        server_bin, model_path = _make_bin_and_model(tmp_path)
        emb = LlamaCppEmbedding(
            EmbeddingSettings(
                provider="llamacpp",
                model="bge",
                dimensions=2,
                server_bin=str(server_bin),
                model_path=str(model_path),
            )
        )
        vectors = emb.embed(["hello"])
        assert vectors == [[0.1, 0.2]]
        mock_run.assert_called_once()
        launched = mock_run.call_args.args[0]
        assert launched is not None
        assert launched.role == ROLE_EMBEDDING
