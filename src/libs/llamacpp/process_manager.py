"""llama-server 进程生命周期：按需启动、角色互斥、空闲后关闭以释放 GPU。

仅管理本进程拉起的实例。若端口上已有健康的外部 llama-server，则复用且不接管，
避免误杀用户手动启动的服务。
"""

from __future__ import annotations

import atexit
import logging
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Callable, Optional
from urllib.parse import urlparse

import httpx

from core.settings import REPO_ROOT, EmbeddingSettings, LLMSettings

logger = logging.getLogger("modular_rag.llamacpp")

ROLE_LLM = "llm"
ROLE_EMBEDDING = "embedding"

DEFAULT_IDLE_TIMEOUT = 8.0
DEFAULT_STARTUP_TIMEOUT = 180.0
DEFAULT_LLM_PORT = 8080
DEFAULT_EMBED_PORT = 8081
DEFAULT_LLM_BASE_URL = "http://localhost:8080/v1"
DEFAULT_EMBED_BASE_URL = "http://localhost:8081/v1"

# Windows：新进程组 + 不弹出控制台窗口
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_CREATE_NO_WINDOW = 0x08000000


class LlamaCppProcessError(RuntimeError):
    """启动、等待就绪或关闭 llama-server 失败。"""


@dataclass(frozen=True)
class LlamaCppLaunchConfig:
    """一次 llama-server 启动所需的全部参数。"""

    role: str
    server_bin: str
    model_path: str
    host: str
    port: int
    alias: str
    extra_args: tuple[str, ...]
    idle_timeout: float
    startup_timeout: float
    exclusive_gpu: bool
    health_url: str


@dataclass
class _Slot:
    """单个角色的运行时槽位。"""

    config: LlamaCppLaunchConfig
    process: subprocess.Popen | None = None
    log_handle: IO[str] | None = None
    owned: bool = False
    refcount: int = 0
    idle_timer: threading.Timer | None = None


def parse_host_port(base_url: str, default_port: int) -> tuple[str, int]:
    """从 OpenAI 兼容 base_url 解析监听地址；localhost 统一绑到 127.0.0.1。"""
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    if host == "localhost":
        host = "127.0.0.1"
    port = parsed.port if parsed.port else default_port
    return host, port


def health_url_from_base(base_url: str) -> str:
    """llama-server OpenAI 兼容 /v1/models 探测地址。"""
    return f"{base_url.rstrip('/')}/models"


def build_llama_server_command(config: LlamaCppLaunchConfig) -> list[str]:
    """组装 llama-server 命令行；Embedding 角色自动补 ``--embedding``。"""
    cmd = [
        config.server_bin,
        "-m",
        config.model_path,
        "--host",
        config.host,
        "--port",
        str(config.port),
    ]
    extra = list(config.extra_args)
    extra_lower = {item.lower() for item in extra}
    if config.alias and "--alias" not in extra_lower:
        cmd.extend(["--alias", config.alias])
    if config.role == ROLE_EMBEDDING and "--embedding" not in extra_lower:
        cmd.append("--embedding")
    cmd.extend(extra)
    return cmd


def _should_auto_manage(auto_manage: Optional[bool], server_bin: Optional[str], model_path: Optional[str]) -> bool:
    """auto_manage 显式 False 关闭；显式 True 开启；未配置时有路径即开启。"""
    if auto_manage is False:
        return False
    if auto_manage is True:
        return True
    return bool(server_bin and model_path)


def _require_auto_manage_paths(role: str, server_bin: Optional[str], model_path: Optional[str]) -> None:
    """auto_manage 已启用时必须同时提供可执行文件与 GGUF 路径。"""
    missing: list[str] = []
    if not server_bin:
        missing.append("server_bin")
    if not model_path:
        missing.append("model_path")
    if missing:
        raise LlamaCppProcessError(
            f"[llamacpp] 已启用按需启停，但 {role} 缺少配置: {', '.join(missing)}。"
            "请在 settings.yaml 的 llamacpp 或对应角色块中填写。"
        )


def launch_config_from_llm(settings: LLMSettings) -> LlamaCppLaunchConfig | None:
    """从 LLM 配置构造启动参数；未启用按需启停时返回 None。"""
    if settings.provider.strip().lower() != "llamacpp":
        return None
    if not _should_auto_manage(settings.auto_manage, settings.server_bin, settings.model_path):
        return None
    _require_auto_manage_paths(ROLE_LLM, settings.server_bin, settings.model_path)
    base_url = settings.base_url or DEFAULT_LLM_BASE_URL
    host, port = parse_host_port(base_url, DEFAULT_LLM_PORT)
    return LlamaCppLaunchConfig(
        role=ROLE_LLM,
        server_bin=str(settings.server_bin),
        model_path=str(settings.model_path),
        host=host,
        port=port,
        alias=settings.model,
        extra_args=tuple(settings.extra_args or ()),
        idle_timeout=_resolve_timeout(settings.idle_timeout, DEFAULT_IDLE_TIMEOUT),
        startup_timeout=_resolve_timeout(settings.startup_timeout, DEFAULT_STARTUP_TIMEOUT),
        exclusive_gpu=True if settings.exclusive_gpu is None else bool(settings.exclusive_gpu),
        health_url=health_url_from_base(base_url),
    )


def launch_config_from_embedding(settings: EmbeddingSettings) -> LlamaCppLaunchConfig | None:
    """从 Embedding 配置构造启动参数；未启用按需启停时返回 None。"""
    if settings.provider.strip().lower() != "llamacpp":
        return None
    if not _should_auto_manage(settings.auto_manage, settings.server_bin, settings.model_path):
        return None
    _require_auto_manage_paths(ROLE_EMBEDDING, settings.server_bin, settings.model_path)
    base_url = settings.base_url or DEFAULT_EMBED_BASE_URL
    host, port = parse_host_port(base_url, DEFAULT_EMBED_PORT)
    return LlamaCppLaunchConfig(
        role=ROLE_EMBEDDING,
        server_bin=str(settings.server_bin),
        model_path=str(settings.model_path),
        host=host,
        port=port,
        alias=settings.model,
        extra_args=tuple(settings.extra_args or ()),
        idle_timeout=_resolve_timeout(settings.idle_timeout, DEFAULT_IDLE_TIMEOUT),
        startup_timeout=_resolve_timeout(settings.startup_timeout, DEFAULT_STARTUP_TIMEOUT),
        exclusive_gpu=True if settings.exclusive_gpu is None else bool(settings.exclusive_gpu),
        health_url=health_url_from_base(base_url),
    )


def _resolve_timeout(value: Optional[float], default: float) -> float:
    """None 用默认值；允许 0 表示立即关闭。"""
    if value is None:
        return default
    if value < 0:
        raise LlamaCppProcessError("[llamacpp] idle_timeout / startup_timeout 不能为负数")
    return float(value)


def probe_health(url: str, timeout: float = 1.5) -> bool:
    """探测 llama-server 是否已可接受 OpenAI 兼容请求。"""
    try:
        response = httpx.get(url, timeout=timeout)
    except httpx.HTTPError:
        return False
    return response.status_code < 500


def _port_open(host: str, port: int) -> bool:
    """TCP 端口是否已被占用（可能仍在加载模型、尚未通过 health）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) == 0


class LlamaCppProcessManager:
    """进程内单例式管理器：同一时刻默认只保留一个自管 llama-server（节省显存）。"""

    def __init__(
        self,
        *,
        popen: Callable[..., subprocess.Popen] = subprocess.Popen,
        health_probe: Callable[[str, float], bool] = probe_health,
        sleep: Callable[[float], None] = time.sleep,
        port_probe: Callable[[str, int], bool] = _port_open,
        log_dir: Path | None = None,
    ) -> None:
        self._popen = popen
        self._health_probe = health_probe
        self._sleep = sleep
        self._port_probe = port_probe
        self._log_dir = log_dir or (REPO_ROOT / "logs")
        self._lock = threading.RLock()
        self._slots: dict[str, _Slot] = {}
        self._atexit_registered = False

    def ensure_ready(self, config: LlamaCppLaunchConfig) -> None:
        """确保该角色的 llama-server 可服务；必要时先关掉另一角色以释放 GPU。"""
        self._register_atexit()
        with self._lock:
            if config.exclusive_gpu:
                self._stop_other_roles(config.role)
            slot = self._slots.get(config.role)
            if slot is not None:
                self._cancel_idle_timer(slot)
            if self._is_slot_ready(config):
                slot = self._slots.setdefault(config.role, _Slot(config=config))
                slot.config = config
                slot.refcount += 1
                return
            self._start_owned(config)

    def release(self, config: LlamaCppLaunchConfig) -> None:
        """一次推理结束：引用计数归零后按 idle_timeout 关闭自管进程。"""
        with self._lock:
            slot = self._slots.get(config.role)
            if slot is None:
                return
            slot.refcount = max(0, slot.refcount - 1)
            if slot.refcount > 0:
                return
            if not slot.owned:
                return
            timeout = slot.config.idle_timeout
            if timeout <= 0:
                self._stop_slot(config.role, reason="调用结束立即释放 GPU")
                return
            self._schedule_idle_stop(slot)

    def shutdown_all(self) -> None:
        """关闭全部自管进程（进程退出或测试复位时调用）。"""
        with self._lock:
            for role in list(self._slots):
                self._stop_slot(role, reason="进程退出，释放 GPU")

    def _is_slot_ready(self, config: LlamaCppLaunchConfig) -> bool:
        """已托管且进程仍活，或端口上已有健康外部服务。"""
        slot = self._slots.get(config.role)
        if slot is not None and slot.owned and slot.process is not None:
            if slot.process.poll() is None:
                return True
            logger.warning("[llamacpp] 自管 %s 进程已退出，将重新拉起", config.role)
            self._stop_slot(config.role, reason="进程异常，准备重启")
        if self._health_probe(config.health_url, 1.5):
            # 外部已启动：复用但不接管，release 时不会杀掉
            reused = self._slots.setdefault(config.role, _Slot(config=config))
            reused.config = config
            reused.owned = False
            reused.process = None
            logger.info("[llamacpp] 复用已在运行的 %s llama-server（%s:%s）", config.role, config.host, config.port)
            return True
        return False

    def _start_owned(self, config: LlamaCppLaunchConfig) -> None:
        """拉起新的 llama-server 并等待 /v1/models 就绪。"""
        if not Path(config.server_bin).is_file():
            raise LlamaCppProcessError(
                f"[llamacpp] 找不到 llama-server 可执行文件: {config.server_bin}"
            )
        if not Path(config.model_path).is_file():
            raise LlamaCppProcessError(
                f"[llamacpp] 找不到 GGUF 模型文件: {config.model_path}"
            )
        if self._port_probe(config.host, config.port) and not self._health_probe(config.health_url, 1.5):
            # 端口被占但尚未健康：可能是刚启动，继续等待而不是再起一个
            logger.info(
                "[llamacpp] %s 端口 %s 已被占用，等待现有进程就绪",
                config.role,
                config.port,
            )
            self._wait_until_healthy(config, owned=False)
            slot = self._slots.setdefault(config.role, _Slot(config=config))
            slot.config = config
            slot.owned = False
            slot.refcount += 1
            return

        cmd = build_llama_server_command(config)
        log_path = self._prepare_log_path(config.role)
        logger.info("[llamacpp] 启动 %s llama-server: %s", config.role, " ".join(cmd))
        log_handle = log_path.open("w", encoding="utf-8")
        try:
            popen_kwargs: dict[str, Any] = {
                "stdout": log_handle,
                "stderr": subprocess.STDOUT,
            }
            if sys.platform == "win32":
                popen_kwargs["creationflags"] = _CREATE_NEW_PROCESS_GROUP | _CREATE_NO_WINDOW
            process = self._popen(cmd, **popen_kwargs)
        except OSError as exc:
            log_handle.close()
            raise LlamaCppProcessError(
                f"[llamacpp] 无法启动 llama-server ({config.role}): {exc}"
            ) from exc

        slot = _Slot(config=config, process=process, log_handle=log_handle, owned=True, refcount=1)
        self._slots[config.role] = slot
        try:
            self._wait_until_healthy(config, owned=True)
        except Exception:
            self._stop_slot(config.role, reason="启动失败，清理进程")
            raise
        logger.info("[llamacpp] %s llama-server 已就绪（%s:%s）", config.role, config.host, config.port)

    def _wait_until_healthy(self, config: LlamaCppLaunchConfig, *, owned: bool) -> None:
        """轮询 health，直到就绪、进程退出或超时。"""
        deadline = time.monotonic() + config.startup_timeout
        while time.monotonic() < deadline:
            if owned:
                slot = self._slots.get(config.role)
                if slot is not None and slot.process is not None and slot.process.poll() is not None:
                    tail = self._read_log_tail(config.role)
                    raise LlamaCppProcessError(
                        f"[llamacpp] llama-server ({config.role}) 启动后立即退出。"
                        f" 日志片段:\n{tail}"
                    )
            if self._health_probe(config.health_url, 1.5):
                return
            self._sleep(0.5)
        tail = self._read_log_tail(config.role) if owned else "(外部进程，无本系统日志)"
        raise LlamaCppProcessError(
            f"[llamacpp] 等待 {config.role} llama-server 就绪超时"
            f"（{config.startup_timeout:.0f}s，{config.health_url}）。日志片段:\n{tail}"
        )

    def _stop_other_roles(self, keep_role: str) -> None:
        """GPU 互斥：立刻关掉其它自管角色，给即将加载的模型腾显存。"""
        for role in list(self._slots):
            if role != keep_role:
                self._stop_slot(role, reason=f"切换到 {keep_role}，释放 GPU")

    def _schedule_idle_stop(self, slot: _Slot) -> None:
        """空闲计时：期间若再次 ensure_ready 会取消。"""
        self._cancel_idle_timer(slot)
        role = slot.config.role
        timeout = slot.config.idle_timeout

        def _fire() -> None:
            with self._lock:
                current = self._slots.get(role)
                if current is None or current.refcount > 0 or not current.owned:
                    return
                self._stop_slot(role, reason=f"空闲 {timeout:.0f}s，释放 GPU")

        timer = threading.Timer(timeout, _fire)
        timer.daemon = True
        slot.idle_timer = timer
        timer.start()
        logger.info("[llamacpp] %s 将在空闲 %.1fs 后关闭", role, timeout)

    def _cancel_idle_timer(self, slot: _Slot) -> None:
        if slot.idle_timer is not None:
            slot.idle_timer.cancel()
            slot.idle_timer = None

    def _stop_slot(self, role: str, *, reason: str) -> None:
        slot = self._slots.pop(role, None)
        if slot is None:
            return
        self._cancel_idle_timer(slot)
        if not slot.owned or slot.process is None:
            logger.info("[llamacpp] 跳过关闭 %s（非本进程启动）: %s", role, reason)
            return
        logger.info("[llamacpp] 关闭 %s llama-server: %s", role, reason)
        self._terminate_process(slot.process)
        if slot.log_handle is not None:
            try:
                slot.log_handle.close()
            except OSError:
                pass

    def _terminate_process(self, process: subprocess.Popen) -> None:
        """结束进程树。Windows 用 taskkill /T，避免残留占用 GPU 的子进程。"""
        if process.poll() is not None:
            return
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True,
                check=False,
            )
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
            return
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()

    def _prepare_log_path(self, role: str) -> Path:
        self._log_dir.mkdir(parents=True, exist_ok=True)
        return self._log_dir / f"llamacpp-{role}.log"

    def _read_log_tail(self, role: str, lines: int = 40) -> str:
        path = self._log_dir / f"llamacpp-{role}.log"
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "(无法读取日志)"
        tail = text.strip().splitlines()[-lines:]
        return "\n".join(tail) if tail else "(日志为空)"

    def _register_atexit(self) -> None:
        if self._atexit_registered:
            return
        atexit.register(self.shutdown_all)
        self._atexit_registered = True


_MANAGER: LlamaCppProcessManager | None = None
_MANAGER_LOCK = threading.Lock()


def get_process_manager() -> LlamaCppProcessManager:
    """获取进程内共享的 ProcessManager 单例。"""
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = LlamaCppProcessManager()
        return _MANAGER


def reset_process_manager() -> None:
    """测试用：关掉自管进程并丢弃单例。"""
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is not None:
            _MANAGER.shutdown_all()
        _MANAGER = None


def run_with_server(config: LlamaCppLaunchConfig | None, fn: Callable[[], Any]) -> Any:
    """若启用按需启停则在调用前后管理 llama-server；否则直接执行 *fn*。"""
    if config is None:
        return fn()
    manager = get_process_manager()
    manager.ensure_ready(config)
    try:
        return fn()
    finally:
        manager.release(config)
