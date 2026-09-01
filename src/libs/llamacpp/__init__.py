"""llama.cpp 本地推理运行时：按需启动 / 用完释放 llama-server 进程。"""

from libs.llamacpp.process_manager import (
    LlamaCppLaunchConfig,
    LlamaCppProcessError,
    LlamaCppProcessManager,
    ROLE_EMBEDDING,
    ROLE_LLM,
    build_llama_server_command,
    get_process_manager,
    launch_config_from_embedding,
    launch_config_from_llm,
    reset_process_manager,
    run_with_server,
)

__all__ = [
    "LlamaCppLaunchConfig",
    "LlamaCppProcessError",
    "LlamaCppProcessManager",
    "ROLE_EMBEDDING",
    "ROLE_LLM",
    "build_llama_server_command",
    "get_process_manager",
    "launch_config_from_embedding",
    "launch_config_from_llm",
    "reset_process_manager",
    "run_with_server",
]
