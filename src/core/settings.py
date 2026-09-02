"""配置加载与校验模块。

职责：
- 从 ``config/settings.yaml`` 读取 YAML 配置
- 解析为强类型 dataclass（``Settings`` 及子配置块）
- 校验必填字段，失败时抛出 :class:`SettingsError`

全项目通过 :func:`load_settings` 获取配置，避免在业务代码中硬编码路径或参数。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml

# ---------------------------------------------------------------------------
# Repo root & path resolution
# ---------------------------------------------------------------------------
# 以本文件为锚点：<repo>/src/core/settings.py → parents[2] 即仓库根
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Default absolute path to settings.yaml
DEFAULT_SETTINGS_PATH: Path = REPO_ROOT / "config" / "settings.yaml"


def resolve_path(relative: Union[str, Path]) -> Path:
    """将仓库相对路径解析为绝对路径。

    若 *relative* 已是绝对路径则原样返回；否则基于 :data:`REPO_ROOT` 拼接并 resolve。

    Args:
        relative: 相对或绝对路径字符串/Path。

    Returns:
        解析后的绝对路径。

    >>> resolve_path("config/settings.yaml")  # doctest: +SKIP
    PosixPath('/home/user/Modular-RAG-MCP-Server/config/settings.yaml')
    """
    p = Path(relative)
    if p.is_absolute():
        return p
    return (REPO_ROOT / p).resolve()


class SettingsError(ValueError):
    """配置缺失、类型不符或校验失败时抛出。"""


def _require_mapping(data: Dict[str, Any], key: str, path: str) -> Dict[str, Any]:
    """要求字段存在且为 dict（YAML 中的嵌套配置块）。"""
    value = data.get(key)
    if value is None:
        raise SettingsError(f"Missing required field: {path}.{key}")
    if not isinstance(value, dict):
        raise SettingsError(f"Expected mapping for field: {path}.{key}")
    return value


def _require_value(data: Dict[str, Any], key: str, path: str) -> Any:
    """要求字段存在且非 None。"""
    if key not in data or data.get(key) is None:
        raise SettingsError(f"Missing required field: {path}.{key}")
    return data[key]


def _require_str(data: Dict[str, Any], key: str, path: str) -> str:
    """要求非空字符串。"""
    value = _require_value(data, key, path)
    if not isinstance(value, str) or not value.strip():
        raise SettingsError(f"Expected non-empty string for field: {path}.{key}")
    return value


def _require_int(data: Dict[str, Any], key: str, path: str) -> int:
    """要求整数类型。"""
    value = _require_value(data, key, path)
    if not isinstance(value, int):
        raise SettingsError(f"Expected integer for field: {path}.{key}")
    return value


def _require_number(data: Dict[str, Any], key: str, path: str) -> float:
    """要求数值类型（int/float），统一返回 float。"""
    value = _require_value(data, key, path)
    if not isinstance(value, (int, float)):
        raise SettingsError(f"Expected number for field: {path}.{key}")
    return float(value)


def _require_bool(data: Dict[str, Any], key: str, path: str) -> bool:
    """要求布尔类型（YAML 的 true/false）。"""
    value = _require_value(data, key, path)
    if not isinstance(value, bool):
        raise SettingsError(f"Expected boolean for field: {path}.{key}")
    return value


def _require_list(data: Dict[str, Any], key: str, path: str) -> List[Any]:
    """要求列表类型。"""
    value = _require_value(data, key, path)
    if not isinstance(value, list):
        raise SettingsError(f"Expected list for field: {path}.{key}")
    return value


def _optional_str(data: Dict[str, Any], key: str, path: str) -> Optional[str]:
    """可选非空字符串；缺省或空串视为 None。"""
    if key not in data or data.get(key) is None:
        return None
    value = data[key]
    if not isinstance(value, str):
        raise SettingsError(f"Expected string for field: {path}.{key}")
    stripped = value.strip()
    return stripped or None


def _optional_bool(data: Dict[str, Any], key: str, path: str) -> Optional[bool]:
    """可选布尔；缺省返回 None，便于「未配置则按路径推断」。"""
    if key not in data or data.get(key) is None:
        return None
    value = data[key]
    if not isinstance(value, bool):
        raise SettingsError(f"Expected boolean for field: {path}.{key}")
    return value


def _optional_number(data: Dict[str, Any], key: str, path: str) -> Optional[float]:
    """可选数值；缺省返回 None。"""
    if key not in data or data.get(key) is None:
        return None
    value = data[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SettingsError(f"Expected number for field: {path}.{key}")
    return float(value)


def _optional_str_tuple(data: Dict[str, Any], key: str, path: str) -> tuple[str, ...]:
    """可选字符串列表，解析为不可变 tuple；缺省为空 tuple。"""
    if key not in data or data.get(key) is None:
        return ()
    value = data[key]
    if not isinstance(value, list):
        raise SettingsError(f"Expected list for field: {path}.{key}")
    return tuple(str(item) for item in value)


def _optional_str_list(data: Dict[str, Any], key: str, path: str) -> Optional[List[str]]:
    """可选字符串列表；缺省返回 None，便于「未配置则走 provider」。"""
    if key not in data or data.get(key) is None:
        return None
    value = data[key]
    if not isinstance(value, list):
        raise SettingsError(f"Expected list for field: {path}.{key}")
    return [str(item) for item in value]


def _llamacpp_runtime_block(data: Dict[str, Any]) -> Dict[str, Any]:
    """读取可选的顶层 ``llamacpp`` 块，供 LLM/Embedding 共享 server_bin 等字段。"""
    raw = data.get("llamacpp")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise SettingsError("Expected mapping for field: settings.llamacpp")
    return raw


def _merge_llamacpp_fields(
    block: Dict[str, Any],
    runtime: Dict[str, Any],
    path: str,
) -> Dict[str, Any]:
    """将角色配置与共享 ``llamacpp`` 块合并；角色字段优先。"""
    merged = dict(runtime)
    for key, value in block.items():
        if value is not None:
            merged[key] = value
    return {
        "server_bin": _optional_str(merged, "server_bin", path),
        "model_path": _optional_str(merged, "model_path", path),
        "extra_args": _optional_str_tuple(merged, "extra_args", path),
        "auto_manage": _optional_bool(merged, "auto_manage", path),
        "idle_timeout": _optional_number(merged, "idle_timeout", path),
        "startup_timeout": _optional_number(merged, "startup_timeout", path),
        "exclusive_gpu": _optional_bool(merged, "exclusive_gpu", path),
    }


@dataclass(frozen=True)
class LLMSettings:
    """大语言模型（LLM）配置，对应 settings.yaml 的 ``llm`` 块。"""

    provider: str
    model: str
    temperature: float
    max_tokens: int
    # Azure/OpenAI-specific optional fields
    api_key: Optional[str] = None
    api_version: Optional[str] = None
    azure_endpoint: Optional[str] = None
    deployment_name: Optional[str] = None
    # Ollama / LlamaCpp HTTP 端点
    base_url: Optional[str] = None
    # llama-server 按需启停（仅 provider=llamacpp 时使用）
    server_bin: Optional[str] = None
    model_path: Optional[str] = None
    extra_args: tuple[str, ...] = ()
    auto_manage: Optional[bool] = None
    idle_timeout: Optional[float] = None
    startup_timeout: Optional[float] = None
    exclusive_gpu: Optional[bool] = None


@dataclass(frozen=True)
class EmbeddingSettings:
    """文本向量化（Embedding）配置，对应 ``embedding`` 块。"""

    provider: str
    model: str
    dimensions: int
    # Azure-specific optional fields
    api_key: Optional[str] = None
    api_version: Optional[str] = None
    azure_endpoint: Optional[str] = None
    deployment_name: Optional[str] = None
    # Ollama / LlamaCpp HTTP 端点
    base_url: Optional[str] = None
    # llama-server 按需启停（仅 provider=llamacpp 时使用）
    server_bin: Optional[str] = None
    model_path: Optional[str] = None
    extra_args: tuple[str, ...] = ()
    auto_manage: Optional[bool] = None
    idle_timeout: Optional[float] = None
    startup_timeout: Optional[float] = None
    exclusive_gpu: Optional[bool] = None


@dataclass(frozen=True)
class VectorStoreSettings:
    """向量库配置，对应 ``vector_store`` 块（如 Chroma 持久化目录与集合名）。"""

    provider: str
    persist_directory: str
    collection_name: str


@dataclass(frozen=True)
class RetrievalSettings:
    """检索与融合参数，对应 ``retrieval`` 块（Dense/Sparse top-k 与 RRF）。"""

    dense_top_k: int
    sparse_top_k: int
    fusion_top_k: int
    rrf_k: int  # Reciprocal Rank Fusion 常数


@dataclass(frozen=True)
class RerankSettings:
    """重排序模型配置，对应 ``rerank`` 块。"""

    enabled: bool
    provider: str
    model: str
    top_k: int


@dataclass(frozen=True)
class EvaluationSettings:
    """RAG 评测配置，对应 ``evaluation`` 块。

    ``backends`` 为可选项：配置两项及以上时，工厂会组合成 CompositeEvaluator。
    未配置则仍按 ``provider`` 创建单一评估器，保持向后兼容。
    """

    enabled: bool
    provider: str
    metrics: List[str]
    backends: Optional[List[str]] = None


@dataclass(frozen=True)
class ObservabilitySettings:
    """可观测性配置：日志级别、链路追踪与结构化日志，对应 ``observability`` 块。"""

    log_level: str
    trace_enabled: bool
    trace_file: str
    structured_logging: bool


@dataclass(frozen=True)
class VisionLLMSettings:
    """多模态视觉 LLM 配置（图表/截图理解），对应可选的 ``vision_llm`` 块。"""

    enabled: bool
    provider: str
    model: str
    max_image_size: int
    api_key: Optional[str] = None
    api_version: Optional[str] = None
    azure_endpoint: Optional[str] = None
    deployment_name: Optional[str] = None
    base_url: Optional[str] = None


@dataclass(frozen=True)
class IngestionSettings:
    """文档入库与分块配置，对应可选的 ``ingestion`` 块。"""

    chunk_size: int
    chunk_overlap: int
    splitter: str
    batch_size: int
    chunk_refiner: Optional[Dict[str, Any]] = None  # 可选：Chunk 精炼子配置
    metadata_enricher: Optional[Dict[str, Any]] = None  # 可选：元数据增强子配置


@dataclass(frozen=True)
class Settings:
    """应用全局配置根对象，聚合各子配置块。

    必填块：llm、embedding、vector_store、retrieval、rerank、evaluation、observability。
    可选块：ingestion、vision_llm。
    """

    llm: LLMSettings
    embedding: EmbeddingSettings
    vector_store: VectorStoreSettings
    retrieval: RetrievalSettings
    rerank: RerankSettings
    evaluation: EvaluationSettings
    observability: ObservabilitySettings
    ingestion: Optional[IngestionSettings] = None
    vision_llm: Optional[VisionLLMSettings] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Settings":
        """将 YAML 解析后的 dict 转为强类型 Settings。

        先校验各必填嵌套块存在，再逐字段类型检查；可选块缺失时对应字段为 None。

        Args:
            data: ``yaml.safe_load`` 得到的根 mapping。

        Returns:
            不可变的 Settings 实例。

        Raises:
            SettingsError: 根节点非 dict、缺少必填块或字段类型不符。
        """
        if not isinstance(data, dict):
            raise SettingsError("Settings root must be a mapping")

        # 提取各必填配置块
        llm = _require_mapping(data, "llm", "settings")
        embedding = _require_mapping(data, "embedding", "settings")
        vector_store = _require_mapping(data, "vector_store", "settings")
        retrieval = _require_mapping(data, "retrieval", "settings")
        rerank = _require_mapping(data, "rerank", "settings")
        evaluation = _require_mapping(data, "evaluation", "settings")
        observability = _require_mapping(data, "observability", "settings")
        # 可选：llama-server 共享运行时（server_bin / idle_timeout 等），合并进 llm/embedding
        llamacpp_runtime = _llamacpp_runtime_block(data)
        llm_llamacpp = _merge_llamacpp_fields(llm, llamacpp_runtime, "llm")
        embedding_llamacpp = _merge_llamacpp_fields(embedding, llamacpp_runtime, "embedding")

        # 可选：文档入库配置
        ingestion_settings = None
        if "ingestion" in data:
            ingestion = _require_mapping(data, "ingestion", "settings")
            ingestion_settings = IngestionSettings(
                chunk_size=_require_int(ingestion, "chunk_size", "ingestion"),
                chunk_overlap=_require_int(ingestion, "chunk_overlap", "ingestion"),
                splitter=_require_str(ingestion, "splitter", "ingestion"),
                batch_size=_require_int(ingestion, "batch_size", "ingestion"),
                chunk_refiner=ingestion.get("chunk_refiner"),
                metadata_enricher=ingestion.get("metadata_enricher"),
            )

        # 可选：视觉多模态 LLM 配置
        vision_llm_settings = None
        if "vision_llm" in data:
            vision_llm = _require_mapping(data, "vision_llm", "settings")
            vision_llm_settings = VisionLLMSettings(
                enabled=_require_bool(vision_llm, "enabled", "vision_llm"),
                provider=_require_str(vision_llm, "provider", "vision_llm"),
                model=_require_str(vision_llm, "model", "vision_llm"),
                max_image_size=_require_int(vision_llm, "max_image_size", "vision_llm"),
                api_key=vision_llm.get("api_key"),
                api_version=vision_llm.get("api_version"),
                azure_endpoint=vision_llm.get("azure_endpoint"),
                deployment_name=vision_llm.get("deployment_name"),
                base_url=vision_llm.get("base_url"),
            )

        settings = cls(
            llm=LLMSettings(
                provider=_require_str(llm, "provider", "llm"),
                model=_require_str(llm, "model", "llm"),
                temperature=_require_number(llm, "temperature", "llm"),
                max_tokens=_require_int(llm, "max_tokens", "llm"),
                api_key=llm.get("api_key"),
                api_version=llm.get("api_version"),
                azure_endpoint=llm.get("azure_endpoint"),
                deployment_name=llm.get("deployment_name"),
                base_url=llm.get("base_url"),
                server_bin=llm_llamacpp["server_bin"],
                model_path=llm_llamacpp["model_path"],
                extra_args=llm_llamacpp["extra_args"],
                auto_manage=llm_llamacpp["auto_manage"],
                idle_timeout=llm_llamacpp["idle_timeout"],
                startup_timeout=llm_llamacpp["startup_timeout"],
                exclusive_gpu=llm_llamacpp["exclusive_gpu"],
            ),
            embedding=EmbeddingSettings(
                provider=_require_str(embedding, "provider", "embedding"),
                model=_require_str(embedding, "model", "embedding"),
                dimensions=_require_int(embedding, "dimensions", "embedding"),
                api_key=embedding.get("api_key"),
                api_version=embedding.get("api_version"),
                azure_endpoint=embedding.get("azure_endpoint"),
                deployment_name=embedding.get("deployment_name"),
                base_url=embedding.get("base_url"),
                server_bin=embedding_llamacpp["server_bin"],
                model_path=embedding_llamacpp["model_path"],
                extra_args=embedding_llamacpp["extra_args"],
                auto_manage=embedding_llamacpp["auto_manage"],
                idle_timeout=embedding_llamacpp["idle_timeout"],
                startup_timeout=embedding_llamacpp["startup_timeout"],
                exclusive_gpu=embedding_llamacpp["exclusive_gpu"],
            ),
            vector_store=VectorStoreSettings(
                provider=_require_str(vector_store, "provider", "vector_store"),
                persist_directory=_require_str(vector_store, "persist_directory", "vector_store"),
                collection_name=_require_str(vector_store, "collection_name", "vector_store"),
            ),
            retrieval=RetrievalSettings(
                dense_top_k=_require_int(retrieval, "dense_top_k", "retrieval"),
                sparse_top_k=_require_int(retrieval, "sparse_top_k", "retrieval"),
                fusion_top_k=_require_int(retrieval, "fusion_top_k", "retrieval"),
                rrf_k=_require_int(retrieval, "rrf_k", "retrieval"),
            ),
            rerank=RerankSettings(
                enabled=_require_bool(rerank, "enabled", "rerank"),
                provider=_require_str(rerank, "provider", "rerank"),
                model=_require_str(rerank, "model", "rerank"),
                top_k=_require_int(rerank, "top_k", "rerank"),
            ),
            evaluation=EvaluationSettings(
                enabled=_require_bool(evaluation, "enabled", "evaluation"),
                provider=_require_str(evaluation, "provider", "evaluation"),
                metrics=[str(item) for item in _require_list(evaluation, "metrics", "evaluation")],
                backends=_optional_str_list(evaluation, "backends", "evaluation"),
            ),
            observability=ObservabilitySettings(
                log_level=_require_str(observability, "log_level", "observability"),
                trace_enabled=_require_bool(observability, "trace_enabled", "observability"),
                trace_file=_require_str(observability, "trace_file", "observability"),
                structured_logging=_require_bool(observability, "structured_logging", "observability"),
            ),
            ingestion=ingestion_settings,
            vision_llm=vision_llm_settings,
        )

        return settings


def validate_settings(settings: Settings) -> None:
    """对解析后的 Settings 做二次业务校验。

    补充 from_dict 未覆盖的「非空 provider」等约束，不通过则抛出 SettingsError。

    Args:
        settings: 已由 from_dict 构造的配置对象。

    Raises:
        SettingsError: 任一关键 provider 或 log_level 为空。
    """
    if not settings.llm.provider:
        raise SettingsError("Missing required field: llm.provider")
    if not settings.embedding.provider:
        raise SettingsError("Missing required field: embedding.provider")
    if not settings.vector_store.provider:
        raise SettingsError("Missing required field: vector_store.provider")
    if not settings.retrieval.rrf_k:
        raise SettingsError("Missing required field: retrieval.rrf_k")
    if not settings.rerank.provider:
        raise SettingsError("Missing required field: rerank.provider")
    if not settings.evaluation.provider:
        raise SettingsError("Missing required field: evaluation.provider")
    if not settings.observability.log_level:
        raise SettingsError("Missing required field: observability.log_level")


def load_settings(path: str | Path | None = None) -> Settings:
    """Load settings from a YAML file and validate required fields.

    Args:
        path: 配置文件路径；默认使用 ``<repo>/config/settings.yaml`` 的绝对路径，
            与工作目录无关。

    Returns:
        校验通过的 Settings 实例。

    Raises:
        SettingsError: 文件不存在、YAML 无效或校验失败。
    """
    settings_path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH
    if not settings_path.is_absolute():
        settings_path = resolve_path(settings_path)
    if not settings_path.exists():
        raise SettingsError(f"Settings file not found: {settings_path}")

    with settings_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    settings = Settings.from_dict(data or {})
    validate_settings(settings)
    return settings
