"""配置加载与校验的单元测试。"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from core.settings import SettingsError, load_settings, validate_settings, Settings


@pytest.mark.unit
class TestLoadSettings:
    """验证默认 settings.yaml 可成功加载并映射为 Settings。"""

    def test_load_default_settings(self) -> None:
        """默认 config/settings.yaml 应包含全部必填字段并能通过校验。"""
        settings = load_settings()
        assert settings.llm.provider == "llamacpp"
        assert settings.embedding.provider == "llamacpp"
        assert settings.vector_store.provider == "chroma"
        assert settings.retrieval.rrf_k == 60
        assert settings.llm.auto_manage is True
        assert settings.embedding.auto_manage is True
        assert settings.llm.server_bin is not None
        assert settings.llm.model_path is not None
        assert settings.embedding.model_path is not None
        assert settings.llm.idle_timeout == 8.0
        assert settings.llm.exclusive_gpu is True


@pytest.mark.unit
class TestValidateSettings:
    """验证 validate_settings 对缺失关键字段给出可读错误。"""

    def test_missing_embedding_provider_raises(self) -> None:
        """缺失 embedding.provider 时应抛出包含字段路径的 SettingsError。"""
        settings = load_settings()
        broken = Settings(
            llm=settings.llm,
            embedding=type(settings.embedding)(
                provider="",
                model=settings.embedding.model,
                dimensions=settings.embedding.dimensions,
                base_url=settings.embedding.base_url,
            ),
            vector_store=settings.vector_store,
            retrieval=settings.retrieval,
            rerank=settings.rerank,
            evaluation=settings.evaluation,
            observability=settings.observability,
            ingestion=settings.ingestion,
            vision_llm=settings.vision_llm,
        )
        with pytest.raises(SettingsError, match="embedding.provider"):
            validate_settings(broken)


@pytest.mark.unit
class TestLoadSettingsFromFile:
    """验证从临时 YAML 文件加载及缺失字段时的错误信息。"""

    def test_missing_field_in_yaml_raises_readable_error(self, tmp_path: Path) -> None:
        """YAML 缺少 embedding 块时应明确指出缺失字段路径。"""
        partial = {
            "llm": {"provider": "ollama", "model": "m", "temperature": 0.0, "max_tokens": 100},
            "vector_store": {
                "provider": "chroma",
                "persist_directory": "./data",
                "collection_name": "test",
            },
            "retrieval": {"dense_top_k": 10, "sparse_top_k": 10, "fusion_top_k": 5, "rrf_k": 60},
            "rerank": {"enabled": False, "provider": "none", "model": "none", "top_k": 5},
            "evaluation": {"enabled": False, "provider": "custom", "metrics": ["hit_rate"]},
            "observability": {
                "log_level": "INFO",
                "trace_enabled": True,
                "trace_file": "./logs/traces.jsonl",
                "structured_logging": True,
            },
        }
        yaml_path = tmp_path / "settings.yaml"
        yaml_path.write_text(yaml.dump(partial), encoding="utf-8")

        with pytest.raises(SettingsError, match="embedding"):
            load_settings(yaml_path)

    def test_load_custom_yaml_file(self, tmp_path: Path) -> None:
        """完整的最小 YAML 应能成功解析为 Settings。"""
        content = textwrap.dedent(
            """
            llm:
              provider: openai
              model: gpt-4o-mini
              temperature: 0.0
              max_tokens: 1024
            embedding:
              provider: openai
              model: text-embedding-3-small
              dimensions: 1536
            vector_store:
              provider: chroma
              persist_directory: "./data/db/chroma"
              collection_name: test
            retrieval:
              dense_top_k: 10
              sparse_top_k: 10
              fusion_top_k: 5
              rrf_k: 60
            rerank:
              enabled: false
              provider: none
              model: none
              top_k: 5
            evaluation:
              enabled: false
              provider: custom
              metrics: [hit_rate]
            observability:
              log_level: INFO
              trace_enabled: true
              trace_file: "./logs/traces.jsonl"
              structured_logging: true
            """
        )
        yaml_path = tmp_path / "ok.yaml"
        yaml_path.write_text(content, encoding="utf-8")

        settings = load_settings(yaml_path)
        assert settings.llm.provider == "openai"
        assert settings.embedding.dimensions == 1536
        assert settings.llm.server_bin is None
        assert settings.llm.auto_manage is None

    def test_llamacpp_runtime_block_merges_into_roles(self, tmp_path: Path) -> None:
        """顶层 llamacpp 块应合并进 llm/embedding，角色字段优先。"""
        content = textwrap.dedent(
            """
            llm:
              provider: llamacpp
              model: qwen
              temperature: 0.0
              max_tokens: 128
              model_path: "D:\\\\models\\\\chat.gguf"
              extra_args: ["--alias", "qwen"]
            embedding:
              provider: llamacpp
              model: bge
              dimensions: 1024
              model_path: "D:\\\\models\\\\embed.gguf"
            vector_store:
              provider: chroma
              persist_directory: "./data/db/chroma"
              collection_name: test
            retrieval:
              dense_top_k: 10
              sparse_top_k: 10
              fusion_top_k: 5
              rrf_k: 60
            rerank:
              enabled: false
              provider: none
              model: none
              top_k: 5
            evaluation:
              enabled: false
              provider: custom
              metrics: [hit_rate]
            observability:
              log_level: INFO
              trace_enabled: true
              trace_file: "./logs/traces.jsonl"
              structured_logging: true
            llamacpp:
              auto_manage: true
              server_bin: "D:\\\\llama\\\\llama-server.exe"
              exclusive_gpu: true
              idle_timeout: 0
              startup_timeout: 60
            """
        )
        yaml_path = tmp_path / "llamacpp.yaml"
        yaml_path.write_text(content, encoding="utf-8")
        settings = load_settings(yaml_path)
        assert settings.llm.server_bin == r"D:\llama\llama-server.exe"
        assert settings.embedding.server_bin == r"D:\llama\llama-server.exe"
        assert settings.llm.model_path == r"D:\models\chat.gguf"
        assert settings.embedding.model_path == r"D:\models\embed.gguf"
        assert settings.llm.extra_args == ("--alias", "qwen")
        assert settings.embedding.extra_args == ()
        assert settings.llm.auto_manage is True
        assert settings.llm.idle_timeout == 0.0
        assert settings.llm.exclusive_gpu is True

