"""Configuration management for Agentic RAG Assistant.

Supports:
- YAML configuration file loading
- Environment variable substitution
- Validation with Pydantic
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def _substitute_env_vars(value: str) -> str:
    """Substitute environment variables in string value.

    Supports ${VAR_NAME} syntax.
    """
    pattern = r"\$\{([^}]+)\}"

    def replace(match: re.Match) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))

    return re.sub(pattern, replace, value)


def _process_config_values(obj: Any) -> Any:
    """Recursively process config values for env var substitution."""
    if isinstance(obj, dict):
        return {k: _process_config_values(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_process_config_values(item) for item in obj]
    elif isinstance(obj, str):
        return _substitute_env_vars(obj)
    return obj


@dataclass
class LLMConfig:
    """LLM provider configuration."""

    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 4096

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMConfig":
        return cls(
            provider=data.get("provider", "openai"),
            model=data.get("model", "gpt-4o"),
            api_key=data.get("api_key"),
            base_url=data.get("base_url"),
            temperature=data.get("temperature", 0.0),
            max_tokens=data.get("max_tokens", 4096),
        )


@dataclass
class AgentConfig:
    """Agent behavior configuration."""

    max_rewrite_attempts: int = 2
    retrieval_top_k: int = 10
    sufficiency_threshold: float = 0.7
    enable_sub_query: bool = True
    enable_query_rewrite: bool = True

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentConfig":
        return cls(
            max_rewrite_attempts=data.get("max_rewrite_attempts", 2),
            retrieval_top_k=data.get("retrieval_top_k", 10),
            sufficiency_threshold=data.get("sufficiency_threshold", 0.7),
            enable_sub_query=data.get("enable_sub_query", True),
            enable_query_rewrite=data.get("enable_query_rewrite", True),
        )


@dataclass
class RAGServerConfig:
    """RAG MCP Server connection configuration."""

    url: str = "http://127.0.0.1:8080"
    collection: str = "knowledge_hub"
    timeout: int = 60

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RAGServerConfig":
        return cls(
            url=data.get("url", "http://127.0.0.1:8080"),
            collection=data.get("collection", "knowledge_hub"),
            timeout=data.get("timeout", 60),
        )


@dataclass
class LangFuseConfig:
    """LangFuse tracing configuration."""

    enabled: bool = False
    public_key: Optional[str] = None
    secret_key: Optional[str] = None
    host: str = "https://cloud.langfuse.com"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LangFuseConfig":
        return cls(
            enabled=data.get("enabled", False),
            public_key=data.get("public_key"),
            secret_key=data.get("secret_key"),
            host=data.get("host", "https://cloud.langfuse.com"),
        )


@dataclass
class UIConfig:
    """Streamlit UI configuration."""

    title: str = "企业知识助手"
    theme: str = "light"
    show_trace: bool = True
    port: int = 8502

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UIConfig":
        return cls(
            title=data.get("title", "企业知识助手"),
            theme=data.get("theme", "light"),
            show_trace=data.get("show_trace", True),
            port=data.get("port", 8502),
        )


@dataclass
class Settings:
    """Application settings."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    rag_server: RAGServerConfig = field(default_factory=RAGServerConfig)
    langfuse: LangFuseConfig = field(default_factory=LangFuseConfig)
    ui: UIConfig = field(default_factory=UIConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Settings":
        return cls(
            llm=LLMConfig.from_dict(data.get("llm", {})),
            agent=AgentConfig.from_dict(data.get("agent", {})),
            rag_server=RAGServerConfig.from_dict(data.get("rag_server", {})),
            langfuse=LangFuseConfig.from_dict(data.get("langfuse", {})),
            ui=UIConfig.from_dict(data.get("ui", {})),
        )


def load_settings(config_path: str = "config/settings.yaml") -> Settings:
    """Load settings from YAML file with environment variable substitution.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Settings object with loaded configuration.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML is invalid.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    if raw_data is None:
        raw_data = {}

    processed_data = _process_config_values(raw_data)

    return Settings.from_dict(processed_data)


def resolve_path(relative_path: str, base_dir: Optional[str] = None) -> Path:
    """Resolve a path relative to the project root or a given base directory.

    Args:
        relative_path: The relative path to resolve.
        base_dir: Optional base directory. If None, uses project root.

    Returns:
        Resolved absolute path.
    """
    if base_dir:
        base = Path(base_dir)
    else:
        base = Path(__file__).parent.parent.parent

    return (base / relative_path).resolve()
