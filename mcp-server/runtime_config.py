#!/usr/bin/env python3
"""Runtime configuration for model routing.

Goal: allow non-technical users to switch between:
- NVIDIA NIM services
- externally hosted model services (any container or service implementing the same REST contract)
- embedded execution inside the MCP server container (last-resort convenience)

The dashboard can read/update this config via MCP server REST endpoints.

Config is optionally persisted to disk via MCP_CONFIG_PATH.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


ServiceName = Literal["alphafold", "rfdiffusion", "proteinmpnn", "alphafold_multimer"]
ProviderName = Literal["nim", "external", "embedded"]


def _truthy_env(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_env_url(key: str) -> Optional[str]:
    if key not in os.environ:
        return None
    value = (os.environ.get(key) or "").strip()
    if not value or value.lower() in {"disabled", "none", "null"}:
        return None
    return value


def default_nim_urls() -> Dict[ServiceName, Optional[str]]:
    # Defaults match the original repo defaults.
    return {
        "alphafold": _resolve_env_url("ALPHAFOLD_URL") or "http://localhost:8081",
        "rfdiffusion": _resolve_env_url("RFDIFFUSION_URL") or "http://localhost:8082",
        "proteinmpnn": _resolve_env_url("PROTEINMPNN_URL") or "http://localhost:8083",
        "alphafold_multimer": _resolve_env_url("ALPHAFOLD_MULTIMER_URL") or "http://localhost:8084",
    }


def default_external_urls() -> Dict[ServiceName, Optional[str]]:
    # Optional secondary URL set for non-NIM model services.
    return {
        "alphafold": _resolve_env_url("EXTERNAL_ALPHAFOLD_URL"),
        "rfdiffusion": _resolve_env_url("EXTERNAL_RFDIFFUSION_URL"),
        "proteinmpnn": _resolve_env_url("EXTERNAL_PROTEINMPNN_URL"),
        "alphafold_multimer": _resolve_env_url("EXTERNAL_ALPHAFOLD_MULTIMER_URL"),
    }


class ProviderConfig(BaseModel):
    enabled: bool = True
    # service_urls is used by REST providers (nim/external). embedded ignores it.
    service_urls: Dict[ServiceName, Optional[str]] = Field(default_factory=dict)


class EmbeddedConfig(BaseModel):
    enabled: bool = True
    # Directory inside the MCP server container for downloading/storing model assets.
    model_dir: str = "/models"
    # If true, embedded providers may attempt a best-effort bootstrap (pip installs, downloads).
    # Keep false by default; explicit opt-in.
    auto_install: bool = False


class RoutingConfig(BaseModel):
    mode: Literal["single", "fallback"] = "fallback"

    # Used when mode == "single"
    primary: ProviderName = "nim"

    # Used when mode == "fallback"; order to try.
    order: List[ProviderName] = Field(default_factory=lambda: ["nim", "external", "embedded"])


class MCPServerConfig(BaseModel):
    version: int = 1

    routing: RoutingConfig = Field(default_factory=RoutingConfig)

    nim: ProviderConfig = Field(default_factory=lambda: ProviderConfig(service_urls=default_nim_urls()))
    external: ProviderConfig = Field(default_factory=lambda: ProviderConfig(service_urls=default_external_urls()))
    embedded: EmbeddedConfig = Field(default_factory=EmbeddedConfig)

    # Safety: allow runtime config edits. Defaults on for local stacks.
    allow_runtime_updates: bool = Field(default_factory=lambda: not _truthy_env("MCP_CONFIG_READONLY"))


class RuntimeConfigManager:
    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else Path(os.getenv("MCP_CONFIG_PATH", "").strip() or "")
        self._config = MCPServerConfig()
        self._revision = 0
        self._load_from_disk_if_present()

    @property
    def revision(self) -> int:
        return self._revision

    def get(self) -> MCPServerConfig:
        return self._config

    def _load_from_disk_if_present(self) -> None:
        if not self.path:
            return
        try:
            if not self.path.exists():
                return
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._config = MCPServerConfig.model_validate(data)
            self._revision += 1
        except Exception:
            # Keep defaults if config file is invalid.
            return

    def _persist(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._config.model_dump(), indent=2), encoding="utf-8")

    def update(self, patch: Dict[str, Any]) -> MCPServerConfig:
        if not self._config.allow_runtime_updates:
            raise PermissionError("Runtime config updates are disabled (MCP_CONFIG_READONLY=1)")
        merged = self._config.model_dump()

        def deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
            for k, v in src.items():
                if isinstance(v, dict) and isinstance(dst.get(k), dict):
                    dst[k] = deep_merge(dst.get(k, {}), v)
                else:
                    dst[k] = v
            return dst

        merged = deep_merge(merged, patch)
        self._config = MCPServerConfig.model_validate(merged)
        self._revision += 1
        self._persist()
        return self._config

    def reset_to_defaults(self) -> MCPServerConfig:
        if not self._config.allow_runtime_updates:
            raise PermissionError("Runtime config updates are disabled (MCP_CONFIG_READONLY=1)")
        self._config = MCPServerConfig()
        self._revision += 1
        self._persist()
        return self._config
