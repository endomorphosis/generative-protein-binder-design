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
from pydantic import ConfigDict


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
    def _env_or_default(env_key: str, default_url: str) -> Optional[str]:
        # If the env var is present, honor it even if it disables the service.
        if env_key in os.environ:
            return _resolve_env_url(env_key)
        return default_url

    return {
        "alphafold": _env_or_default("ALPHAFOLD_URL", "http://localhost:8081"),
        "rfdiffusion": _env_or_default("RFDIFFUSION_URL", "http://localhost:8082"),
        "proteinmpnn": _env_or_default("PROTEINMPNN_URL", "http://localhost:8083"),
        "alphafold_multimer": _env_or_default("ALPHAFOLD_MULTIMER_URL", "http://localhost:8084"),
    }


_NIM_LOCALHOST_DEFAULTS: Dict[ServiceName, str] = {
    "alphafold": "http://localhost:8081",
    "rfdiffusion": "http://localhost:8082",
    "proteinmpnn": "http://localhost:8083",
    "alphafold_multimer": "http://localhost:8084",
}


_NIM_ENV_KEYS: Dict[ServiceName, str] = {
    "alphafold": "ALPHAFOLD_URL",
    "rfdiffusion": "RFDIFFUSION_URL",
    "proteinmpnn": "PROTEINMPNN_URL",
    "alphafold_multimer": "ALPHAFOLD_MULTIMER_URL",
}


def _migrate_localhost_nim_urls_from_env(cfg: "MCPServerConfig") -> bool:
    """If a persisted config still uses old localhost defaults, migrate those
    entries to current env-provided URLs (or disable if env explicitly disables).

    This avoids confusing "not_ready" caused by stale defaults inside Docker
    stacks where service discovery uses compose DNS (e.g. http://alphafold:8000).

    It is intentionally conservative: it only rewrites values that exactly match
    the historical localhost defaults.
    """

    changed = False

    def _running_in_docker() -> bool:
        try:
            return Path("/.dockerenv").exists() or _truthy_env("DOCKER_CONTAINER")
        except Exception:
            return False

    def _looks_like_compose_dns() -> bool:
        try:
            urls = (cfg.nim.service_urls or {}).values()
            for u in urls:
                if not isinstance(u, str):
                    continue
                # Compose DNS style (service name on internal network)
                if "://alphafold:" in u or "://rfdiffusion:" in u or "://proteinmpnn:" in u or "://alphafold-multimer:" in u:
                    return True
        except Exception:
            return False
        return False
    nim_urls = dict(cfg.nim.service_urls or {})
    for service_name, default_url in _NIM_LOCALHOST_DEFAULTS.items():
        current = nim_urls.get(service_name)
        if current != default_url:
            continue
        env_key = _NIM_ENV_KEYS[service_name]

        # In containerized stacks, localhost defaults are almost always wrong
        # (they point at the MCP container itself). If the env var is absent,
        # treat it as disabled for that service.
        if (env_key in os.environ) or _running_in_docker() or _looks_like_compose_dns():
            nim_urls[service_name] = _resolve_env_url(env_key)
            changed = True

    if changed:
        cfg.nim.service_urls = nim_urls
    return changed


def _apply_embedded_env_overrides(cfg: "MCPServerConfig") -> bool:
    """Apply runtime env overrides for embedded provisioning toggles.

    This supports zero-touch deployments where operators control bootstrap via
    environment variables even if a persisted config exists.
    """

    changed = False
    try:
        if "MCP_EMBEDDED_AUTO_DOWNLOAD" in os.environ:
            v = _truthy_env("MCP_EMBEDDED_AUTO_DOWNLOAD")
            if getattr(cfg.embedded, "auto_download", False) != v:
                cfg.embedded.auto_download = v
                changed = True
        if "MCP_EMBEDDED_AUTO_INSTALL" in os.environ:
            v = _truthy_env("MCP_EMBEDDED_AUTO_INSTALL")
            if getattr(cfg.embedded, "auto_install", False) != v:
                cfg.embedded.auto_install = v
                changed = True
    except Exception:
        return changed

    return changed


def _apply_routing_env_overrides(cfg: "MCPServerConfig") -> bool:
    """Apply runtime env overrides for routing.

    Supported env vars:
      - MCP_ROUTING_MODE: "single" or "fallback"
      - MCP_ROUTING_PRIMARY: ProviderName
      - MCP_ROUTING_ORDER: comma-separated ProviderName list
    """

    changed = False
    try:
        if "MCP_ROUTING_MODE" in os.environ:
            mode = (os.getenv("MCP_ROUTING_MODE") or "").strip().lower()
            if mode in {"single", "fallback"} and cfg.routing.mode != mode:
                cfg.routing.mode = mode  # type: ignore[assignment]
                changed = True

        if "MCP_ROUTING_PRIMARY" in os.environ:
            primary = (os.getenv("MCP_ROUTING_PRIMARY") or "").strip().lower()
            if primary in {"nim", "external", "embedded"} and cfg.routing.primary != primary:
                cfg.routing.primary = primary  # type: ignore[assignment]
                changed = True

        if "MCP_ROUTING_ORDER" in os.environ:
            raw = (os.getenv("MCP_ROUTING_ORDER") or "").strip()
            if raw:
                parts = [p.strip().lower() for p in raw.split(",") if p.strip()]
                order = [p for p in parts if p in {"nim", "external", "embedded"}]
                # de-dup preserving order
                dedup: List[ProviderName] = []
                for p in order:
                    if p not in dedup:
                        dedup.append(p)  # type: ignore[arg-type]
                if dedup and cfg.routing.order != dedup:
                    cfg.routing.order = dedup
                    changed = True
    except Exception:
        return changed

    return changed


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


class EmbeddedDownloads(BaseModel):
    # ProteinMPNN
    proteinmpnn_source_tarball_url: Optional[str] = None
    proteinmpnn_weights_url: Optional[str] = None

    # RFdiffusion
    rfdiffusion_weights_url: Optional[str] = None

    # AlphaFold2
    # NOTE: AlphaFold databases are very large; this is opt-in and requires an explicit URL.
    # Reduced/initial DB pack (recommended default when doing staged installs)
    alphafold_db_url: Optional[str] = None
    # Optional follow-on pack for staging additional DB assets after the reduced pack is available.
    alphafold_db_url_full: Optional[str] = None
    # Subdirectory under model_dir to place/extract databases.
    alphafold_db_subdir: str = "alphafold_db"

    # AlphaFold2 / MGnify
    # Some environments block the default MGnify download (HTTP 403). These settings
    # allow users to provide an alternate URL (e.g., a signed URL or internal mirror)
    # or opt into a HuggingFace-backed fallback.
    alphafold_mgnify_url: Optional[str] = None
    alphafold_mgnify_fallback: Literal["none", "huggingface"] = "none"
    alphafold_mgnify_hf_dataset: str = "tattabio/OMG_prot50"
    # If set, this token may be persisted to disk alongside other runtime config.
    # Prefer supplying via environment variables when possible.
    alphafold_mgnify_hf_token: Optional[str] = None


class RunnerCommand(BaseModel):
    """A command template (argv) for running a model inside the MCP container.

    The dashboard can store commands as an array of strings.

    Supported placeholders:
        - {model_dir}
        - {work_dir}
        - {fasta_path}
        - {output_pdb_path}
        - {target_pdb_path}
        - {output_dir}
        - {num_designs}
        - {design_id}

    Example AlphaFold:
        ["python", "/opt/alphafold/run.py", "--fasta", "{fasta_path}", "--out", "{output_pdb_path}"]
    """

    argv: List[str] = Field(default_factory=list)
    timeout_seconds: int = 3600


class EmbeddedRunners(BaseModel):
    # If argv is empty, the embedded backend will report not_ready with a clear reason.
    alphafold: RunnerCommand = Field(default_factory=RunnerCommand)
    rfdiffusion: RunnerCommand = Field(default_factory=RunnerCommand)
    alphafold_multimer: RunnerCommand = Field(default_factory=RunnerCommand)


class EmbeddedConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    enabled: bool = True
    # Directory inside the MCP server container for downloading/storing model assets.
    model_dir: str = "/models"
    # If true, embedded providers may attempt a best-effort bootstrap (pip installs, downloads).
    # Keep false by default; explicit opt-in.
    auto_install: bool = Field(default_factory=lambda: _truthy_env("MCP_EMBEDDED_AUTO_INSTALL"))
    # If true, the server may download configured assets (URLs must be explicitly provided).
    auto_download: bool = Field(default_factory=lambda: _truthy_env("MCP_EMBEDDED_AUTO_DOWNLOAD"))
    downloads: EmbeddedDownloads = Field(default_factory=EmbeddedDownloads)
    runners: EmbeddedRunners = Field(default_factory=EmbeddedRunners)


class RoutingConfig(BaseModel):
    mode: Literal["single", "fallback"] = "fallback"

    # Used when mode == "single"
    primary: ProviderName = "nim"

    # Used when mode == "fallback"; order to try.
    order: List[ProviderName] = Field(default_factory=lambda: ["nim", "external", "embedded"])


class MCPServerConfig(BaseModel):
    version: int = 3

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
        # Apply env overrides to the initial in-memory defaults.
        if _apply_routing_env_overrides(self._config):
            self._revision += 1
        if _apply_embedded_env_overrides(self._config):
            self._revision += 1
        self._load_from_disk_if_present()

        # Ensure there's always a persisted config file when MCP_CONFIG_PATH is set.
        # This makes the dashboard settings editable out-of-the-box.
        try:
            if self.path and not self.path.exists() and self._config.allow_runtime_updates:
                self._persist()
                self._revision += 1
        except Exception:
            pass

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

            # Optional startup migration for stale localhost defaults.
            if _migrate_localhost_nim_urls_from_env(self._config):
                self._revision += 1
                if self._config.allow_runtime_updates:
                    self._persist()

            # Apply env overrides after loading persisted config.
            if _apply_embedded_env_overrides(self._config):
                self._revision += 1

            if _apply_routing_env_overrides(self._config):
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
