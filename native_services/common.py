from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional


def _truthy_env(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    val = os.getenv(name)
    if val is None:
        return default
    val = val.strip()
    return val if val else default


def require_env(name: str) -> str:
    val = env_str(name)
    if not val:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return val


def run_cmd(cmd: str, *, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    # Use bash -lc so users can rely on conda init / module loads, etc.
    # This also allows users to provide complex pipelines.
    #
    # IMPORTANT: These native services run under a Python venv (for FastAPI/uvicorn),
    # but the model commands often need to run under conda envs or system installs.
    # If we leak VIRTUAL_ENV and the venv's bin/ prefix into the child process,
    # some tools (notably `conda run`) may end up selecting the wrong python.
    child_env = os.environ.copy()
    venv = child_env.pop("VIRTUAL_ENV", None)
    if venv:
        path = child_env.get("PATH", "")
        venv_bin_prefix = f"{venv}/bin:"
        if path.startswith(venv_bin_prefix):
            child_env["PATH"] = path[len(venv_bin_prefix) :]

    proc = subprocess.run(
        ["bash", "-lc", cmd],
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        env=child_env,
    )
    return proc


def read_text(path: str) -> str:
    p = Path(path)
    return p.read_text(encoding="utf-8")


def ensure_file_exists(path: str, *, label: str) -> None:
    p = Path(path)
    if not p.exists() or not p.is_file() or p.stat().st_size <= 0:
        raise RuntimeError(f"{label} not found or empty: {path}")
