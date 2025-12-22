from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .common import (
    env_str,
    require_env,
    run_cmd,
    ensure_file_exists,
    read_text,
    available_cpu_count,
    nvidia_gpu_present,
    meminfo_snapshot,
    alphafold_db_paths,
    estimate_file_cache_residency,
)

app = FastAPI(title="AlphaFold2 Native Service", version="1.0.0")


class StructureRequest(BaseModel):
    sequence: str = Field(..., description="Protein amino acid sequence")


def _maybe_inject_runtime_flags(cmd: str) -> str:
    # Auto-threading for MSA tools.
    # - If user already set flags, respect them.
    # - Otherwise pick a process-available CPU count (affinity/cgroup-aware).
    msa_n_cpu_raw = env_str("ALPHAFOLD_MSA_N_CPU")
    msa_n_cpu_max_raw = env_str("ALPHAFOLD_MSA_N_CPU_MAX")

    try:
        msa_n_cpu = int(msa_n_cpu_raw) if msa_n_cpu_raw else available_cpu_count()
    except Exception:
        msa_n_cpu = available_cpu_count()

    if msa_n_cpu_max_raw:
        try:
            msa_n_cpu = min(msa_n_cpu, int(msa_n_cpu_max_raw))
        except Exception:
            pass

    msa_n_cpu = max(1, msa_n_cpu)

    # Only add if none are present.
    if "--jackhmmer_n_cpu" not in cmd:
        cmd += f" --jackhmmer_n_cpu={msa_n_cpu}"
    if "--hmmsearch_n_cpu" not in cmd:
        cmd += f" --hmmsearch_n_cpu={msa_n_cpu}"
    if "--hhsearch_n_cpu" not in cmd:
        cmd += f" --hhsearch_n_cpu={msa_n_cpu}"

    # Optional: switch MSA implementation.
    msa_mode = (env_str("ALPHAFOLD_MSA_MODE", "") or "").strip().lower()
    if msa_mode and "--msa_mode" not in cmd:
        cmd += f" --msa_mode={msa_mode}"

    # Optional: MMseqs2 settings (only used when --msa_mode=mmseqs2).
    mmseqs_db = (env_str("ALPHAFOLD_MMSEQS2_DATABASE_PATH", "") or "").strip()
    if mmseqs_db and "--mmseqs2_database_path" not in cmd:
        cmd += f" --mmseqs2_database_path={mmseqs_db}"
    mmseqs_bin = (env_str("ALPHAFOLD_MMSEQS2_BINARY_PATH", "") or "").strip()
    if mmseqs_bin and "--mmseqs2_binary_path" not in cmd:
        cmd += f" --mmseqs2_binary_path={mmseqs_bin}"
    mmseqs_max = (env_str("ALPHAFOLD_MMSEQS2_MAX_SEQS", "") or "").strip()
    if mmseqs_max and "--mmseqs2_max_seqs" not in cmd:
        cmd += f" --mmseqs2_max_seqs={mmseqs_max}"

    # Optional: enable GPU relax when NVIDIA GPUs exist.
    # Note: this only affects the OpenMM relaxation step, not the main JAX model compute.
    use_gpu_relax = (env_str("ALPHAFOLD_USE_GPU_RELAX", "auto") or "auto").strip().lower()
    if use_gpu_relax in {"1", "true", "yes", "y", "on", "auto"}:
        if use_gpu_relax != "auto" or nvidia_gpu_present():
            if "--use_gpu_relax=true" not in cmd:
                if "--use_gpu_relax=false" in cmd:
                    cmd = cmd.replace("--use_gpu_relax=false", "--use_gpu_relax=true")
                elif "--use_gpu_relax" not in cmd:
                    cmd += " --use_gpu_relax=true"

    # Minimal logging to help debug performance without changing responses.
    try:
        print(f"[alphafold_service] msa_n_cpu={msa_n_cpu} gpu_present={nvidia_gpu_present()}")
    except Exception:
        pass

    return cmd


def _ready_reason() -> str | None:
    cmd = env_str("ALPHAFOLD_NATIVE_CMD")
    if not cmd:
        return "Set ALPHAFOLD_NATIVE_CMD to a command that produces a PDB file"
    return None


@app.get("/v1/health/ready")
def health_ready():
    reason = _ready_reason()
    if reason:
        raise HTTPException(status_code=503, detail=f"AlphaFold2 native service not configured: {reason}")
    return {"status": "ready"}


@app.get("/v1/metrics")
def metrics(include_residency: bool = False, max_samples: int = 128):
    """Diagnostics endpoint for cache/memory visibility.

    - include_residency=0: fast meminfo only
    - include_residency=1: also estimates page-cache residency of key DB files

    Residency estimation uses mincore sampling and does not read the file.
    """

    data_dir = env_str("ALPHAFOLD_DATA_DIR", env_str("ALPHAFOLD_DB_DIR", "")) or ""
    tier = (env_str("ALPHAFOLD_DB_TIER", "auto") or "auto").strip().lower()

    payload: dict[str, object] = {
        "meminfo": meminfo_snapshot(),
        "alphafold": {
            "data_dir": data_dir,
            "db_tier": tier,
        },
    }

    if include_residency and data_dir:
        paths = alphafold_db_paths(data_dir=data_dir, tier=tier)
        payload["db_files"] = paths
        payload["db_cache_residency"] = [estimate_file_cache_residency(p, max_samples=max_samples) for p in paths]

    return payload


@app.post("/v1/structure")
def structure(req: StructureRequest):
    seq = (req.sequence or "").strip()
    if not seq:
        raise HTTPException(status_code=400, detail="Missing sequence")

    cmd_template = require_env("ALPHAFOLD_NATIVE_CMD")
    timeout = int(env_str("ALPHAFOLD_NATIVE_TIMEOUT_SECONDS", "7200") or "7200")
    output_name = env_str("ALPHAFOLD_NATIVE_OUTPUT_PDB", "result.pdb") or "result.pdb"

    # Optional variables passed into template.
    db_dir = env_str("ALPHAFOLD_DB_DIR")
    model_dir = env_str("ALPHAFOLD_MODEL_DIR")

    with tempfile.TemporaryDirectory(prefix="alphafold_native_") as tmpdir:
        tmp = Path(tmpdir)
        fasta_path = tmp / "input.fasta"
        out_dir = tmp / "out"
        out_dir.mkdir(parents=True, exist_ok=True)

        fasta_path.write_text(f">query\n{seq}\n", encoding="utf-8")

        try:
            cmd = cmd_template.format(
                fasta=str(fasta_path),
                out_dir=str(out_dir),
                db_dir=str(db_dir) if db_dir else "",
                model_dir=str(model_dir) if model_dir else "",
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Invalid ALPHAFOLD_NATIVE_CMD template: {exc}")

        cmd = _maybe_inject_runtime_flags(cmd)

        try:
            proc = run_cmd(cmd, timeout_seconds=timeout)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            detail = (stderr or stdout or f"exit code {proc.returncode}")
            raise HTTPException(
                status_code=500,
                detail=f"AlphaFold2 native command failed (cmd={cmd}): {detail}",
            )

        pdb_path = out_dir / output_name
        if not pdb_path.exists():
            # AlphaFold writes outputs under a subdirectory named after the FASTA basename.
            # Our command template typically sets --output_dir={out_dir}, so the actual
            # PDB lands at {out_dir}/{fasta_name}/ranked_0.pdb.
            matches = sorted(out_dir.glob(f"*/{output_name}"))
            if matches:
                pdb_path = matches[0]
        try:
            ensure_file_exists(str(pdb_path), label="AlphaFold2 output PDB")
            pdb_text = read_text(str(pdb_path))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"AlphaFold2 did not produce expected output: {exc}")

    return {
        "backend": "native",
        "pdb": pdb_text,
        "sequence": seq,
    }
