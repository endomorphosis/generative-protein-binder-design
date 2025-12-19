from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .common import env_str, require_env, run_cmd, ensure_file_exists, read_text

app = FastAPI(title="AlphaFold2 Native Service", version="1.0.0")


class StructureRequest(BaseModel):
    sequence: str = Field(..., description="Protein amino acid sequence")


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

        proc = run_cmd(cmd, timeout_seconds=timeout)
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            stdout = (proc.stdout or "").strip()
            detail = (stderr or stdout or f"exit code {proc.returncode}")
            raise HTTPException(status_code=500, detail=f"AlphaFold2 native command failed: {detail}")

        pdb_path = out_dir / output_name
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
