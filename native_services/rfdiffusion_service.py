from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .common import env_str, require_env, run_cmd, ensure_file_exists, read_text

app = FastAPI(title="RFdiffusion Native Service", version="1.0.0")


class DesignRequest(BaseModel):
    target_pdb: str = Field(..., description="Target protein PDB content")
    num_designs: int = Field(5, ge=1, le=50, description="Number of designs")


def _ready_reason() -> str | None:
    cmd = env_str("RFDIFFUSION_NATIVE_CMD")
    if not cmd:
        return "Set RFDIFFUSION_NATIVE_CMD to a command that produces design_{design_id}.pdb"
    return None


@app.get("/v1/health/ready")
def health_ready():
    reason = _ready_reason()
    if reason:
        raise HTTPException(status_code=503, detail=f"RFdiffusion native service not configured: {reason}")
    return {"status": "ready"}


@app.post("/v1/design")
def design(req: DesignRequest):
    target_pdb = (req.target_pdb or "").strip()
    if not target_pdb:
        raise HTTPException(status_code=400, detail="Missing target_pdb")

    n = max(1, min(int(req.num_designs or 5), 50))

    cmd_template = require_env("RFDIFFUSION_NATIVE_CMD")
    timeout = int(env_str("RFDIFFUSION_NATIVE_TIMEOUT_SECONDS", "7200") or "7200")
    # RFdiffusion's run_inference.py writes files as: {output_prefix}_{i_des}.pdb.
    # Our default command template sets output_prefix={out_dir}/design_{design_id} and num_designs=1,
    # so the produced file is typically design_{design_id}_0.pdb.
    output_template = env_str("RFDIFFUSION_NATIVE_OUTPUT_PDB", "design_{design_id}_0.pdb") or "design_{design_id}_0.pdb"

    models_dir = env_str("RFDIFFUSION_MODELS_DIR")

    with tempfile.TemporaryDirectory(prefix="rfdiffusion_native_") as tmpdir:
        tmp = Path(tmpdir)
        target_path = tmp / "target.pdb"
        out_dir = tmp / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        target_path.write_text(target_pdb + "\n", encoding="utf-8")

        designs = []
        for design_id in range(n):
            try:
                cmd = cmd_template.format(
                    target_pdb=str(target_path),
                    out_dir=str(out_dir),
                    design_id=str(design_id),
                    models_dir=str(models_dir) if models_dir else "",
                )
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"Invalid RFDIFFUSION_NATIVE_CMD template: {exc}")

            try:
                proc = run_cmd(cmd, timeout_seconds=timeout)
            except RuntimeError as exc:
                raise HTTPException(status_code=503, detail=str(exc))
            if proc.returncode != 0:
                stderr = (proc.stderr or "").strip()
                stdout = (proc.stdout or "").strip()
                detail = (stderr or stdout or f"exit code {proc.returncode}")
                raise HTTPException(status_code=500, detail=f"RFdiffusion native command failed (design {design_id}): {detail}")

            out_name = output_template.format(design_id=design_id)
            pdb_path = out_dir / out_name
            try:
                ensure_file_exists(str(pdb_path), label=f"RFdiffusion output PDB for design {design_id}")
                pdb_text = read_text(str(pdb_path))
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"RFdiffusion did not produce expected output for design {design_id}: {exc}")

            designs.append({"design_id": design_id, "pdb": pdb_text})

    return {
        "backend": "native",
        "designs": designs,
    }
