from __future__ import annotations

import os
import tempfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import rfdiffusion_runner

app = FastAPI(title="RFDiffusion ARM64 Service", version="0.1.0")


class DesignRequest(BaseModel):
    target_pdb: str = Field(..., description="Target protein PDB content")
    num_designs: int = Field(5, ge=1, le=50, description="Number of designs")


@app.get("/v1/health/ready")
def health_ready():
    if not getattr(rfdiffusion_runner, "is_ready", lambda: True)():
        raise HTTPException(
            status_code=503,
            detail=(
                "RFDiffusion service not ready: ARM64 container is a CI-only shim (no real inference). "
                "Configure a real provider via MCP Dashboard (External/NIM) or deploy native RFdiffusion."
            ),
        )
    return {"status": "ready"}


@app.post("/v1/design")
def design(req: DesignRequest):
    target_pdb = (req.target_pdb or "").strip()
    if not target_pdb:
        raise HTTPException(status_code=400, detail="Missing target_pdb")

    num_designs = int(req.num_designs or 5)

    with tempfile.TemporaryDirectory(prefix="rfdiffusion_") as tmpdir:
        target_path = os.path.join(tmpdir, "target.pdb")
        out_dir = os.path.join(tmpdir, "out")
        os.makedirs(out_dir, exist_ok=True)

        with open(target_path, "w", encoding="utf-8") as f:
            f.write(target_pdb)
            f.write("\n")

        designs = []
        for design_id in range(num_designs):
            pdb_path = rfdiffusion_runner.design_binder(target_path, out_dir, design_id)
            try:
                with open(pdb_path, "r", encoding="utf-8") as f:
                    pdb_text = f.read()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to read design {design_id} output: {e}")

            designs.append({"design_id": design_id, "pdb": pdb_text})

    return {
        "backend": "arm64-native",
        "designs": designs,
    }
