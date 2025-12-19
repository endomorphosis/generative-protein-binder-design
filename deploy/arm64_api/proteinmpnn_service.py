from __future__ import annotations

import os
import tempfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import proteinmpnn_runner

app = FastAPI(title="ProteinMPNN ARM64 Service", version="0.1.0")


class SequenceRequest(BaseModel):
    backbone_pdb: str = Field(..., description="Backbone PDB content")


@app.get("/v1/health/ready")
def health_ready():
    if not getattr(proteinmpnn_runner, "is_ready", lambda: True)():
        # Use 503 to align with typical readiness semantics.
        raise HTTPException(
            status_code=503,
            detail="ProteinMPNN service not ready: real model weights/dependencies not available",
        )
    return {"status": "ready"}


@app.post("/v1/sequence")
def sequence(req: SequenceRequest):
    backbone_pdb = (req.backbone_pdb or "").strip()
    if not backbone_pdb:
        raise HTTPException(status_code=400, detail="Missing backbone_pdb")

    with tempfile.TemporaryDirectory(prefix="proteinmpnn_") as tmpdir:
        backbone_path = os.path.join(tmpdir, "backbone.pdb")
        with open(backbone_path, "w", encoding="utf-8") as f:
            f.write(backbone_pdb)
            f.write("\n")

        seq = proteinmpnn_runner.generate_sequence(backbone_path)
        if not seq:
            raise HTTPException(status_code=500, detail="Empty sequence result")

    return {
        "backend": "arm64-native",
        "sequence": seq,
    }
