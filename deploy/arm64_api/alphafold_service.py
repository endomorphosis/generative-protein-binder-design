from __future__ import annotations

import os
import tempfile
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import alphafold_runner

app = FastAPI(title="AlphaFold2 ARM64 Service", version="0.1.0")


class StructureRequest(BaseModel):
    sequence: str = Field(..., description="Protein amino acid sequence")


@app.get("/v1/health/ready")
def health_ready():
    if not getattr(alphafold_runner, "is_ready", lambda: True)():
        raise HTTPException(
            status_code=503,
            detail="AlphaFold2 service not ready: real model/DBs not available (or mock outputs disabled)",
        )
    return {"status": "ready"}


@app.post("/v1/structure")
def structure(req: StructureRequest):
    sequence = (req.sequence or "").strip()
    if not sequence:
        raise HTTPException(status_code=400, detail="Missing sequence")

    with tempfile.TemporaryDirectory(prefix="alphafold2_") as tmpdir:
        fasta_path = os.path.join(tmpdir, "input.fasta")
        out_dir = os.path.join(tmpdir, "out")
        os.makedirs(out_dir, exist_ok=True)

        with open(fasta_path, "w", encoding="utf-8") as f:
            f.write(">query\n")
            f.write(sequence)
            f.write("\n")

        pdb_path = alphafold_runner.predict_structure(fasta_path, out_dir)
        try:
            with open(pdb_path, "r", encoding="utf-8") as f:
                pdb_text = f.read()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to read PDB output: {e}")

    return {
        "backend": "arm64-native",
        "pdb": pdb_text,
        "sequence": sequence,
    }
