#!/usr/bin/env python3
"""ProteinMPNN runner.

This module originally shipped as a lightweight mock/fallback to keep the ARM64
stack bootable. For real-weight execution, we now try to run the actual
ProteinMPNN code (if present) and only allow mock outputs when explicitly
enabled (or in CI).
"""

from __future__ import annotations

import logging
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _truthy_env(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def allow_mock_outputs() -> bool:
    # Safety: mock outputs are for CI/testing only.
    return _truthy_env("CI")


def proteinmpnn_home() -> Optional[Path]:
    # Allow container or host to override where ProteinMPNN is located.
    env = (os.getenv("PROTEINMPNN_HOME") or "").strip()
    if env:
        p = Path(env)
        return p if p.exists() else None

    # Common layouts:
    # - copied into container at /app/ProteinMPNN
    # - repo workspace at tools/proteinmpnn/ProteinMPNN
    candidates = [Path("/app/ProteinMPNN"), Path(__file__).resolve().parents[2] / "tools" / "proteinmpnn" / "ProteinMPNN"]
    for c in candidates:
        if c.exists():
            return c
    return None


def is_ready() -> bool:
    home = proteinmpnn_home()
    if not home:
        return False

    script = home / "protein_mpnn_run.py"
    if not script.exists():
        return False

    weights = home / "vanilla_model_weights" / "v_48_020.pt"
    if not weights.exists():
        # Weights may differ, but absence usually means install incomplete.
        return False

    try:
        import torch  # noqa: F401
        import numpy  # noqa: F401
    except Exception:
        return False

    return True


def _parse_fasta_alignment_file(path: Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    header: Optional[str] = None
    seq_lines: List[str] = []

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq_lines)))
            header = line[1:].strip()
            seq_lines = []
        else:
            seq_lines.append(line)

    if header is not None:
        records.append((header, "".join(seq_lines)))
    return records


def _pick_designed_sequence(records: List[Tuple[str, str]]) -> Optional[str]:
    # Prefer sampled sequences (usually marked with T=...)
    for header, seq in records:
        if "T=" in header:
            return seq
    if records:
        return records[-1][1]
    return None


def generate_sequence(backbone_pdb_file: str) -> str:
    """Generate a sequence for a backbone PDB.

    In real mode, this runs ProteinMPNN and parses its output FASTA.
    In mock mode, returns a deterministic fallback.
    """
    if not is_ready():
        if allow_mock_outputs():
            logger.warning("ProteinMPNN real dependencies not available; returning mock sequence (CI enabled)")
            return generate_fallback_sequence()
        raise RuntimeError(
            "ProteinMPNN real execution is not available in this environment. "
            "Install ProteinMPNN + weights and required deps (torch, numpy). (Mock outputs are CI-only.)"
        )

    home = proteinmpnn_home()
    assert home is not None

    with tempfile.TemporaryDirectory(prefix="proteinmpnn_real_") as tmpdir:
        out_dir = Path(tmpdir)
        cmd = [
            sys.executable,
            str(home / "protein_mpnn_run.py"),
            "--pdb_path",
            backbone_pdb_file,
            "--out_folder",
            str(out_dir),
            "--num_seq_per_target",
            "1",
            "--batch_size",
            "1",
            "--sampling_temp",
            os.getenv("PROTEINMPNN_SAMPLING_TEMP", "0.1"),
            "--seed",
            os.getenv("PROTEINMPNN_SEED", "1"),
            "--model_name",
            os.getenv("PROTEINMPNN_MODEL_NAME", "v_48_020"),
            "--suppress_print",
            "1",
        ]

        logger.info("Running ProteinMPNN: %s", " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.error("ProteinMPNN stderr: %s", proc.stderr)
            raise RuntimeError(f"ProteinMPNN execution failed (exit {proc.returncode}).")

        seqs_dir = out_dir / "seqs"
        if not seqs_dir.exists():
            raise RuntimeError("ProteinMPNN did not produce a seqs/ output directory.")

        fasta_files = sorted(seqs_dir.glob("*.fa"))
        if not fasta_files:
            raise RuntimeError("ProteinMPNN did not produce any .fa outputs.")

        records = _parse_fasta_alignment_file(fasta_files[0])
        seq = _pick_designed_sequence(records)
        if not seq:
            raise RuntimeError("Failed to parse designed sequence from ProteinMPNN output.")

        # ProteinMPNN may use '/' between chains and wrap sequences; MCP expects a flat sequence.
        allowed = set("ACDEFGHIKLMNPQRSTVWYX")
        cleaned = "".join([c for c in seq.replace("/", "").upper() if c in allowed])
        if not cleaned:
            raise RuntimeError("ProteinMPNN returned an empty/invalid sequence after cleaning")
        return cleaned

def parse_backbone_structure(pdb_content: str) -> Dict[str, Any]:
    """Parse backbone structure from PDB content"""
    ca_coords = []
    residue_count = 0
    
    for line in pdb_content.split('\n'):
        if line.startswith('ATOM') and line[12:16].strip() == 'CA':  # Alpha carbon
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            ca_coords.append((x, y, z))
            residue_count += 1
    
    # Calculate secondary structure hints from coordinates
    secondary_structure = predict_secondary_structure(ca_coords)
    
    return {
        'length': residue_count,
        'ca_coords': ca_coords,
        'secondary_structure': secondary_structure
    }

def predict_secondary_structure(ca_coords):
    """Simple secondary structure prediction from CA coordinates"""
    if len(ca_coords) < 4:
        return ['C'] * len(ca_coords)  # All coil
    
    ss = []
    for i in range(len(ca_coords)):
        if i < 2 or i >= len(ca_coords) - 2:
            ss.append('C')  # Coil at ends
        else:
            # Simple distance-based prediction
            # Check i to i+3 distance for helix (should be ~5.4 Å)
            if i + 3 < len(ca_coords):
                dist = calculate_distance(ca_coords[i], ca_coords[i+3])
                if 4.5 <= dist <= 6.5:
                    ss.append('H')  # Helix
                elif dist > 10:
                    ss.append('E')  # Extended/sheet
                else:
                    ss.append('C')  # Coil
            else:
                ss.append('C')
    
    return ss

def calculate_distance(coord1, coord2):
    """Calculate Euclidean distance between two 3D coordinates"""
    return ((coord1[0] - coord2[0])**2 + 
            (coord1[1] - coord2[1])**2 + 
            (coord1[2] - coord2[2])**2)**0.5

def generate_mpnn_sequence(backbone_info: Dict[str, Any]) -> str:
    """Generate sequence using ProteinMPNN-inspired logic"""
    length = backbone_info['length']
    secondary_structure = backbone_info['secondary_structure']
    
    # Set seed for reproducible sequences
    random.seed(hash(str(backbone_info['ca_coords'][:5])) % 2**32)
    
    sequence = []
    
    # Amino acid preferences by secondary structure
    helix_preferred = ['A', 'E', 'K', 'L', 'R', 'Q']  # Alpha helix formers
    sheet_preferred = ['V', 'I', 'F', 'Y', 'T', 'W']  # Beta sheet formers  
    coil_preferred = ['G', 'S', 'P', 'N', 'D', 'H']   # Loop/coil formers
    
    all_amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                       'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    for i in range(length):
        ss = secondary_structure[i] if i < len(secondary_structure) else 'C'
        
        # Choose amino acid based on secondary structure preference
        if ss == 'H':  # Helix
            if random.random() < 0.7:  # 70% chance to use helix-preferred
                aa = random.choice(helix_preferred)
            else:
                aa = random.choice(all_amino_acids)
        elif ss == 'E':  # Sheet
            if random.random() < 0.7:
                aa = random.choice(sheet_preferred)
            else:
                aa = random.choice(all_amino_acids)
        else:  # Coil
            if random.random() < 0.6:
                aa = random.choice(coil_preferred)
            else:
                aa = random.choice(all_amino_acids)
        
        # Add some position-specific biases
        if i == 0:  # N-terminus
            aa = random.choice(['M', 'A', 'S', 'T'])  # Common N-terminal residues
        elif i == length - 1:  # C-terminus
            aa = random.choice(['A', 'G', 'S', 'L'])  # Common C-terminal residues
        
        # Avoid too many consecutive identical residues
        if len(sequence) >= 2 and sequence[-1] == sequence[-2] == aa:
            aa = random.choice(all_amino_acids)
        
        sequence.append(aa)
    
    return ''.join(sequence)

def generate_fallback_sequence() -> str:
    """Generate a basic fallback sequence"""
    # A reasonable protein sequence with good properties
    return "MKGSDKIHLTDDSFDITDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAKLNIDQNPGTAPKYGIRGIPTLLLFKNGEVAATKVGALSKGQLKEFLDANLA"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python proteinmpnn_runner.py <backbone_pdb>")
        sys.exit(1)
    
    backbone_pdb = sys.argv[1]
    
    result = generate_sequence(backbone_pdb)
    print(f"ProteinMPNN sequence generation completed: {result}")