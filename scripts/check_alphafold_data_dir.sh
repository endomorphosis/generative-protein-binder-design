#!/usr/bin/env bash
set -euo pipefail

# Checks that AlphaFold databases are present in the *actual* on-disk format
# AlphaFold expects. This is meant to be run by non-experts.
#
# Usage:
#   ./scripts/check_alphafold_data_dir.sh [DATA_DIR]
#
# Defaults:
#   DATA_DIR=$ALPHAFOLD_DATA_DIR or ~/.cache/alphafold

DATA_DIR="${1:-${ALPHAFOLD_DATA_DIR:-$HOME/.cache/alphafold}}"

fail=0

say() { printf '%s\n' "$*"; }
ok() { say "✓ $*"; }
bad() { say "✗ $*"; fail=1; }

need_file() {
  local path="$1"
  local label="$2"
  if [[ -f "$path" ]]; then
    ok "$label: $path"
  else
    bad "$label missing: $path"
  fi
}

need_dir() {
  local path="$1"
  local label="$2"
  if [[ -d "$path" ]]; then
    ok "$label: $path"
  else
    bad "$label missing: $path"
  fi
}

say "AlphaFold data directory check"
say "DATA_DIR=$DATA_DIR"
say ""

need_dir "$DATA_DIR" "data dir"
need_dir "$DATA_DIR/params" "model params dir"

# Params: one file is enough to prove the tarball extract happened.
need_file "$DATA_DIR/params/params_model_1.npz" "params_model_1.npz"

# These are plain FASTA files.
need_file "$DATA_DIR/uniref90/uniref90.fasta" "UniRef90 FASTA"
need_file "$DATA_DIR/mgnify/mgy_clusters_2022_05.fa" "MGnify clusters"
need_file "$DATA_DIR/small_bfd/bfd-first_non_consensus_sequences.fasta" "Small BFD FASTA"
need_file "$DATA_DIR/uniprot/uniprot.fasta" "UniProt FASTA (multimer)"
need_file "$DATA_DIR/pdb_seqres/pdb_seqres.txt" "PDB SeqRes"

# Templates.
need_dir "$DATA_DIR/pdb_mmcif/mmcif_files" "PDB mmCIF templates"
need_file "$DATA_DIR/pdb_mmcif/obsolete.dat" "obsolete.dat"

# HH-suite databases are *prefix-based*.
# AlphaFold passes a prefix like .../pdb70/pdb70; HHsuite expects files like
# pdb70_hhm.ffdata/.ffindex under that directory.
need_file "$DATA_DIR/pdb70/pdb70_hhm.ffdata" "PDB70 HH-suite DB (hhm ffdata)"
need_file "$DATA_DIR/pdb70/pdb70_hhm.ffindex" "PDB70 HH-suite DB (hhm ffindex)"

need_file "$DATA_DIR/uniref30/UniRef30_2021_03_hhm.ffdata" "UniRef30 HH-suite DB (hhm ffdata)"
need_file "$DATA_DIR/uniref30/UniRef30_2021_03_hhm.ffindex" "UniRef30 HH-suite DB (hhm ffindex)"

say ""
if [[ "$fail" -eq 0 ]]; then
  say "OK: AlphaFold DBs look complete for reduced_dbs."
  say "Note: UniRef30/PDB70 are present as HH-suite DB files (ffdata/ffindex), which is expected."
  exit 0
fi

say "ERROR: Some AlphaFold DB files are missing."
say "If you're using the built-in provisioner, run:"
say "  ./scripts/start_alphafold_reduced_download.sh"
exit 2
