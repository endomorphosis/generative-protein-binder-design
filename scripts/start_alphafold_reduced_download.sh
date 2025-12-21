#!/usr/bin/env bash
set -euo pipefail

# Idempotent-ish reduced DB downloader for AlphaFold, using the official scripts
# but skipping steps that appear complete.
#
# Writes:
#   /tmp/alphafold_download_reduced.pid
#   /tmp/alphafold_download_reduced.log

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ALPHAFOLD_DIR="$ROOT_DIR/tools/alphafold2"
DATA_DIR="${ALPHAFOLD_DATA_DIR:-$HOME/.cache/alphafold}"

PID_FILE="/tmp/alphafold_download_reduced.pid"
LOG_FILE="/tmp/alphafold_download_reduced.log"

mkdir -p "$DATA_DIR"

run_step() {
  local name="$1"
  shift
  echo "[$(date -Is)] === $name ===" | tee -a "$LOG_FILE"
  "$@" 2>&1 | tee -a "$LOG_FILE"
}

run_step_allow_fail() {
  local name="$1"
  shift
  if ! run_step "$name" "$@"; then
    echo "[$(date -Is)] WARNING: step failed: $name" | tee -a "$LOG_FILE"
    return 1
  fi
  return 0
}

download_pdb70_robust() {
  local root_dir="$DATA_DIR/pdb70"
  local url="http://wwwuser.gwdg.de/~compbiol/data/hhsuite/databases/hhsuite_dbs/old-releases/pdb70_from_mmcif_200401.tar.gz"
  local tar_name
  tar_name="$(basename "$url")"

  mkdir -p "$root_dir"

  # HHsearch uses a database *prefix*; AlphaFold passes .../pdb70/pdb70 and HHsuite
  # looks for files like pdb70_hhm.ffdata/.ffindex under that prefix.
  if [[ -f "$root_dir/pdb70_hhm.ffdata" && -f "$root_dir/pdb70_hhm.ffindex" ]]; then
    echo "[$(date -Is)] ✓ already present, skipping: pdb70 ($root_dir/pdb70_*)" | tee -a "$LOG_FILE"
    return 0
  fi

  if command -v aria2c >/dev/null 2>&1; then
    # -c continues partial downloads; -x/-s improve throughput when mirrors allow.
    aria2c -c -x 8 -s 8 "$url" --dir="$root_dir" >>"$LOG_FILE" 2>&1
  else
    wget -c "$url" -O "$root_dir/$tar_name" >>"$LOG_FILE" 2>&1
  fi

  if [[ ! -f "$root_dir/$tar_name" ]]; then
    echo "ERR: pdb70 tarball missing after download" | tee -a "$LOG_FILE" >&2
    return 1
  fi

  tar --extract --file="$root_dir/$tar_name" --directory="$root_dir" >>"$LOG_FILE" 2>&1
  rm -f "$root_dir/$tar_name" >>"$LOG_FILE" 2>&1 || true

  [[ -f "$root_dir/pdb70_hhm.ffdata" && -f "$root_dir/pdb70_hhm.ffindex" ]] || {
    echo "ERR: pdb70 HHsearch db files not found after extract" | tee -a "$LOG_FILE" >&2
    return 1
  }
  return 0
}

download_uniref90_robust() {
  local root_dir="$DATA_DIR/uniref90"
  local url="https://ftp.ebi.ac.uk/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz"
  local gz_path="$root_dir/uniref90.fasta.gz"
  local out_path="$root_dir/uniref90.fasta"

  mkdir -p "$root_dir"

  if [[ -f "$out_path" ]]; then
    echo "[$(date -Is)] ✓ already present, skipping: uniref90 ($out_path)" | tee -a "$LOG_FILE"
    return 0
  fi

  if command -v aria2c >/dev/null 2>&1; then
    aria2c -c -x 8 -s 8 "$url" --dir="$root_dir" >>"$LOG_FILE" 2>&1
  else
    wget -c "$url" -O "$gz_path" >>"$LOG_FILE" 2>&1
  fi

  [[ -f "$gz_path" ]] || {
    echo "ERR: uniref90 archive missing after download" | tee -a "$LOG_FILE" >&2
    return 1
  }

  # Extract without relying on the official script (which doesn't resume).
  if command -v pigz >/dev/null 2>&1; then
    pigz -dc "$gz_path" >"${out_path}.tmp" 2>>"$LOG_FILE"
  else
    gzip -dc "$gz_path" >"${out_path}.tmp" 2>>"$LOG_FILE"
  fi
  mv -f "${out_path}.tmp" "$out_path"
  # Save space.
  rm -f "$gz_path" "${gz_path}.aria2" >>"$LOG_FILE" 2>&1 || true
  return 0
}

download_uniprot_robust() {
  local root_dir="$DATA_DIR/uniprot"
  local trembl_url="https://ftp.ebi.ac.uk/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.fasta.gz"
  local sprot_url="https://ftp.ebi.ac.uk/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz"
  local trembl_gz="$root_dir/uniprot_trembl.fasta.gz"
  local sprot_gz="$root_dir/uniprot_sprot.fasta.gz"
  local out_path="$root_dir/uniprot.fasta"

  mkdir -p "$root_dir"

  if [[ -f "$out_path" ]]; then
    echo "[$(date -Is)] ✓ already present, skipping: uniprot ($out_path)" | tee -a "$LOG_FILE"
    return 0
  fi

  if command -v aria2c >/dev/null 2>&1; then
    aria2c -c -x 8 -s 8 "$trembl_url" --dir="$root_dir" >>"$LOG_FILE" 2>&1
    aria2c -c -x 8 -s 8 "$sprot_url" --dir="$root_dir" >>"$LOG_FILE" 2>&1
  else
    wget -c "$trembl_url" -O "$trembl_gz" >>"$LOG_FILE" 2>&1
    wget -c "$sprot_url" -O "$sprot_gz" >>"$LOG_FILE" 2>&1
  fi

  [[ -f "$trembl_gz" && -f "$sprot_gz" ]] || {
    echo "ERR: uniprot archives missing after download" | tee -a "$LOG_FILE" >&2
    return 1
  }

  # Extract + merge (SwissProt first, then TrEMBL), without writing huge intermediate files.
  rm -f "${out_path}.tmp" 2>/dev/null || true
  if command -v pigz >/dev/null 2>&1; then
    pigz -dc "$sprot_gz" >"${out_path}.tmp" 2>>"$LOG_FILE"
    pigz -dc "$trembl_gz" >>"${out_path}.tmp" 2>>"$LOG_FILE"
  else
    gzip -dc "$sprot_gz" >"${out_path}.tmp" 2>>"$LOG_FILE"
    gzip -dc "$trembl_gz" >>"${out_path}.tmp" 2>>"$LOG_FILE"
  fi
  mv -f "${out_path}.tmp" "$out_path"

  # Save space.
  rm -f "$trembl_gz" "$sprot_gz" "${trembl_gz}.aria2" "${sprot_gz}.aria2" >>"$LOG_FILE" 2>&1 || true
  return 0
}

skip_if_exists() {
  local path="$1"
  local name="$2"
  if [[ -e "$path" ]]; then
    echo "[$(date -Is)] ✓ already present, skipping: $name ($path)" | tee -a "$LOG_FILE"
    return 0
  fi
  return 1
}

(
  echo $$ >"$PID_FILE"
  echo "[$(date -Is)] Starting reduced DB download into $DATA_DIR" | tee -a "$LOG_FILE"

  # small_bfd
  if ! skip_if_exists "$DATA_DIR/small_bfd/bfd-first_non_consensus_sequences.fasta" "small_bfd"; then
    run_step_allow_fail "download_small_bfd" bash "$ALPHAFOLD_DIR/scripts/download_small_bfd.sh" "$DATA_DIR" || true
  fi

  # mgnify
  if ! skip_if_exists "$DATA_DIR/mgnify/mgy_clusters_2022_05.fa" "mgnify"; then
    run_step_allow_fail "download_mgnify" bash "$ALPHAFOLD_DIR/scripts/download_mgnify.sh" "$DATA_DIR" || true
  fi

  # pdb70
  run_step_allow_fail "download_pdb70" download_pdb70_robust || true

  # pdb_mmcif (templates)
  if [[ -d "$DATA_DIR/pdb_mmcif/mmcif_files" && -f "$DATA_DIR/pdb_mmcif/obsolete.dat" ]]; then
    echo "[$(date -Is)] ✓ already present, skipping: pdb_mmcif" | tee -a "$LOG_FILE"
  else
    run_step_allow_fail "download_pdb_mmcif" bash "$ALPHAFOLD_DIR/scripts/download_pdb_mmcif.sh" "$DATA_DIR" || true
  fi

  # uniref30
  if [[ -f "$DATA_DIR/uniref30/UniRef30_2021_03_hhm.ffdata" && -f "$DATA_DIR/uniref30/UniRef30_2021_03_hhm.ffindex" ]]; then
    echo "[$(date -Is)] ✓ already present, skipping: uniref30 ($DATA_DIR/uniref30/UniRef30_2021_03_*)" | tee -a "$LOG_FILE"
  else
    run_step_allow_fail "download_uniref30" bash "$ALPHAFOLD_DIR/scripts/download_uniref30.sh" "$DATA_DIR" || true
  fi

  # uniref90
  if ! skip_if_exists "$DATA_DIR/uniref90/uniref90.fasta" "uniref90"; then
    run_step_allow_fail "download_uniref90" download_uniref90_robust || true
  fi

  # uniprot
  if ! skip_if_exists "$DATA_DIR/uniprot/uniprot.fasta" "uniprot"; then
    run_step_allow_fail "download_uniprot" download_uniprot_robust || true
  fi

  # pdb_seqres
  if ! skip_if_exists "$DATA_DIR/pdb_seqres/pdb_seqres.txt" "pdb_seqres"; then
    run_step_allow_fail "download_pdb_seqres" bash "$ALPHAFOLD_DIR/scripts/download_pdb_seqres.sh" "$DATA_DIR" || true
  fi

  echo "[$(date -Is)] Done." | tee -a "$LOG_FILE"
) 
