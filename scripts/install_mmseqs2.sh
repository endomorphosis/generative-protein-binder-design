#!/usr/bin/env bash
# Zero-touch MMseqs2 installer + optional DB builder + GPU server setup.
#
# Installs MMseqs2 into an existing conda environment (default: alphafold2)
# and optionally builds an MMseqs2 searchable DB from an AlphaFold reduced/full
# FASTA database (default: UniRef90) so AlphaFold can use --msa_mode=mmseqs2.
#
# NEW: Automatically detects and configures GPU support via GPU server mode.

set -euo pipefail

# Color codes (match other installers)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

CONDA_ENV="${ALPHAFOLD_CONDA_ENV:-alphafold2}"
DATA_DIR="${ALPHAFOLD_DATA_DIR:-$HOME/.cache/alphafold}"
DB_TIER="${ALPHAFOLD_DB_TIER:-auto}"

DO_INSTALL=true
DO_BUILD_DB=false
PRINT_ENV=false
FORCE=false

# Optional overrides for DB build
DB_FASTA_PATH="${ALPHAFOLD_MMSEQS2_DB_FASTA_PATH:-}"
DB_PREFIX="${ALPHAFOLD_MMSEQS2_DB_PREFIX:-}"
DB_THREADS="${ALPHAFOLD_MMSEQS2_DB_THREADS:-}"
DB_TMPDIR="${ALPHAFOLD_MMSEQS2_DB_TMPDIR:-}"

show_help() {
  cat <<'EOF'
Install MMseqs2 (and optionally build an MMseqs2 DB) for AlphaFold.

Usage:
  ./scripts/install_mmseqs2.sh [OPTIONS]

Options:
  --conda-env NAME        Conda env to install mmseqs2 into (default: alphafold2)
  --data-dir PATH         AlphaFold data dir (default: $ALPHAFOLD_DATA_DIR or ~/.cache/alphafold)
  --db-tier TIER          minimal|reduced|full|auto (default: auto)
  --install-only          Only install mmseqs2 into the conda env
  --build-db              Build an MMseqs2 DB from reduced/full FASTA
  --db-fasta PATH         FASTA to build the DB from (default: auto-detect, prefer UniRef90)
  --db-prefix PREFIX      Output DB prefix (default: $DATA_DIR/mmseqs2/uniref90_db)
  --tmp-dir PATH          Temp directory for createindex (default: $DATA_DIR/mmseqs2/tmp_mmseqs)
  --threads N             Threads for createindex (default: nproc capped at 32)
  --force                 Rebuild DB even if it already exists
  --print-env             Print shell exports for ALPHAFOLD_MMSEQS2_* (for eval)
  --help                  Show help

Examples:
  # Install mmseqs2 into alphafold2 env
  ./scripts/install_mmseqs2.sh --conda-env alphafold2 --install-only

  # Build a reduced DB prefix from UniRef90 (after AlphaFold reduced DB install)
  ./scripts/install_mmseqs2.sh --conda-env alphafold2 --data-dir ~/.cache/alphafold --db-tier reduced --build-db

  # Capture exports for use in another script
  eval "$(./scripts/install_mmseqs2.sh --build-db --print-env)"
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --conda-env)
      CONDA_ENV="${2:?missing value}"; shift 2 ;;
    --data-dir)
      DATA_DIR="${2:?missing value}"; shift 2 ;;
    --db-tier)
      DB_TIER="${2:?missing value}"; shift 2 ;;
    --install-only)
      DO_INSTALL=true
      DO_BUILD_DB=false
      shift
      ;;
    --build-db)
      DO_BUILD_DB=true
      shift
      ;;
    --db-fasta)
      DB_FASTA_PATH="${2:?missing value}"; shift 2 ;;
    --db-prefix)
      DB_PREFIX="${2:?missing value}"; shift 2 ;;
    --tmp-dir)
      DB_TMPDIR="${2:?missing value}"; shift 2 ;;
    --threads)
      DB_THREADS="${2:?missing value}"; shift 2 ;;
    --force)
      FORCE=true
      shift
      ;;
    --print-env)
      PRINT_ENV=true
      shift
      ;;
    --help|-h)
      show_help
      exit 0
      ;;
    *)
      log_error "Unknown option: $1"
      show_help
      exit 2
      ;;
  esac
done

if [[ "$DB_TIER" == "auto" ]]; then
  if [[ -f "$DATA_DIR/.tier" ]]; then
    # The alphafold installer writes either "tier=reduced" or "tier=full".
    # Accept either key=value or raw strings.
    tier_line="$(head -n 1 "$DATA_DIR/.tier" | tr -d '\r\n' || true)"
    tier_line="${tier_line#tier=}"
    if [[ "$tier_line" =~ ^(minimal|reduced|full)$ ]]; then
      DB_TIER="$tier_line"
    else
      DB_TIER="reduced"
    fi
  else
    DB_TIER="reduced"
  fi
fi

if ! command -v conda >/dev/null 2>&1; then
  log_error "conda not found. Run the AlphaFold zero-touch installer first (scripts/install_alphafold2_complete.sh)."
  exit 1
fi

# Initialize conda in this script.
eval "$(conda shell.bash hook)"

if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  log_error "Conda env '$CONDA_ENV' does not exist. Install AlphaFold first or pass --conda-env." 
  exit 1
fi

conda activate "$CONDA_ENV"

if [[ "$DO_INSTALL" == true ]]; then
  if command -v mmseqs >/dev/null 2>&1; then
    log_info "MMseqs2 already on PATH in env '$CONDA_ENV': $(command -v mmseqs)"
  else
    if command -v mamba >/dev/null 2>&1; then
      log_info "Installing MMseqs2 into env '$CONDA_ENV' via mamba..."
      mamba install -y -q -c bioconda -c conda-forge mmseqs2
    else
      log_info "Installing MMseqs2 into env '$CONDA_ENV' via conda..."
      conda install -y -q -c bioconda -c conda-forge mmseqs2
    fi
  fi
fi

if [[ "$DO_BUILD_DB" == true ]]; then
  if [[ "$DB_TIER" == "minimal" ]]; then
    log_warning "DB tier is minimal; no FASTA DBs are expected. Skipping MMseqs2 DB build."
  else
    if ! command -v mmseqs >/dev/null 2>&1; then
      log_error "mmseqs not found on PATH after install step; cannot build DB."
      exit 1
    fi

    fasta_path="$DB_FASTA_PATH"
    if [[ -z "$fasta_path" ]]; then
      if [[ -f "$DATA_DIR/uniref90/uniref90.fasta" ]]; then
        fasta_path="$DATA_DIR/uniref90/uniref90.fasta"
      elif [[ -f "$DATA_DIR/small_bfd/bfd-first_non_consensus_sequences.fasta" ]]; then
        fasta_path="$DATA_DIR/small_bfd/bfd-first_non_consensus_sequences.fasta"
      elif [[ -f "$DATA_DIR/mgnify/mgy_clusters_2022_05.fa" ]]; then
        fasta_path="$DATA_DIR/mgnify/mgy_clusters_2022_05.fa"
      fi
    fi

    if [[ -z "$fasta_path" || ! -f "$fasta_path" ]]; then
      log_error "Could not find FASTA for DB build. Pass --db-fasta or ensure AlphaFold reduced/full DBs are installed in: $DATA_DIR"
      exit 1
    fi

    db_prefix="$DB_PREFIX"
    if [[ -z "$db_prefix" ]]; then
      db_prefix="$DATA_DIR/mmseqs2/uniref90_db"
    fi

    tmp_dir="$DB_TMPDIR"
    if [[ -z "$tmp_dir" ]]; then
      tmp_dir="$(dirname "$db_prefix")/tmp_mmseqs"
    fi

    threads="$DB_THREADS"
    if [[ -z "$threads" ]]; then
      if command -v nproc >/dev/null 2>&1; then
        threads="$(nproc)"
      else
        threads=16
      fi
      if [[ "$threads" -gt 32 ]]; then
        threads=32
      fi
    fi

    # Check available disk space before building (rough heuristic: need ~2x FASTA size)
    fasta_size_mb="$(du -m "$fasta_path" 2>/dev/null | awk '{print $1}' || echo 0)"
    required_mb="$((fasta_size_mb * 2))"
    tmp_parent="$(dirname "$tmp_dir")"
    mkdir -p "$tmp_parent" 2>/dev/null || true
    available_mb="$(df -BM "$tmp_parent" 2>/dev/null | awk 'NR==2 {print $4}' | sed 's/M//' || echo 999999)"
    if [[ "$available_mb" -lt "$required_mb" ]]; then
      log_error "Insufficient disk space for MMseqs2 DB build:"
      log_error "  FASTA size: ${fasta_size_mb}MB"
      log_error "  Estimated requirement: ${required_mb}MB"
      log_error "  Available on $(df "$tmp_parent" | awk 'NR==2 {print $1}'): ${available_mb}MB"
      log_error "Free up space or override with --tmp-dir pointing to a larger partition."
      exit 1
    fi

    if [[ "$FORCE" == true ]]; then
      log_warning "Force rebuild requested; removing existing DB files for prefix: $db_prefix"
      db_dir="$(dirname "$db_prefix")"
      db_base="$(basename "$db_prefix")"
      if [[ -n "$db_dir" && -n "$db_base" && -d "$db_dir" ]]; then
        rm -f "$db_dir/$db_base"* || true
      fi
    fi

    if [[ -f "${db_prefix}.dbtype" || -f "${db_prefix}.index" || -f "${db_prefix}.0" ]]; then
      log_success "MMseqs2 DB already exists: $db_prefix"
    else
      log_info "Building MMseqs2 DB (threads=$threads, tmp=$tmp_dir)"
      log_info "  FASTA: $fasta_path"
      log_info "  OUT:   $db_prefix"
      bash "$ROOT_DIR/scripts/build_mmseqs_db.sh" "$fasta_path" "$db_prefix" --threads "$threads" --tmp-dir "$tmp_dir"
      log_success "MMseqs2 DB ready: $db_prefix"
    fi

    export ALPHAFOLD_MMSEQS2_DATABASE_PATH="$db_prefix"
  fi
fi

mmseqs_path="$(command -v mmseqs 2>/dev/null || true)"

# GPU Support Detection and Setup
GPU_SUPPORT_AVAILABLE=false
GPU_SERVER_SETUP=false

if [[ -n "$mmseqs_path" ]]; then
  # Check if MMseqs2 has GPU support
  if mmseqs search --help 2>&1 | grep -q "\-\-gpu"; then
    GPU_SUPPORT_AVAILABLE=true
    log_info "MMseqs2 GPU support detected"
    
    # Check if GPU is available
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
      log_info "NVIDIA GPU detected"
      
      # Auto-setup GPU server mode if database was built
      if [[ -n "${ALPHAFOLD_MMSEQS2_DATABASE_PATH:-}" ]] && [[ -f "${ALPHAFOLD_MMSEQS2_DATABASE_PATH}.dbtype" ]]; then
        log_info "Setting up MMseqs2 GPU server mode..."
        
        # Create GPU server run script
        GPU_SERVER_SCRIPT="$HOME/.local/bin/mmseqs2-gpu-server"
        mkdir -p "$(dirname "$GPU_SERVER_SCRIPT")"
        
        cat > "$GPU_SERVER_SCRIPT" << EOFGPU
#!/bin/bash
# MMseqs2 GPU Server (auto-generated by install_mmseqs2.sh)
# Start with: nohup $GPU_SERVER_SCRIPT &

LOG_FILE="\$HOME/.cache/mmseqs2-gpu-server.log"
echo "\$(date): Starting MMseqs2 GPU Server..." >> "\$LOG_FILE"
echo "Database: ${ALPHAFOLD_MMSEQS2_DATABASE_PATH}" >> "\$LOG_FILE"

$mmseqs_path gpuserver ${ALPHAFOLD_MMSEQS2_DATABASE_PATH} >> "\$LOG_FILE" 2>&1
EOFGPU
        
        chmod +x "$GPU_SERVER_SCRIPT"
        
        # Create systemd service file (user can install with sudo)
        SYSTEMD_FILE="$HOME/.local/share/mmseqs2-gpu-server.service"
        mkdir -p "$(dirname "$SYSTEMD_FILE")"
        
        cat > "$SYSTEMD_FILE" << EOFSYS
[Unit]
Description=MMseqs2 GPU Server
Documentation=https://github.com/soedinglab/MMseqs2
After=network.target

[Service]
Type=simple
User=$(whoami)
Environment="PATH=$PATH"
ExecStart=$GPU_SERVER_SCRIPT
Restart=always
RestartSec=10
StandardOutput=append:$HOME/.cache/mmseqs2-gpu-server.log
StandardError=append:$HOME/.cache/mmseqs2-gpu-server-error.log
LimitNOFILE=65536
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
EOFSYS
        
        GPU_SERVER_SETUP=true
        log_success "GPU server scripts created"
      fi
    fi
  fi
fi

if [[ "$PRINT_ENV" == true ]]; then
  if [[ -n "$mmseqs_path" ]]; then
    printf 'export ALPHAFOLD_MMSEQS2_BINARY_PATH=%q\n' "$mmseqs_path"
  else
    printf 'export ALPHAFOLD_MMSEQS2_BINARY_PATH=%q\n' ""
  fi
  printf 'export ALPHAFOLD_MMSEQS2_DATABASE_PATH=%q\n' "${ALPHAFOLD_MMSEQS2_DATABASE_PATH:-}"
  printf 'export ALPHAFOLD_MMSEQS2_GPU_SUPPORT=%q\n' "$GPU_SUPPORT_AVAILABLE"
  printf 'export ALPHAFOLD_MMSEQS2_USE_GPU_SERVER=%q\n' "$GPU_SERVER_SETUP"
fi

if [[ "$PRINT_ENV" == false ]]; then
  if [[ -n "$mmseqs_path" ]]; then
    log_success "MMseqs2 ready: $mmseqs_path"
  else
    log_warning "MMseqs2 not found on PATH (env '$CONDA_ENV')"
  fi
  if [[ -n "${ALPHAFOLD_MMSEQS2_DATABASE_PATH:-}" ]]; then
    log_success "MMseqs2 DB prefix: $ALPHAFOLD_MMSEQS2_DATABASE_PATH"
  fi
  
  # Print GPU setup info
  if [[ "$GPU_SUPPORT_AVAILABLE" == true ]]; then
    log_success "GPU support available"
    if [[ "$GPU_SERVER_SETUP" == true ]]; then
      echo ""
      log_success "GPU server mode configured!"
      echo ""
      echo "To enable GPU acceleration:"
      echo "  1. Start GPU server (one-time):"
      echo "     nohup $HOME/.local/bin/mmseqs2-gpu-server &"
      echo ""
      echo "  2. Or install as systemd service (with sudo):"
      echo "     sudo cp $HOME/.local/share/mmseqs2-gpu-server.service /etc/systemd/system/"
      echo "     sudo systemctl enable mmseqs2-gpu-server"
      echo "     sudo systemctl start mmseqs2-gpu-server"
      echo ""
      echo "  3. Use GPU in searches:"
      echo "     mmseqs search query.db target.db result.db tmp/ --gpu-server 1"
      echo ""
      echo "Expected speedup: 5-10x faster than CPU-only mode"
    fi
  fi
fi
