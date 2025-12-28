#!/usr/bin/env bash
# Multi-stage AlphaFold database conversion to MMseqs2 format
# Converts AlphaFold databases to MMseqs2 searchable databases

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step() { echo -e "${MAGENTA}[STEP]${NC} $1"; }
log_header() { echo -e "${MAGENTA}=== $1 ===${NC}"; }

TIER="${TIER:-reduced}"
DATA_DIR="${ALPHAFOLD_DATA_DIR:-$HOME/.cache/alphafold}"
OUTPUT_DIR="${ALPHAFOLD_MMSEQS2_OUTPUT_DIR:-$DATA_DIR/mmseqs2}"
GPU_ENABLED=false
SKIP_EXISTING=false

declare -A STAGE_CONFIG
STAGE_CONFIG[minimal]="uniref90"
STAGE_CONFIG[reduced]="uniref90 bfd"
STAGE_CONFIG[full]="uniref90 bfd pdb_seqres uniprot"

show_help() {
    cat << 'EOF'
Convert AlphaFold databases to MMseqs2 format

Usage:
  ./scripts/convert_alphafold_db_to_mmseqs2_multistage.sh [OPTIONS]

Options:
  --tier TIER              minimal|reduced|full (default: reduced)
  --data-dir PATH          AlphaFold data directory
  --output-dir PATH        Output directory for MMseqs2 databases
  --gpu                    Enable GPU acceleration
  --skip-existing          Skip existing databases
  --help                   Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tier) TIER="${2:?missing value}"; shift 2 ;;
        --data-dir) DATA_DIR="${2:?missing value}"; shift 2 ;;
        --output-dir) OUTPUT_DIR="${2:?missing value}"; shift 2 ;;
        --gpu) GPU_ENABLED=true; shift ;;
        --skip-existing) SKIP_EXISTING=true; shift ;;
        --help) show_help; exit 0 ;;
        *) log_error "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ ! " minimal reduced full " =~ " $TIER " ]]; then
    log_error "Invalid tier: $TIER"
    exit 1
fi

if [[ ! -d "$DATA_DIR" ]]; then
    log_error "Data directory not found: $DATA_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

check_dependencies() {
    if ! command -v mmseqs >/dev/null 2>&1; then
        log_error "mmseqs not found"
        exit 1
    fi
    
    if $GPU_ENABLED && ! command -v nvidia-smi >/dev/null 2>&1; then
        log_warning "GPU not available, using CPU"
        GPU_ENABLED=false
    fi
}

extract_fasta() {
    local db_name="$1"
    local output_fasta="$2"
    
    if [[ -f "$output_fasta" ]]; then
        log_info "FASTA exists: $output_fasta"
        return 0
    fi
    
    case "$db_name" in
        uniref90)
            if [[ ! -f "$DATA_DIR/uniref90/uniref90.fasta" ]]; then
                log_error "UniRef90 not found"
                return 1
            fi
            log_info "Copying UniRef90..."
            cp "$DATA_DIR/uniref90/uniref90.fasta" "$output_fasta"
            ;;
        bfd)
            if [[ ! -f "$DATA_DIR/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt" ]]; then
                log_warning "BFD not found"
                return 1
            fi
            log_info "Processing BFD..."
            cp "$DATA_DIR/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt" "$output_fasta"
            ;;
        pdb_seqres)
            if [[ ! -f "$DATA_DIR/pdb_seqres/pdb_seqres.txt" ]]; then
                log_warning "PDB SeqRes not found"
                return 1
            fi
            log_info "Converting PDB SeqRes..."
            cp "$DATA_DIR/pdb_seqres/pdb_seqres.txt" "$output_fasta"
            ;;
        uniprot)
            if [[ ! -f "$DATA_DIR/uniprot/uniprot.fasta" ]]; then
                log_warning "UniProt not found"
                return 1
            fi
            log_info "Copying UniProt..."
            cp "$DATA_DIR/uniprot/uniprot.fasta" "$output_fasta"
            ;;
        *)
            log_error "Unknown database: $db_name"
            return 1
            ;;
    esac
    
    if [[ -f "$output_fasta" ]]; then
        local size=$(du -h "$output_fasta" | cut -f1)
        log_success "FASTA ready: $size"
        return 0
    fi
    
    return 1
}

build_mmseqs_db() {
    local db_name="$1"
    local input_fasta="$2"
    local output_db="$3"
    
    if [[ -f "${output_db}.dbtype" ]]; then
        log_info "Database exists"
        if $SKIP_EXISTING; then
            return 0
        fi
    fi
    
    if [[ ! -f "$input_fasta" ]]; then
        log_error "FASTA not found: $input_fasta"
        return 1
    fi
    
    log_step "Building MMseqs2 for $db_name"
    
    local tmp_dir="${OUTPUT_DIR}/tmp_${db_name}_$$"
    mkdir -p "$tmp_dir"
    trap "rm -rf '$tmp_dir'" RETURN
    
    local threads=$(($(nproc) > 16 ? 16 : $(nproc)))
    
    log_info "Creating database..."
    mmseqs createdb "$input_fasta" "$output_db" || return 1
    
    log_info "Creating index (threads=$threads)..."
    mmseqs createindex "$output_db" "$tmp_dir" --threads "$threads" || return 1
    
    log_success "Database built: $output_db"
    return 0
}

convert_tier() {
    local tier="$1"
    read -r -a stages <<< "${STAGE_CONFIG[$tier]}"
    
    log_header "Converting tier: $tier"
    echo "Stages: ${stages[*]}"
    echo ""
    
    local stage_num=0
    for stage in "${stages[@]}"; do
        stage_num=$((stage_num + 1))
        log_step "Stage $stage_num: $stage"
        
        local fasta_file="${OUTPUT_DIR}/${stage}.fasta"
        local db_prefix="${OUTPUT_DIR}/${stage}_db"
        
        if ! extract_fasta "$stage" "$fasta_file"; then
            log_warning "Skipping $stage"
            continue
        fi
        
        if ! build_mmseqs_db "$stage" "$fasta_file" "$db_prefix"; then
            log_warning "Build failed for $stage"
            continue
        fi
        
        rm -f "$fasta_file"
        echo ""
    done
    
    log_success "Tier conversion complete"
}

update_env_file() {
    local env_file="${DATA_DIR}/.env.mmseqs2"
    
    log_info "Updating environment file"
    
    cat > "$env_file" << EOF
export ALPHAFOLD_MMSEQS2_DATABASE_PATH="$OUTPUT_DIR/uniref90_db"
export ALPHAFOLD_MMSEQS2_USE_GPU="$GPU_ENABLED"
EOF
    
    log_success "Environment configured"
}

main() {
    log_header "AlphaFold to MMseqs2 Conversion"
    echo "Tier: $TIER"
    echo "Data: $DATA_DIR"
    echo "Output: $OUTPUT_DIR"
    echo ""
    
    check_dependencies
    convert_tier "$TIER"
    update_env_file
    
    log_header "Complete"
    echo "Next: source $DATA_DIR/.env.mmseqs2"
    echo ""
}

main
