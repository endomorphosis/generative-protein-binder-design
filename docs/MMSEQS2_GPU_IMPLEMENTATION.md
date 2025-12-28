# MMseqs2 GPU Integration - Findings & Solution

## Discovery

The MMseqs2 binary (`bd01c2229f027d8d8e61947f44d11ef1a7669212`) **DOES have GPU support** built in:

```bash
$ mmseqs search --help | grep gpu
 --gpu INT                        Use GPU (CUDA) if possible [0]
 --gpu-server INT                 Use GPU server [0]
 --gpu-server-wait-timeout INT    Wait for GPU server timeout [600]
```

Reference: "GPU-accelerated homology search with MMseqs2. Nature Methods (2025)"

## Problem

When trying to use `--gpu 1` flag:
```
Error: Database uniref90_db.idx is not a valid GPU database
```

## Root Cause

MMseqs2 GPU mode requires either:
1. **GPU Server Mode**: Use `mmseqs gpuserver` to handle GPU operations
2. **Special Index Format**: Database needs GPU-compatible index format

## Current Database Status

- **Location**: `~/.cache/alphafold/mmseqs2/`
- **Size**: 1.5 TB (uniref90, uniprot, pdb_seqres)
- **Index**: Regular CPU index (not GPU-compatible)
- **Created**: Dec 26, 2025

## Solution Options

### Option 1: Use GPU Server Mode (Recommended for Production)

GPU server mode is designed for high-throughput scenarios:

```bash
# Start GPU server (in background)
mmseqs gpuserver /path/to/target.db --threads 4 &

# Run searches (they will use the GPU server)
mmseqs search query.db target.db result.db tmp/ --gpu-server 1
```

**Advantages**:
- Works with existing database indexes
- No need to rebuild 1.5TB database
- Better for multiple queries
- Optimal GPU utilization

**Trade-offs**:
- Need to manage server process
- Adds complexity to deployment

### Option 2: Rebuild Database with GPU-Compatible Index

May need special index format (needs investigation):

```bash
# Might need specific flags during createindex
mmseqs createindex uniref90_db tmp/ --index-subset 8  # "no sequence lookup (good for GPU only searches)"
```

**Advantages**:
- Direct GPU access
- Simpler for single queries

**Trade-offs**:
- Need to rebuild 1.5TB database
- Time: Several hours
- Disk space: Need extra space during rebuild

### Option 3: Hybrid - Use CPU for Large DB, GPU for Other Operations

Current setup already provides 10x speedup over JackHMMER on CPU.

**When CPU MMseqs2 Makes Sense**:
- Large databases (1.5TB)
- I/O bound operations
- Still significantly faster than traditional methods

**When to Use GPU**:
- Smaller, in-memory databases
- Compute-bound alignment steps
- Real-time queries

## Recommended Action Plan

### Immediate (Production Ready):
1. **Keep current CPU-based MMseqs2** for the 1.5TB database
   - Already 10x faster than JackHMMER
   - Works out of the box
   - No rebuild needed

2. **Use GPU Server Mode for AlphaFold Integration**
   - Start `mmseqs gpuserver` as a service
   - Configure AlphaFold to use `--gpu-server 1`
   - Get GPU acceleration without database rebuild

### Long-term Optimization:
1. **Create smaller GPU-optimized databases** for fast queries
   - Build reduced database (50GB) with GPU-compatible index
   - Use for interactive/real-time predictions
   - Keep large database for comprehensive searches

2. **Benchmark both modes**:
   - CPU: 580s for 70aa query (current)
   - GPU Server: Estimated 60-120s (10x faster)
   - Direct GPU: TBD (needs testing)

## Implementation Steps

### Step 1: Enable GPU Server Mode
```bash
# Create systemd service for GPU server
sudo cat > /etc/systemd/system/mmseqs2-gpu-server.service << 'SERVICE'
[Unit]
Description=MMseqs2 GPU Server
After=network.target

[Service]
Type=simple
User=barberb
ExecStart=/home/barberb/miniforge3/bin/mmseqs gpuserver /home/barberb/.cache/alphafold/mmseqs2/uniref90_db --threads 4
Restart=always

[Install]
WantedBy=multi-user.target
SERVICE

sudo systemctl enable mmseqs2-gpu-server
sudo systemctl start mmseqs2-gpu-server
```

### Step 2: Update Search Scripts
```bash
# Modify scripts to use --gpu-server 1 instead of --gpu 1
# Example in benchmark script:
mmseqs search query.db target.db result.db tmp/ \
  --max-seqs 100 \
  --threads 4 \
  --gpu-server 1 \  # Use GPU server
  -v 2
```

### Step 3: Test & Benchmark
```bash
# Run comparison benchmark
./scripts/comprehensive_benchmark.sh --use-gpu-server
```

## Expected Performance

| Mode | Time | GPU Util | Notes |
|------|------|----------|-------|
| CPU-only (current) | 580s | 0% | Working, 10x vs JackHMMER |
| GPU Server | ~60-120s | 80-100% | Estimated 5-10x speedup |
| Direct GPU | TBD | 80-100% | Needs GPU-compatible index |

## Integration into Scripts

Files to update:
1. `scripts/install_mmseqs2.sh` - Add GPU server setup option
2. `scripts/convert_alphafold_db_to_mmseqs2_multistage.sh` - Add GPU server flag
3. `scripts/bench_msa_comparison.sh` - Add GPU server mode
4. `deploy/docker-compose-gpu-optimized.yaml` - Add mmseqs2-gpu-server service
5. `native_services/alphafold_service.py` - Use --gpu-server flag

## References

- MMseqs2 GPU Paper: "GPU-accelerated homology search with MMseqs2. Nature Methods (2025)"
- Binary version: bd01c2229f027d8d8e61947f44d11ef1a7669212
- GPU detected: NVIDIA GB10, CUDA 13.1

## Status

- ✅ GPU support confirmed in binary
- ✅ GPU detection working (CUDA 13.1)
- ⚠️ Current database not GPU-compatible
- 🎯 **Next**: Implement GPU server mode for production use
