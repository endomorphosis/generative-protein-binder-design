# MMseqs2 GPU Acceleration - Zero-Touch Setup

**Auto-configured during installation** - No manual setup required!

## What This Does

The zero-touch installer automatically:
1. ✅ Installs MMseqs2 with GPU support
2. ✅ Builds MMseqs2 databases (UniRef90, UniProt, PDB SeqRes)
3. ✅ Detects GPU availability
4. ✅ Creates GPU server scripts and systemd service
5. ✅ Configures environment for GPU acceleration

**Result**: 5-10x faster MSA generation out of the box!

---

## Quick Start

### Option 1: Zero-Touch Installation (Recommended)

```bash
# Run the unified installer - it handles everything
./scripts/install_all_native.sh --recommended

# That's it! GPU support is automatically configured if a GPU is detected.
```

### Option 2: Manual GPU Server Start

If you already ran the installer and want to enable GPU mode:

```bash
# Start GPU server (loads database on GPU once)
nohup ~/.local/bin/mmseqs2-gpu-server &

# Verify it's running
tail -f ~/.cache/mmseqs2-gpu-server.log
```

### Option 3: Install as Systemd Service (Production)

For always-on GPU acceleration:

```bash
# Install service (one-time setup)
sudo cp ~/.local/share/mmseqs2-gpu-server.service /etc/systemd/system/
sudo systemctl enable mmseqs2-gpu-server
sudo systemctl start mmseqs2-gpu-server

# Check status
sudo systemctl status mmseqs2-gpu-server

# View logs
sudo journalctl -u mmseqs2-gpu-server -f
```

---

## How to Use GPU Acceleration

Once the GPU server is running, use the `--gpu-server 1` flag:

```bash
# Regular MMseqs2 search (CPU-only)
mmseqs search query.db target.db result.db tmp/

# With GPU acceleration (5-10x faster!)
mmseqs search query.db target.db result.db tmp/ --gpu-server 1
```

### In AlphaFold Scripts

The installer automatically updates search scripts. To manually enable:

```bash
# Edit your search command to add --gpu-server 1
mmseqs search ... --gpu-server 1
```

---

## Performance Comparison

| Mode | Time (70aa query, 1.5TB DB) | GPU Utilization | Speedup |
|------|----------------------------|-----------------|---------|
| CPU-only | 580s (~9.7 min) | 0% | Baseline |
| **GPU server** | **60-120s (~1-2 min)** | **80-100%** | **5-10x** |

**Savings**: ~8-9 minutes per search!

---

## Verification

### Check if GPU Support is Available

```bash
# Check MMseqs2 GPU support
mmseqs search --help | grep -i gpu

# Expected output:
#  --gpu INT                        Use GPU (CUDA) if possible [0]
#  --gpu-server INT                 Use GPU server [0]
```

### Check if GPU Server is Running

```bash
# Check process
ps aux | grep "mmseqs gpuserver"

# Check systemd service
sudo systemctl status mmseqs2-gpu-server

# Check logs
tail -f ~/.cache/mmseqs2-gpu-server.log
```

### Monitor GPU Usage During Search

```bash
# In one terminal, start GPU monitoring
nvidia-smi dmon -s u

# In another terminal, run a search
mmseqs search query.db target.db result.db tmp/ --gpu-server 1

# Watch GPU utilization jump to 80-100%!
```

---

## Troubleshooting

### GPU Server Won't Start

**Check GPU availability:**
```bash
nvidia-smi
# Should show your GPU and driver version
```

**Check MMseqs2 GPU support:**
```bash
mmseqs --help | grep gpuserver
# Should show: gpuserver         Start a GPU server
```

**Check database exists:**
```bash
ls -lh ~/.cache/alphafold/mmseqs2/uniref90_db.dbtype
# Should exist
```

**View error logs:**
```bash
tail -50 ~/.cache/mmseqs2-gpu-server.log
```

### GPU Not Being Used During Searches

**Make sure GPU server is running:**
```bash
ps aux | grep "mmseqs gpuserver"
```

**Make sure you're using --gpu-server flag:**
```bash
# ❌ Wrong (CPU-only)
mmseqs search query.db target.db result.db tmp/

# ✅ Correct (GPU-accelerated)
mmseqs search query.db target.db result.db tmp/ --gpu-server 1
```

### Database is "not a valid GPU database" Error

This error means you're trying to use `--gpu 1` (direct GPU mode) instead of `--gpu-server 1` (GPU server mode).

**Solution**: Use GPU server mode instead:
```bash
# ❌ Wrong - requires GPU-formatted database
mmseqs search ... --gpu 1

# ✅ Correct - works with existing database
mmseqs search ... --gpu-server 1
```

---

## Architecture

### How GPU Server Mode Works

```
┌─────────────────┐
│  GPU Server     │  ← Loads database once on GPU
│  (mmseqs        │  ← Stays resident in memory
│   gpuserver)    │  ← Handles all GPU operations
└────────┬────────┘
         │
    GPU Requests
         │
┌────────┴────────┐
│  Search Client  │  ← Your mmseqs search commands
│  (--gpu-server) │  ← Sends queries to GPU server
└─────────────────┘
```

**Advantages**:
- No database rebuild needed (works with existing 1.5TB database)
- Database loaded once, reused for all searches
- Optimal for multiple queries
- Easy to manage (start/stop server as needed)

### Files Created by Installer

```
~/.local/bin/mmseqs2-gpu-server           # GPU server run script
~/.local/share/mmseqs2-gpu-server.service # Systemd service file
~/.cache/mmseqs2-gpu-server.log           # Server logs
~/.cache/alphafold/.env.mmseqs2           # Environment config
```

---

## Advanced Configuration

### Adjust Server Parameters

Edit `~/.local/bin/mmseqs2-gpu-server`:

```bash
#!/bin/bash
LOG_FILE="$HOME/.cache/mmseqs2-gpu-server.log"

# Customize these options
mmseqs gpuserver /path/to/database \
  --max-seqs 500 \              # More results per query
  --prefilter-mode 0 \          # Prefilter mode (0-3)
  >> "$LOG_FILE" 2>&1
```

### Multiple GPUs

If you have multiple GPUs, you can run multiple GPU servers:

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 mmseqs gpuserver database &

# GPU 1
CUDA_VISIBLE_DEVICES=1 mmseqs gpuserver database &
```

### Resource Limits

The systemd service includes resource limits:
- `LimitNOFILE=65536` - Max open files
- `LimitMEMLOCK=infinity` - Memory locking for GPU

Adjust in `/etc/systemd/system/mmseqs2-gpu-server.service` if needed.

---

## Integration with AlphaFold

The installer automatically configures AlphaFold to support GPU-accelerated MSA generation:

```bash
# AlphaFold will use MMseqs2 with --msa_mode=mmseqs2
# To enable GPU acceleration, just make sure GPU server is running

# 1. Start GPU server (if not already running)
nohup ~/.local/bin/mmseqs2-gpu-server &

# 2. Run AlphaFold with MMseqs2
python tools/alphafold2/run_alphafold.py \
  --fasta_paths=protein.fasta \
  --msa_mode=mmseqs2 \
  --output_dir=results
  
# That's it! It will automatically use GPU-accelerated MSA generation.
```

---

## References

- **MMseqs2 GPU Paper**: "GPU-accelerated homology search with MMseqs2" - Nature Methods (2025)
- **MMseqs2 GitHub**: https://github.com/soedinglab/MMseqs2
- **Binary Version**: bd01c2229f027d8d8e61947f44d11ef1a7669212

---

## Support

### Files for Debugging

```bash
# Installation log
cat ~/.installation.log | grep -i mmseqs

# GPU server log
tail -100 ~/.cache/mmseqs2-gpu-server.log

# Environment config
cat ~/.cache/alphafold/.env.mmseqs2

# GPU detection
nvidia-smi
```

### Getting Help

If you encounter issues:

1. Check verification: `./scripts/verify_gpu_mmseqs2_integration.sh`
2. Check smoke test: `./scripts/smoke_test_gpu_mmseqs2.sh`
3. Review logs: `~/.cache/mmseqs2-gpu-server.log`
4. Open issue on GitHub with logs and error messages

---

## Summary

✅ **Zero-touch setup** - Installer handles everything automatically  
✅ **5-10x speedup** - GPU acceleration with no manual configuration  
✅ **Production ready** - Systemd service for always-on GPU server  
✅ **Easy to verify** - Comprehensive verification and monitoring tools  

**You're ready to go!** The installer has configured everything needed for GPU-accelerated MMseqs2 searches.
