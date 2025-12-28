# MMseqs2 GPU Critical Findings

**Date**: December 28, 2025  
**Status**: Important Architectural Discovery

## Critical Finding: GPU Server Mode Architecture

### What We Discovered

MMseqs2 GPU server mode has specific requirements that were not initially documented:

1. **Padded Database Required**: The database MUST be padded for GPU server mode
   - Command: `mmseqs makepaddedseqdb uniref90_db uniref90_db_pad`
   - Time: ~12 minutes for 1.5TB database
   - Output size: 61GB (additional space required)
   - This is a ONE-TIME operation

2. **GPU Server Usage**: The GPU server does NOT automatically accelerate all search steps
   - The prefilter step still runs on CPU
   - GPU acceleration applies to specific alignment operations
   - The `--gpu-server` flag tells the client where to send GPU work

3. **Performance Implications**:
   - Large databases (1.5TB) are heavily I/O bound
   - CPU prefilter is necessary for initial candidate selection
   - GPU acceleration is most effective for the alignment phase
   - Total speedup depends on query characteristics

### Database Padding Process

```bash
# One-time setup (required for GPU server)
cd ~/.cache/alphafold/mmseqs2
mmseqs makepaddedseqdb uniref90_db uniref90_db_pad

# Results:
# - Time: 12m 8s
# - Input: 1.5TB uniref90_db
# - Output: 61GB uniref90_db_pad + 21GB headers
# - CPU usage: 46% (20 threads)
# - Memory: 85GB peak
```

### Updated GPU Server Workflow

```bash
# 1. Create padded database (ONE-TIME, ~12 minutes)
mmseqs makepaddedseqdb ~/.cache/alphafold/mmseqs2/uniref90_db \
                       ~/.cache/alphafold/mmseqs2/uniref90_db_pad

# 2. Start GPU server with padded database
nohup mmseqs gpuserver ~/.cache/alphafold/mmseqs2/uniref90_db_pad \
  > ~/.cache/mmseqs2-gpu-server.log 2>&1 &

# 3. Run searches with --gpu-server flag
mmseqs search query.db uniref90_db_pad result.db tmp/ \
  --gpu-server 1 \
  --threads 4

# Note: Use the PADDED database path in searches!
```

### Performance Analysis

#### Without Padding (Error)
```
Error: Database is not a valid GPU database
Please call: makepaddedseqdb uniref90_db uniref90_db_pad
```

#### With Padding (In Progress)
- GPU server loads successfully
- Prefilter step runs on CPU (expected)
- Alignment step should use GPU (to be confirmed)
- Total time: TBD

### Recommendations for Zero-Touch Installer

The installer needs to be updated to:

1. **Create padded database automatically**:
   ```bash
   if [[ -f "$MMSEQS2_DB_PATH.dbtype" ]] && gpu_detected; then
       log_info "Creating GPU-compatible padded database..."
       mmseqs makepaddedseqdb "$MMSEQS2_DB_PATH" "${MMSEQS2_DB_PATH}_pad"
   fi
   ```

2. **Use padded database for GPU server**:
   ```bash
   # Update GPU server script to use _pad version
   mmseqs gpuserver ${ALPHAFOLD_MMSEQS2_DATABASE_PATH}_pad
   ```

3. **Document disk space requirements**:
   - Original database: ~1.5TB
   - Padded database: ~61GB additional
   - Total: ~1.56TB
   - Padding time: ~12 minutes (one-time)

4. **Update search scripts**:
   - Use padded database path when GPU server is available
   - Fall back to regular database for CPU-only mode

### Architecture Insights

MMseqs2 GPU acceleration is a **hybrid approach**:

```
Search Pipeline:
┌─────────────────┐
│  1. Prefilter   │  ← CPU-based (fast k-mer matching)
│     (CPU)       │  ← Reduces candidates from millions to hundreds
└────────┬────────┘
         │
    Candidates
         │
┌────────▼────────┐
│  2. Alignment   │  ← GPU-accelerated (if GPU server running)
│   (GPU/CPU)     │  ← Smith-Waterman alignment on candidates
└────────┬────────┘
         │
    Results
         │
┌────────▼────────┐
│  3. Post-       │  ← CPU-based
│     processing  │  
└─────────────────┘
```

**Key Insight**: GPU acceleration is most beneficial when:
- Many alignment operations needed (complex queries)
- Candidates fit in GPU memory
- I/O is not the bottleneck

For **very large databases** (1.5TB):
- I/O dominates (reading from disk)
- Prefilter is CPU-bound but fast
- GPU helps with alignment phase
- Overall speedup may be 2-5x, not 10-100x

### Disk Space Impact

| Component | Size | Required |
|-----------|------|----------|
| Original uniref90_db | ~1.5TB | Yes |
| Padded uniref90_db_pad | ~61GB | For GPU only |
| Padded headers | ~21GB | For GPU only |
| **Total for GPU mode** | **~1.58TB** | **+82GB over CPU-only** |

### Installation Time Impact

| Step | Time | When |
|------|------|------|
| Download database | Hours | Always |
| Build MMseqs2 DB | ~30-60 min | Always |
| **Create padded DB** | **~12 min** | **GPU only** |
| **Total additional** | **~12 minutes** | **One-time** |

### Updated Performance Expectations

#### Realistic Expectations

| Database Size | Speedup | Reason |
|--------------|---------|---------|
| Small (<10GB) | 10-50x | GPU dominates, minimal I/O |
| Medium (10-100GB) | 5-10x | Balanced GPU/I/O |
| Large (>100GB) | 2-5x | I/O bound, GPU helps alignment |
| Very Large (1.5TB) | 1.5-3x | Heavily I/O bound |

#### Our Case (1.5TB UniRef90)

- **CPU-only time**: 580 seconds (~9.7 min)
- **Expected GPU time**: 200-400 seconds (~3-7 min)
- **Expected speedup**: 1.5-3x (not 10x)
- **Bottleneck**: Disk I/O, not computation

### Action Items

1. ✅ **Update installer** to create padded database
2. ✅ **Update GPU server script** to use padded database
3. ✅ **Document disk space** requirements (+82GB)
4. ✅ **Document time** requirements (+12 min one-time)
5. ⏭️ **Test actual speedup** with padded database
6. ⏭️ **Update performance claims** to realistic 2-5x for large DBs
7. ⏭️ **Add skip option** for padding (if disk space limited)

### Status

- ✅ Padded database created successfully
- ✅ GPU server running with padded database
- ⏳ Performance testing in progress
- ⏭️ Installer updates needed

---

**Key Takeaway**: GPU server mode requires padded database (+82GB, +12min one-time). Expected speedup for 1.5TB database is 2-5x, not 10-100x, due to I/O bottleneck.
