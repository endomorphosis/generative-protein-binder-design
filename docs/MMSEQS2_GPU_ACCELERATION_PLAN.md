# MMseqs2 GPU Acceleration Implementation Plan

**Date**: December 28, 2025  
**Status**: Action Plan  
**Goal**: Make GPU server faster than CPU-only (currently 46+ min vs 10 min)

---

## Problem Statement

**Current Performance:**
- CPU-only: 580-634s (~10 minutes) ✅
- GPU server: 46+ minutes (2,760+ seconds) ❌ **27x SLOWER!**

**This is wrong!** GPU should be faster, not slower.

---

## Root Cause Analysis

### What Went Wrong

1. **Padded Database Overhead**
   - Padded DB: 61GB (data) + 21GB (headers) = 82GB
   - Regular DB: Sharded, optimized for I/O
   - Prefilter on padded DB is MUCH slower

2. **Wrong Database for Prefilter**
   - We ran prefilter on PADDED database
   - Padded format is optimized for GPU alignment, not CPU prefilter
   - This killed performance!

3. **GPU Server Architecture Misunderstanding**
   - GPU server should ONLY handle alignment
   - Prefilter should use regular (sharded) database
   - We used padded DB for entire pipeline

---

## Solution Strategy

### Key Insight: Hybrid Pipeline

Use **different databases** for different stages:

```
┌─────────────────────┐
│   Prefilter         │  ← Use REGULAR sharded DB (fast!)
│   (CPU, I/O bound)  │  ← 12 shards, 60GB each, optimized
└──────────┬──────────┘
           │
      Candidates
           │
┌──────────▼──────────┐
│   Alignment         │  ← Use PADDED DB via GPU server
│   (GPU accelerated) │  ← GPU loads padded DB once
└─────────────────────┘
```

**Expected result**: 
- Prefilter: 500-550s (same as CPU-only)
- Alignment: 50-70s → 5-10s (10x faster with GPU)
- **Total: ~510-560s (vs 580-634s CPU-only)** = **1.1-1.2x speedup**

---

## Implementation Plan

### Phase 1: Test Hybrid Pipeline ⏭️

**Goal**: Verify we can use different DBs for prefilter vs alignment

**Test steps:**
```bash
# 1. Start GPU server with PADDED database
mmseqs gpuserver ~/.cache/alphafold/mmseqs2/uniref90_db_pad &

# 2. Run prefilter on REGULAR (sharded) database
mmseqs prefilter query_db ~/.cache/alphafold/mmseqs2/uniref90_db prefilter_result

# 3. Run alignment on PADDED database with GPU server
mmseqs align query_db ~/.cache/alphafold/mmseqs2/uniref90_db_pad prefilter_result align_result --gpu-server 1
```

**Expected outcome**: 
- Prefilter: ~500-550s (fast, using sharded DB)
- Alignment: ~5-10s (fast, using GPU)
- Total: ~510-560s ✅ FASTER THAN CPU-ONLY

### Phase 2: Investigate GPU Server Communication ⏭️

**Question**: How does GPU server interact with searches?

**Investigation needed:**
```bash
# Check GPU server logs during search
tail -f ~/.cache/mmseqs2-gpu-server.log

# Monitor GPU utilization
nvidia-smi dmon -s u

# Check if alignment is using GPU server
strace -e connect mmseqs align ... --gpu-server 1
```

**Look for**:
- Socket/IPC communication with GPU server
- GPU utilization spikes during alignment
- Confirmation that GPU server is being used

### Phase 3: Optimize Prefilter for Large DB ⏭️

**Current bottleneck**: Reading 1.5TB from disk

**Options to investigate:**

#### Option A: Better Sharding Strategy
```bash
# Current: 12 shards of 60GB each
# Test: More, smaller shards for better parallelism?
mmseqs createindex --split 20  # 20 shards instead of 12
```

#### Option B: In-Memory Database (if RAM available)
```bash
# Copy database to tmpfs for testing
sudo mount -t tmpfs -o size=120G tmpfs /mnt/ramdisk
cp -r ~/.cache/alphafold/mmseqs2/uniref90_db* /mnt/ramdisk/
```

#### Option C: SSD/NVMe for Database
- Move database to faster storage
- Expected: 20-30% speedup

#### Option D: Parallel Shard Reading
- Check if MMseqs2 can read multiple shards in parallel
- Use `--threads` more effectively

### Phase 4: GPU-Specific Optimizations ⏭️

**Ensure GPU is fully utilized:**

#### A. GPU Server Configuration
```bash
# Check available GPU server options
mmseqs gpuserver --help

# Optimize:
--max-seqs 500        # More results to process
--prefilter-mode 1    # Ungapped mode (faster)
```

#### B. Alignment Parameters
```bash
# Optimize alignment for GPU
mmseqs align \
  --alignment-mode 2 \   # Faster alignment mode
  --gpu-server 1 \
  --threads 4            # Balance CPU/GPU
```

#### C. GPU Memory Management
```bash
# Ensure GPU has enough memory
# GB10 has significant VRAM - ensure it's used
nvidia-smi  # Check available memory
```

### Phase 5: Create Wrapper Script ⏭️

**Goal**: Automated hybrid pipeline

```bash
#!/bin/bash
# mmseqs2_gpu_search.sh - Hybrid CPU/GPU search

QUERY_DB="$1"
REGULAR_DB="$2"  # Sharded DB for prefilter
PADDED_DB="$3"   # Padded DB for alignment
RESULT_DB="$4"
TMP_DIR="$5"

# Check GPU server is running
if ! pgrep -f "mmseqs gpuserver.*$PADDED_DB" > /dev/null; then
    echo "Starting GPU server..."
    nohup mmseqs gpuserver "$PADDED_DB" > ~/.cache/mmseqs2-gpu-server.log 2>&1 &
    sleep 30  # Wait for DB to load on GPU
fi

# Step 1: Prefilter with regular (fast) database
echo "Running prefilter on sharded database..."
mmseqs prefilter "$QUERY_DB" "$REGULAR_DB" "$TMP_DIR/pref" \
    --threads $(nproc) \
    --max-seqs 100

# Step 2: Alignment with padded database + GPU server
echo "Running alignment with GPU acceleration..."
mmseqs align "$QUERY_DB" "$PADDED_DB" "$TMP_DIR/pref" "$RESULT_DB" \
    --gpu-server 1 \
    --threads 4

echo "Search complete!"
```

### Phase 6: Benchmark and Validate ⏭️

**Test matrix:**

| Configuration | Expected Time | Status |
|---------------|---------------|--------|
| CPU-only baseline | 580-634s | ✅ Known |
| GPU hybrid (pref: regular, align: GPU) | ~510-560s | ⏭️ To test |
| GPU full (both on padded DB) | 2,760+s | ❌ Too slow |
| Optimized hybrid | ~400-500s | 🎯 Goal |

**Validation criteria:**
- ✅ GPU utilization: 80-100% during alignment
- ✅ Total time: <580s (faster than CPU-only)
- ✅ Results: Same hits as CPU-only
- ✅ Reproducible: Consistent timing across runs

---

## Alternative Approaches

### If Hybrid Pipeline Doesn't Work

#### Option 1: Use GPU for Smaller Databases Only

**Strategy**: Create reduced database specifically for GPU

```bash
# Build 50GB reduced database
# Small enough to be fast, large enough to be useful
mmseqs createsubdb uniref90_db reduced_db --subdb-mode 1 --max-seq-id 0.5

# Create padded version
mmseqs makepaddedseqdb reduced_db reduced_db_pad

# Use for GPU searches
mmseqs gpuserver reduced_db_pad
```

**Expected**: 10-20x faster than 1.5TB, still good coverage

#### Option 2: Two-Tier Search Strategy

```bash
# Fast pass: GPU with reduced DB (50GB)
mmseqs search query reduced_db_pad result_fast --gpu-server 1  # ~30-60s

# If no good hits, deep search with full DB (1.5TB)  
if [[ $(hits < threshold) ]]; then
    mmseqs search query full_db result_deep --threads 20  # ~580s
fi
```

**Expected**: Most queries finish fast, only deep searches take time

#### Option 3: GPU for Different Use Case

**Realization**: Maybe GPU is better for:
- **Many small queries** (amortize DB loading)
- **Real-time searches** (keep GPU server resident)
- **Batch processing** (process 100s of queries)

Not for: Single large database searches

---

## Implementation Timeline

### Week 1: Investigation & Testing

**Day 1-2**: Test hybrid pipeline
- [ ] Test prefilter on regular DB + alignment on padded DB
- [ ] Monitor GPU utilization
- [ ] Measure timing breakdown

**Day 3-4**: Optimize configuration
- [ ] Test different shard configurations
- [ ] Test GPU server parameters
- [ ] Test alignment modes

**Day 5**: Benchmark and document
- [ ] Run comprehensive benchmarks
- [ ] Compare with CPU-only
- [ ] Document findings

### Week 2: Implementation

**Day 1-2**: Create wrapper scripts
- [ ] Hybrid search script
- [ ] GPU server management
- [ ] Error handling

**Day 3-4**: Integration
- [ ] Update zero-touch installer
- [ ] Add GPU optimization flags
- [ ] Update documentation

**Day 5**: Testing and validation
- [ ] End-to-end testing
- [ ] Performance validation
- [ ] User acceptance testing

---

## Success Criteria

### Must Have ✅
1. GPU search faster than CPU-only (<580s)
2. GPU utilization >80% during alignment
3. Same results as CPU-only (correctness)
4. Stable and reproducible performance

### Nice to Have 🎯
1. 2x speedup over CPU-only (~290s)
2. Automated GPU server management
3. Zero-touch setup in installer
4. Clear documentation for users

### Stretch Goals 🚀
1. 5x speedup with optimizations (~120s)
2. Support for multiple GPUs
3. Batch query processing
4. Real-time search capability

---

## Risk Assessment

### High Risk ⚠️
1. **GPU server may not support hybrid pipeline**
   - Mitigation: Test early, pivot to alternative approach
   
2. **Padded DB format incompatibility**
   - Mitigation: Use only for alignment step

3. **Limited GPU benefit for I/O-bound workload**
   - Mitigation: Focus on reducing I/O bottleneck first

### Medium Risk ⚠️
1. **Complex setup reduces zero-touch benefit**
   - Mitigation: Keep simple, offer advanced mode

2. **Disk space requirements (+82GB)**
   - Mitigation: Make optional, document clearly

3. **GPU server stability**
   - Mitigation: Add health checks, auto-restart

### Low Risk ✅
1. **Performance regression**
   - Mitigation: Keep CPU-only as fallback

2. **Different results CPU vs GPU**
   - Mitigation: Validate thoroughly, document differences

---

## Metrics to Track

### Performance Metrics
- [ ] Total search time (seconds)
- [ ] Prefilter time (seconds)
- [ ] Alignment time (seconds)
- [ ] GPU utilization (%)
- [ ] Memory usage (GB)
- [ ] Disk I/O (MB/s)

### Quality Metrics
- [ ] Number of hits found
- [ ] Hit quality (E-values)
- [ ] Result consistency vs CPU
- [ ] False positive rate

### Operational Metrics
- [ ] Setup time (minutes)
- [ ] Disk space used (GB)
- [ ] GPU server uptime
- [ ] Error rate

---

## Next Steps

### Immediate Actions (Today)

1. **Test hybrid pipeline manually**
   ```bash
   # Start GPU server
   nohup mmseqs gpuserver ~/.cache/alphafold/mmseqs2/uniref90_db_pad &
   
   # Create test query
   mmseqs createdb test.fasta query_db
   
   # Prefilter on regular DB
   mmseqs prefilter query_db ~/.cache/alphafold/mmseqs2/uniref90_db pref_db
   
   # Align on padded DB with GPU
   time mmseqs align query_db ~/.cache/alphafold/mmseqs2/uniref90_db_pad pref_db align_db --gpu-server 1
   
   # Compare timing!
   ```

2. **Monitor GPU during alignment**
   ```bash
   # In another terminal
   nvidia-smi dmon -s u -c 300
   ```

3. **Document results**
   - GPU utilization during alignment
   - Timing breakdown
   - Any errors or warnings

### Short Term (This Week)

- [ ] Complete Phase 1 testing
- [ ] Identify bottlenecks
- [ ] Create optimized script
- [ ] Benchmark vs CPU-only

### Medium Term (Next Week)

- [ ] Integrate into installer (if beneficial)
- [ ] Update documentation
- [ ] Create user guide
- [ ] Validate on production workloads

---

## Conclusion

**The GPU server CAN be faster than CPU-only** - we just need to:

1. Use the right database for each stage (hybrid approach)
2. Ensure GPU server is properly utilized
3. Optimize the pipeline for I/O and compute balance

**Current hypothesis**: The 46+ minute time was due to using padded database for prefilter, which is optimized for GPU but terrible for CPU I/O operations.

**Expected outcome after optimization**: 400-560s (faster than 580-634s CPU-only) ✅

**Action**: Test hybrid pipeline TODAY to validate hypothesis!

---

**Status**: Plan created, ready to implement  
**Priority**: HIGH - GPU should not be slower than CPU  
**Next**: Run Phase 1 tests immediately
