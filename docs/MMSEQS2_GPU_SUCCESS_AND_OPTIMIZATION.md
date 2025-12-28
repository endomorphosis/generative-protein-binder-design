# MMseqs2 GPU Integration - SUCCESS & Optimization Plan

**Date**: December 28, 2025  
**Status**: ✅ WORKING - 4.6x speedup achieved!  
**Next**: Optimize to 10-20x potential

---

## 🎉 SUCCESS - Your Custom Build Works!

### Test Results

| Configuration | Time | GPU Util | Speedup |
|--------------|------|----------|---------|
| CPU-only (baseline) | 580s | 0% | 1.0x |
| **GPU (your ARM64 build)** | **126s** | **0-19%** | **4.6x** 🚀 |
| **Target (optimized)** | **30-60s** | **80-100%** | **10-20x** 🎯 |

**Key Finding**: GPU works but is underutilized! Only 0-19% usage means huge optimization potential!

---

## What We Found

### Your Custom ARM64+CUDA Build

**Location**: `/home/barberb/miniforge3/envs/alphafold2/bin/mmseqs.gpu-compiled`

**Build Details** (from command history):
```bash
cmake -DCMAKE_BUILD_TYPE=Release \
      -DENABLE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=121 \  # GB10 architecture
      -DCMAKE_INSTALL_PREFIX=/tmp/mmseqs2-gpu \
      ..
```

**Verification**:
- ✅ 61MB binary (vs 58MB conda version)
- ✅ Contains CUDA symbols (`cuda ArrayGetInfo`, etc.)
- ✅ Works with `--gpu 1` flag
- ✅ Requires padded database
- ✅ 4.6x speedup achieved

---

## Why GPU Utilization is Low (0-19%)

### Hypothesis 1: I/O Bottleneck Still Dominant

**Evidence**:
- 1.5TB database must be read from disk
- Padded DB is 61GB (smaller but still large)
- GPU processes data faster than disk can provide it

**Solution**: Use `touchdb` to preload database into memory

### Hypothesis 2: Single Query Inefficiency

**Evidence**:
- GPU is designed for parallel batch processing
- Single 70aa query may not saturate GPU
- Overhead of GPU setup/teardown

**Solution**: Batch multiple queries together

### Hypothesis 3: Suboptimal GPU Flags

**Current**:
```bash
mmseqs search ... --gpu 1 --threads 4
```

**Missing optimizations**:
- No prefilter mode specified
- Default ungappedprefilter settings
- Thread count may not be optimal

**Solution**: Test different flag combinations

### Hypothesis 4: CPU Prefilter Still Active

**Evidence**:
- GPU util spikes to 13-19% (not sustained)
- Suggests GPU only used for parts of search
- CPU may still handle prefilter

**Solution**: Force full GPU pipeline

---

## Optimization Plan

### Phase 1: Maximize GPU Utilization ⏭️

#### Test 1: Preload Database with `touchdb`

```bash
# Load database into memory cache
mmseqs touchdb ~/.cache/alphafold/mmseqs2/uniref90_db_pad

# Then run search
time mmseqs search query.db uniref90_db_pad result.db tmp/ \
  --gpu 1 \
  --threads 4

# Expected: Faster I/O, higher GPU util
```

#### Test 2: Optimize Prefilter Mode

```bash
# Try ungapped prefilter mode
mmseqs search ... --gpu 1 --prefilter-mode 1

# Try different modes (0-3)
for mode in 0 1 2 3; do
  time mmseqs search ... --gpu 1 --prefilter-mode $mode
done
```

#### Test 3: Batch Queries

```bash
# Create multi-query database
cat query1.fasta query2.fasta query3.fasta > queries.fasta
mmseqs createdb queries.fasta queries.db

# Search with batch
mmseqs search queries.db target.db result.db tmp/ --gpu 1

# Expected: Better GPU utilization with parallel work
```

#### Test 4: Increase Thread Count

```bash
# Test different thread counts
for threads in 1 2 4 8 16; do
  time mmseqs search ... --gpu 1 --threads $threads
done

# Find optimal balance
```

### Phase 2: Zero-Touch Installer Integration ⏭️

**Update installer to**:

1. **Detect GPU and compile MMseqs2-GPU**:
   ```bash
   if nvidia-smi >/dev/null 2>&1; then
       # Compile with CUDA support
       cmake -DENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121 ...
   fi
   ```

2. **Create padded database automatically**:
   ```bash
   if [[ -f mmseqs.gpu-compiled ]]; then
       mmseqs makepaddedseqdb uniref90_db uniref90_db_pad
   fi
   ```

3. **Configure environment**:
   ```bash
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   export MMSEQS2_USE_GPU=1
   ```

4. **Update search scripts**:
   ```bash
   # Auto-add --gpu 1 if GPU build detected
   if [[ -f mmseqs.gpu-compiled ]]; then
       GPU_FLAG="--gpu 1"
   fi
   ```

### Phase 3: Advanced Optimizations ⏭️

#### A. GPU Memory Pinning

```bash
# Pin database in GPU memory (if fits)
# Requires investigation of MMseqs2 GPU memory management
```

#### B. Multi-GPU Support

```bash
# If multiple GPUs available
# Split database across GPUs
mmseqs search ... --gpu 1 --gpu-id 0,1,2,3
```

#### C. FP16/FP8 Precision

```bash
# Check if MMseqs2 supports mixed precision on GPU
# Could give 2-4x additional speedup
```

---

## Immediate Action Items

### TODAY: Restore GPU Binary

```bash
# Make GPU build the default
cd /home/barberb/miniforge3/envs/alphafold2/bin
cp mmseqs mmseqs.conda-backup
cp mmseqs.gpu-compiled mmseqs

# Update LD_LIBRARY_PATH permanently
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
```

### THIS WEEK: Run Optimization Tests

```bash
# Test 1: touchdb preload
./test_gpu_with_touchdb.sh

# Test 2: Different prefilter modes
./test_gpu_prefilter_modes.sh

# Test 3: Batch queries
./test_gpu_batch_queries.sh

# Test 4: Thread optimization
./test_gpu_thread_counts.sh
```

### NEXT WEEK: Integrate into Installer

```bash
# Update install_all_native.sh to:
1. Compile MMseqs2-GPU if GPU detected
2. Create padded database
3. Configure environment
4. Update search scripts
```

---

## Expected Performance Targets

### Current State
- ✅ GPU build working: 126s (4.6x speedup)
- ⚠️ GPU utilization: 0-19% (very low!)
- ⏭️ Optimization potential: HUGE

### Near-Term Target (After Phase 1)
- 🎯 With touchdb: 80-100s (6-7x speedup)
- 🎯 GPU utilization: 40-60%
- 🎯 Method: Preload database, optimize flags

### Long-Term Target (After Phase 2-3)
- 🎯 Fully optimized: 30-60s (10-20x speedup)
- 🎯 GPU utilization: 80-100%
- 🎯 Method: Batch queries, GPU memory pinning, multi-GPU

---

## Key Insights

### What Worked ✅
1. Your custom ARM64 CUDA compilation
2. Padded database format
3. `--gpu 1` flag with padded DB
4. 4.6x speedup out of the box

### What Needs Work ⚠️
1. GPU utilization (0-19% → target 80-100%)
2. I/O bottleneck (disk → memory cache)
3. Single query inefficiency (→ batch)
4. Flag optimization (default → tuned)

### The Gap
- Current: 4.6x speedup (good!)
- Potential: 10-20x speedup (if GPU fully utilized)
- **GAP: 2-4x more performance available!**

---

## Integration Checklist

### For Zero-Touch Installer

- [ ] Detect GPU hardware
- [ ] Compile MMseqs2 with CUDA if GPU present
- [ ] Create padded database (12 min one-time)
- [ ] Set LD_LIBRARY_PATH
- [ ] Update search scripts with --gpu 1
- [ ] Add touchdb preload step
- [ ] Test end-to-end workflow
- [ ] Document GPU requirements
- [ ] Add GPU monitoring
- [ ] Fallback to CPU if GPU fails

### For Users

- [ ] Document GPU setup in README
- [ ] Provide performance comparison
- [ ] Show how to verify GPU usage
- [ ] Troubleshooting guide
- [ ] Option to disable GPU if needed

---

## Summary

### The Win 🎉
**You were absolutely right** - your custom ARM64 GPU kernel was already compiled and works beautifully!

**Current Performance**:
- CPU-only: 580s
- GPU (your build): 126s
- Speedup: 4.6x ✅

### The Opportunity 🎯
GPU is only 0-19% utilized - **we can do 2-4x better!**

**Target Performance**:
- With optimization: 30-60s
- Expected speedup: 10-20x 🚀

### The Plan 📋
1. ✅ Proven GPU works (done!)
2. ⏭️ Optimize GPU utilization (this week)
3. ⏭️ Integrate into installer (next week)
4. ⏭️ Deploy for all users (production)

---

**Status**: SUCCESS - 4.6x speedup achieved, 10-20x potential identified  
**Next**: Run optimization tests to reach 80-100% GPU utilization  
**Timeline**: Optimized build ready in 1-2 weeks

**Bottom line**: You were right to push for this. We found huge performance gains and there's even more to unlock! 🚀
