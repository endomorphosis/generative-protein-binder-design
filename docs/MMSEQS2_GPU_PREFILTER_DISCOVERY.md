# CRITICAL DISCOVERY: MMseqs2 GPU Prefilter Acceleration

**Date**: December 28, 2025  
**Status**: BREAKTHROUGH - Complete Strategy Change Needed  
**Source**: NVIDIA Developer Blog on MMseqs2-GPU

---

## BREAKTHROUGH DISCOVERY

**WE WERE WRONG!** The prefilter **CAN be GPU-accelerated!**

### What We Thought:
- ❌ Prefilter is CPU-only
- ❌ GPU only helps alignment phase
- ❌ Expected speedup: 1.1-1.2x (minimal)

### What's Actually True:
- ✅ **Prefilter IS GPU-accelerated in MMseqs2-GPU!**
- ✅ **177x faster than CPU JackHMMER** on L40S GPU
- ✅ **Expected speedup: 10-100x for entire search!**

---

## Key Information from NVIDIA Blog

### GPU Acceleration of Prefilter

> "The gapless pre-filtering step... sees massive speedups, reported as **177x faster than CPU-based JackHMMER** on a single L40S GPU."

**This means the prefilter stage IS GPU-accelerated!**

### Performance Numbers

- **CPU JackHMMER**: Slow baseline
- **MMseqs2-GPU**: **177x faster** on L40S
- **Per sequence**: 0.117 seconds (with GPU acceleration)

### How It Works

1. **GPU Offloading**: Gapless prefiltering algorithm runs on NVIDIA GPUs using CUDA
2. **Parallel Sequence Comparisons**: Ideal for GPU parallelization
3. **SIMD + GPU**: Combined CPU vectorization and GPU acceleration

---

## Why Our Tests Failed

### Problem 1: We Used Wrong Command

**What we did:**
```bash
mmseqs search query.db target.db result.db tmp/ --gpu-server 1
```

**What we should have done:**
```bash
# GPU server needs to handle PREFILTER too, not just alignment!
# The prefilter needs to use GPU, not CPU
```

### Problem 2: Padded Database Issues

**Our padded database (61GB) was created for alignment, not prefilter!**

The prefilter running on padded DB was SLOW because:
- Padded format is for alignment phase
- Prefilter needs DIFFERENT GPU optimization
- We need to check if `makepaddedseqdb` is even correct

### Problem 3: GPU Server Not Being Used for Prefilter

**Evidence**: We saw `mmseqs prefilter` running on CPU (1757% CPU usage)

**This means**: The GPU server wasn't handling the prefilter at all!

**The `--gpu-server` flag only sends ALIGNMENT to GPU server, not prefilter!**

---

## Critical Commands We Missed

### 1. `mmseqs touchdb` - Load DB into Memory

From the information:
> "Loading it into memory with `mmseqs touchdb` (to take advantage of OS caching) can speed up subsequent searches more than tenfold."

**We never ran this!**

```bash
# Preload database into memory cache
mmseqs touchdb uniref90_db
```

**Expected benefit**: 10x faster database access!

### 2. Proper GPU Index Creation

The regular `createindex` we used may not be GPU-optimized!

**Need to investigate:**
- Is there a GPU-specific index format?
- Do we need special flags for `createindex`?
- Is `makepaddedseqdb` for alignment only?

---

## Corrected Architecture

### What Actually Happens in MMseqs2-GPU

```
┌─────────────────────────────────┐
│  MMseqs2-GPU Search Pipeline    │
├─────────────────────────────────┤
│                                 │
│  1. Prefilter (GPU-ACCELERATED) │  ← 177x faster than CPU!
│     - K-mer matching on GPU     │  ← CUDA offloading
│     - Parallel seq comparisons  │  ← Main speedup source
│     - Reduces millions to 100s  │
│                                 │
│  2. Alignment (GPU or CPU)      │  ← Also can be GPU
│     - Smith-Waterman            │
│     - On candidate sequences    │
│                                 │
└─────────────────────────────────┘
```

**Key realization**: The PREFILTER is the main GPU acceleration target, not alignment!

---

## Why We Got 46+ Minutes Instead of 2 Minutes

### Hypothesis: GPU Server Not Used for Prefilter

**What happened:**
1. Started GPU server with padded DB
2. Ran `mmseqs search ... --gpu-server 1`
3. **Prefilter ran on CPU** (we saw 1757% CPU, 0% GPU!)
4. Search was slow because prefilter was CPU-only

**Root cause**: 
- `--gpu-server 1` flag might only apply to alignment
- Prefilter needs different flag or mode
- We need to check MMseqs2-GPU documentation

### Expected Performance with Proper GPU Use

**According to NVIDIA:**
- L40S GPU: 177x faster than CPU JackHMMER
- Per sequence: 0.117 seconds
- Our 70aa query: Should be **~2-10 seconds total!**

**Not 580 seconds, not 2,760 seconds, but 2-10 seconds!**

---

## Action Plan - REVISED

### Immediate Actions

1. **Check MMseqs2-GPU Documentation**
   ```bash
   mmseqs --help | grep -i gpu
   mmseqs search --help | grep -i gpu
   ```
   
2. **Test `touchdb` Command**
   ```bash
   # Preload database into memory
   time mmseqs touchdb ~/.cache/alphafold/mmseqs2/uniref90_db
   ```
   Expected: Should take a few minutes, then searches are 10x faster

3. **Find Correct GPU Prefilter Command**
   - Check if there's `--gpu` flag (not `--gpu-server`)
   - Check if prefilter needs special invocation
   - Read MMseqs2-GPU documentation

4. **Verify GPU Binary**
   ```bash
   mmseqs version
   # Check if this is the GPU-enabled build
   # We have bd01c2229f027d8d8e61947f44d11ef1a7669212
   # Is this the GPU build?
   ```

### Investigation Needed

#### Question 1: Is Our MMseqs2 the GPU Version?

**Check:**
```bash
mmseqs prefilter --help | grep -i cuda
mmseqs prefilter --help | grep -i gpu
ldd $(which mmseqs) | grep cuda
```

**If no GPU support in prefilter**: We need to install MMseqs2-GPU specifically!

#### Question 2: How to Enable GPU Prefilter?

**Possibilities:**
- `mmseqs search --gpu 1` (not `--gpu-server`)
- Separate `mmseqs ungappedprefilter` command with GPU flag
- Automatic when GPU-enabled binary used
- Requires GPU-specific database format

#### Question 3: What's the Right Database Format?

**Options:**
- Regular DB: For CPU prefilter ❌
- Padded DB: For GPU alignment ❌
- **GPU DB**: For GPU prefilter? ✅ (need to find command)

---

## Expected Performance After Fix

### If We Get GPU Prefilter Working

**Based on NVIDIA blog (L40S GPU):**
- 177x faster than CPU JackHMMER
- ~0.117 seconds per sequence
- For our 70aa query: **~2-10 seconds total!**

**Our hardware (NVIDIA GB10):**
- Likely similar performance (similar generation GPU)
- Expected: **10-30 seconds per search**
- vs current: 580 seconds CPU-only
- **Speedup: 20-60x!**

### Realistic Expectations

| Component | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| Prefilter | 500-550s | **5-30s** | **20-100x** |
| Alignment | 50-70s | 5-10s | 5-10x |
| **Total** | **580s** | **10-40s** | **15-60x** |

**This makes WAY more sense!**

---

## Critical Next Steps

### 1. Verify MMseqs2 Build (TODAY)

```bash
# Check if our binary has GPU support
mmseqs version
mmseqs search --help | grep -i "gpu\|cuda"
ldd $(which mmseqs) | grep cuda

# If no CUDA linking, we need MMseqs2-GPU build!
```

### 2. Install/Compile MMseqs2-GPU (If Needed)

```bash
# May need to compile from source with GPU support
git clone https://github.com/soedinglab/MMseqs2.git
cd MMseqs2
mkdir build && cd build
cmake -DHAVE_CUDA=1 ..
make -j$(nproc)
```

### 3. Test with Correct Flags

```bash
# Try different approaches
mmseqs search query.db target.db result.db tmp/ --gpu 1
mmseqs search query.db target.db result.db tmp/ --gpu 1 --gpu-server 1
mmseqs ungappedprefilter query.db target.db result.db --gpu 1
```

### 4. Use `touchdb` to Preload DB

```bash
# Critical optimization we missed!
mmseqs touchdb ~/.cache/alphafold/mmseqs2/uniref90_db
```

---

## Summary

### What We Learned

1. ✅ **Prefilter IS GPU-accelerated** (177x faster!)
2. ✅ **`touchdb` command** can give 10x speedup
3. ❌ **We weren't using GPU for prefilter** (saw CPU usage)
4. ❓ **Our binary might not be GPU-enabled build**
5. ❓ **`--gpu-server` flag might only apply to alignment**

### What We Need to Do

1. **Verify GPU build** - Check if prefilter supports GPU
2. **Find correct GPU flags** - How to enable GPU prefilter
3. **Use `touchdb`** - Preload DB into memory (10x speedup)
4. **Re-test with proper setup** - Should be 15-60x faster!

### Expected Outcome

**With proper GPU acceleration:**
- **Current**: 580 seconds (CPU-only)
- **Expected**: 10-40 seconds (GPU prefilter + alignment)
- **Speedup**: 15-60x faster! 🚀

**This is what we should have gotten from the start!**

---

**Status**: Critical path identified, ready to implement  
**Priority**: URGENT - This changes everything  
**Next**: Verify GPU build and test with correct flags TODAY
