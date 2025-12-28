# MMseqs2 GPU Setup - The Correct Way

**Date**: December 28, 2025  
**Status**: Definitive Guide  
**Goal**: Get GPU acceleration working properly

---

## Problem Identified ✅

**Root Cause**: The conda MMseqs2 package (`bioconda`) is **NOT compiled with CUDA support!**

### Evidence:
```
✓ GPU hardware present: NVIDIA GB10
✓ CUDA installed: Version 13.1
✓ MMseqs2 has --gpu flag: Yes (in help text)
✗ MMseqs2 linked with CUDA: NO!
✗ CUDA in LD_LIBRARY_PATH: NO!
```

**Result**: The `--gpu` flags exist but **don't actually work** because the binary has no CUDA libraries linked.

---

## Why This Happened

### Conda Package Limitations

The bioconda `mmseqs2` package was built for broad compatibility:
- **Generic build**: Works on CPU-only systems
- **No CUDA dependency**: Avoids CUDA installation requirement
- **Has stub flags**: `--gpu` flags present but non-functional
- **ARM64/aarch64**: Our architecture may not have GPU builds

**This is why**:
- ✓ Package installs easily
- ✓ Has `--gpu` flags in help
- ❌ GPU mode doesn't actually work
- ❌ No CUDA libraries linked

---

## The Correct Way Forward

### Option 1: Set LD_LIBRARY_PATH (Quick Test)

**If CUDA libraries exist but aren't linked**:

```bash
# Add CUDA to library path
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Test if this fixes linking
ldd $(which mmseqs) | grep cuda
```

**Expected**: Still won't work (binary needs to be compiled with CUDA)

### Option 2: Compile from Source (Recommended)

**Build MMseqs2 with CUDA support**:

```bash
# Clone repository
git clone https://github.com/soedinglab/MMseqs2.git
cd MMseqs2

# Get latest release
git checkout $(git describe --tags $(git rev-list --tags --max-count=1))

# Build with CUDA
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DHAVE_CUDA=1 \
      -DCMAKE_INSTALL_PREFIX=$HOME/.local \
      -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
      ..

# Compile (10-20 minutes)
make -j$(nproc)

# Install
make install
```

**Result**: GPU-enabled MMseqs2 at `~/.local/bin/mmseqs`

### Option 3: Use Docker (Alternative)

**Pre-built GPU-enabled image**:

```bash
# Pull official GPU image
docker pull soedinglab/mmseqs2-gpu:latest

# Run with GPU
docker run --gpus all soedinglab/mmseqs2-gpu:latest \
  mmseqs search query.db target.db result.db tmp/ --gpu 1
```

---

## Verification Steps

### After Installation/Compilation

```bash
# 1. Check binary location
which mmseqs

# 2. Verify CUDA linking
ldd $(which mmseqs) | grep cuda
# Should show: libcudart.so, libcublas.so, etc.

# 3. Test GPU flag
mmseqs search --help | grep "\-\-gpu"

# 4. Try a test search
mmseqs search query.db small_db.db result.db tmp/ --gpu 1
```

**Success criteria**:
- ✅ Binary shows CUDA libraries in `ldd` output
- ✅ `--gpu 1` flag works without errors
- ✅ GPU utilization visible in `nvidia-smi`

---

## Expected Performance

### Once GPU is Working

**Based on NVIDIA blog (177x faster than JackHMMER)**:

| Stage | CPU Time | GPU Time | Speedup |
|-------|----------|----------|---------|
| Prefilter | 500-550s | 5-30s | 20-100x |
| Alignment | 50-70s | 5-10s | 5-10x |
| **Total** | **580s** | **10-40s** | **15-60x** |

**For our 70aa query against 1.5TB database**:
- Current (CPU-only): 580-634 seconds
- Expected (GPU): 10-40 seconds
- **Speedup: 15-60x faster!**

---

## Current Status Assessment

### What We Have Now:
- ✅ GPU hardware (NVIDIA GB10)
- ✅ CUDA toolkit (13.1)
- ✅ MMseqs2 installed (from conda)
- ❌ MMseqs2 NOT GPU-enabled (no CUDA linking)
- ❌ Cannot use GPU acceleration currently

### What We Need:
- 🔄 GPU-enabled MMseqs2 binary
  - Either: Compile from source with `-DHAVE_CUDA=1`
  - Or: Find pre-compiled GPU build for ARM64
  - Or: Use Docker with GPU support

### Why CPU-Only is Still Good:
- ✓ Works reliably (580-634s)
- ✓ Already 10x faster than JackHMMER
- ✓ No setup complexity
- ✓ Handles 1.5TB database well

---

## Decision Matrix

### Should You Enable GPU?

| Factor | CPU-Only | GPU-Enabled |
|--------|----------|-------------|
| **Setup Time** | ✓ Already done | ⚠️ 20-30 min compile |
| **Reliability** | ✓ Proven working | ❓ Needs testing |
| **Performance** | 580-634s | 10-40s (expected) |
| **Complexity** | ✓ Simple | ⚠️ Requires compilation |
| **Maintenance** | ✓ Conda managed | ⚠️ Manual updates |

### Recommendation:

**For Production (Now)**:
- Keep CPU-only setup (already working well)
- Already 10x faster than JackHMMER
- Reliable and maintained

**For Optimization (Later)**:
- Compile GPU-enabled version
- Test thoroughly
- Benchmark against CPU baseline
- Deploy if 10x+ improvement confirmed

---

## Action Plan

### Immediate (Today):

1. **Test LD_LIBRARY_PATH** (5 minutes):
   ```bash
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ldd $(which mmseqs) | grep cuda
   ```
   Expected: Won't help (binary not compiled with CUDA)

2. **Document findings** (Done ✅):
   - Current setup is CPU-only
   - GPU requires recompilation
   - Performance is good without GPU

3. **Keep current setup** for production:
   - 580-634s is acceptable
   - 10x faster than JackHMMER
   - Reliable and working

### Short Term (This Week):

1. **Compile GPU-enabled MMseqs2**:
   - Use build script
   - Test with small database first
   - Benchmark against CPU baseline

2. **Verify performance**:
   - Measure actual speedup
   - Check GPU utilization
   - Validate results match CPU

3. **Update installer** (if beneficial):
   - Add GPU compilation option
   - Make it optional (not default)
   - Document requirements

### Long Term (Next Month):

1. **Monitor for ARM64 GPU builds**:
   - Check conda-forge updates
   - Check bioconda for GPU packages
   - May become available

2. **Consider Docker deployment**:
   - Pre-built GPU images
   - Easier to maintain
   - Better for reproducibility

---

## Summary

### The Truth:
- ❌ Current MMseqs2 is **NOT GPU-enabled** (no CUDA linking)
- ✅ CPU-only mode **works well** (580-634s, 10x vs JackHMMER)
- 🔄 GPU mode **requires recompilation** from source
- 🎯 GPU mode **could be 15-60x faster** (10-40s vs 580s)

### The Decision:
**Keep CPU-only for now, optionally compile GPU later**

Why:
1. Current performance is acceptable
2. GPU compilation is complex (ARM64 + CUDA 13.1)
3. Need to test/validate GPU build thoroughly
4. CPU-only is proven and reliable

### Next Steps:
1. ✅ Document current limitations
2. ⏭️ Create GPU compilation guide
3. ⏭️ Test GPU build when time permits
4. ⏭️ Deploy GPU if 10x+ improvement confirmed

---

**Status**: Root cause identified and documented  
**Current Setup**: CPU-only, working, acceptable performance  
**GPU Path**: Available but requires compilation effort  
**Recommendation**: Production ready as-is, GPU optimization optional
