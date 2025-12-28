# MMseqs2 Optimization Strategy

## Problem Statement

The initial full UniRef90 MMseqs2 database (570GB) exceeds available RAM during search operations, causing OOM kills. Need to find optimal DB size that balances:
- Memory footprint (must fit in available ~113GB RAM)
- Search speed (target 5-10x faster than JackHMMER)
- MSA quality (sufficient sequence coverage for accurate structure prediction)

## Completed Steps

### 1. FASTA Subset Creation ✓
- Created 5M sequence subset from UniRef90 (~9.3GB FASTA)
- Memory-appropriate size: targeting ~30GB indexed DB
- Location: `~/.cache/alphafold/mmseqs2/uniref90_large_5000000.fasta`

### 2. MMseqs2 DB Build ✓
- Building indexed DB with `createindex` for fast searches
- Using 20 threads, data-dir relative temp directory
- Progress: ~79% complete (currently running)
- Estimated final size: ~30-40GB (fits comfortably in 113GB available RAM)

## Pending Steps

### 3. Single Search Benchmark
**Goal**: Verify the new DB works without OOM and measure baseline search speed

**Method**:
```bash
./scripts/rebuild_optimized_mmseqs2_db.sh
# Script includes built-in search benchmark at the end
```

**Expected outcome**:
- Search completes successfully (no OOM)
- Search time: 5-30 seconds for single 70-residue query
- Memory usage: <40GB peak

### 4. Full Pipeline Comparison
**Goal**: Measure end-to-end performance improvement vs JackHMMER

**Method**:
```bash
chmod +x scripts/bench_msa_comparison.sh
./scripts/bench_msa_comparison.sh
```

**Metrics to collect**:
- Total inference time (JackHMMER vs MMseqs2)
- MSA generation time breakdown
- GPU utilization % (expect increase with MMseqs2)
- MSA depth/coverage (validate quality)

**Success criteria**:
- 5-10x speedup in MSA generation
- GPU utilization increases from ~30-40% to 60-80%
- MSA quality comparable to JackHMMER output

### 5. Parameter Tuning (if needed)
**Variables to optimize**:
- `--max-seqs`: 128, 256, 512, 1024 (current default: 512)
- `--sensitivity`: 6.0 (faster), 7.5 (default), 9.0 (more sensitive)
- `--threads`: Match available CPUs (currently 20)

**Trade-offs**:
- Lower max-seqs = faster search, potentially lower quality
- Lower sensitivity = faster search, may miss remote homologs
- More threads = faster search, higher CPU load

### 6. Alternative DB Sizes (if 5M is suboptimal)

#### If 5M DB is too large (OOM):
Build smaller DBs:
- **Small**: 500K sequences (~5GB indexed)
- **Tiny**: 50K sequences (~500MB indexed)

#### If 5M DB gives poor MSA quality:
Consider:
- **Clustering**: Use `mmseqs cluster` to get diverse representatives instead of first N sequences
- **Quality filtering**: Sort by sequence coverage/identity before subsetting
- **Multiple DBs**: Small fast DB for initial pass, larger DB for hard queries

## Implementation Notes

### DB Subset Strategy
Current approach: Take first 5M sequences from UniRef90

**Pros**:
- Simple, fast to create
- Deterministic
- Preserves high-quality annotated sequences (often at top of UniRef90)

**Cons**:
- May not be representative of full diversity
- No quality filtering

**Future improvements** (if needed):
1. **Clustering-based**: 
   ```bash
   mmseqs cluster uniref90_db cluster tmp --min-seq-id 0.3
   mmseqs result2representative uniref90_db cluster cluster_rep
   # Take top 5M from representatives
   ```

2. **Coverage-based**:
   - Filter for sequences with >80% query coverage
   - Sort by e-value/bit score
   - Take top N

### Memory Estimation Formula

Rough estimate for MMseqs2 DB size:
```
Indexed_DB_size ≈ FASTA_size * 5-7x

For 5M sequences:
- FASTA: 9.3GB
- Indexed DB: ~30-65GB (within 113GB RAM limit)
```

For precomputed indices:
```
With k-mer index: DB_size * 1.5-2x additional
```

### Integration Points

After benchmarking confirms optimal configuration:

1. **Update zero-touch installer**:
   - Modify `scripts/install_mmseqs2.sh` to build optimized 5M DB by default
   - Add flag `--db-size={tiny|small|large}` for different presets

2. **Update AlphaFold service**:
   - Set `ALPHAFOLD_MSA_MODE=mmseqs2` by default in `.env`
   - Point to optimized DB: `ALPHAFOLD_MMSEQS2_DATABASE_PATH=...`

3. **Documentation**:
   - Update `docs/ZERO_TOUCH_QUICKSTART.md` with MMseqs2 setup
   - Add benchmarking results to `docs/PERFORMANCE.md`
   - Document DB size trade-offs in `README.md`

## Success Metrics

| Metric | Current (JackHMMER) | Target (MMseqs2) | Status |
|--------|---------------------|------------------|---------|
| MSA time | ~300-600s | ~30-60s (10x) | Pending |
| Total inference time | ~400-700s | ~150-250s (2-3x) | Pending |
| GPU utilization | 30-40% | 60-80% | Pending |
| Memory usage | <30GB | <40GB | In Progress |
| MSA depth | ~500-5000 seqs | ~500-5000 seqs | Pending |

## Timeline

- ✅ **Phase 1**: DB rebuild (Est: 30-60 min) - **IN PROGRESS** (79% complete)
- **Phase 2**: Single search benchmark (Est: 5 min)
- **Phase 3**: Full pipeline comparison (Est: 15-30 min)
- **Phase 4**: Analysis and reporting (Est: 10 min)
- **Phase 5**: Integration (Est: 15 min)

**Total estimated time**: 1-2 hours

## Rollback Plan

If MMseqs2 optimization doesn't meet goals:

1. Keep JackHMMER as default MSA method
2. Document MMseqs2 as experimental feature
3. Provide opt-in flag `--msa_mode=mmseqs2` for users with specific needs
4. Consider alternative fast MSA tools (e.g., HH-suite, DIAMOND)

## References

- [MMseqs2 paper](https://www.nature.com/articles/nbt.3988)
- [AlphaFold database sizes](https://github.com/google-deepmind/alphafold#genetic-databases)
- [MMseqs2 wiki](https://github.com/soedinglab/MMseqs2/wiki)
