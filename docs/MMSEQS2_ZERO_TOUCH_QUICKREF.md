# MMseqs2-GPU Installation Quick Reference

## One-Command Installation

### Minimal (Fast, ~15 min, 35GB total)
```bash
cd ~/generative-protein-binder-design
./scripts/install_all_native.sh --minimal
```

### Recommended (Balanced, ~1 hour, 100GB total) 
```bash
./scripts/install_all_native.sh --recommended
```

### Full (Production, ~6 hours, 2.5TB total)
```bash
./scripts/install_all_native.sh --full
```

## What Gets Installed

Each installation automatically includes:
- ✅ AlphaFold2 (with specified database tier)
- ✅ **MMseqs2** (with MMseqs2 database build)
- ✅ RFDiffusion
- ✅ ProteinMPNN
- ✅ MCP Server configuration
- ✅ GPU optimization (auto-detected)

## Verify Installation

```bash
# Check MMseqs2 installation
conda activate alphafold2
mmseqs --version

# Check database
ls -lh ~/.cache/alphafold/mmseqs2/uniref90_db*

# Check environment
source ~/.cache/alphafold/.env.mmseqs2
echo "MMseqs2 Database: $ALPHAFOLD_MMSEQS2_DATABASE_PATH"
```

## Use in AlphaFold

```bash
# Activate environment
source ~/.cache/alphafold/.env.gpu
source ~/.cache/alphafold/.env.mmseqs2

# Run with MMseqs2 MSA (faster than jackhmmer)
python tools/alphafold2/run_alphafold.py \
  --fasta_paths=target.fa \
  --output_dir=results \
  --msa_mode=mmseqs2 \
  --mmseqs2_database_path="$ALPHAFOLD_MMSEQS2_DATABASE_PATH"
```

## Custom Database Conversion Only

If you already have AlphaFold installed:

```bash
# Convert existing AlphaFold databases to MMseqs2
./scripts/convert_alphafold_db_to_mmseqs2_multistage.sh --tier reduced --gpu
```

## Test End-to-End Setup

```bash
# Validate entire installation
./scripts/test_mmseqs2_zero_touch_e2e.sh --tier minimal --skip-install

# Result should show: "15/15 tests PASSED"
```

## GPU Modes

Auto-detected, but can be overridden:
- `cuda`: NVIDIA GPUs (best performance)
- `metal`: Apple Silicon (limited)
- `rocm`: AMD GPUs (experimental)
- `cpu`: CPU-only fallback

## Troubleshooting

### MMseqs2 not found
```bash
# Reinstall MMseqs2 only
./scripts/install_mmseqs2.sh --conda-env alphafold2 --install-only
```

### Database not found
```bash
# Check location
echo $ALPHAFOLD_MMSEQS2_DATABASE_PATH

# Rebuild database
./scripts/install_mmseqs2.sh --build-db --force
```

### Out of memory during conversion
```bash
# Use reduced tier instead
./scripts/convert_alphafold_db_to_mmseqs2_multistage.sh --tier reduced
```

### GPU not working
```bash
# Fall back to CPU
./scripts/install_all_native.sh --minimal --gpu cpu
```

## Performance Tips

1. **First run is slowest** (database building)
   - Subsequent runs use cached databases
   - Much faster MSA generation after conversion

2. **Use GPU if available**
   - ~3-5x faster MSA searches
   - Requires NVIDIA CUDA 11.8+

3. **Choose right tier**
   - minimal: Demo, testing
   - reduced: Development (good accuracy/speed balance)
   - full: Production, research

## What's New (vs Previous Setup)

| Aspect | Before | After |
|--------|--------|-------|
| MMseqs2 setup | Manual, separate script | Automatic, integrated |
| Database conversion | Not implemented | Automatic, 3 tiers |
| GPU support | Limited | Full integration |
| Zero-touch? | No | Yes |
| Database tier choice | Manual | Inherited from install |
| Time to production | 2+ hours | 1 hour (reduced) |

## Environment Files

After installation, check:
```bash
# GPU configuration
cat ~/.cache/alphafold/.env.gpu

# MMseqs2 configuration  
cat ~/.cache/alphafold/.env.mmseqs2

# Combined (use this for everything)
source ~/.cache/alphafold/.env.gpu
source ~/.cache/alphafold/.env.mmseqs2
```

## Files Reference

| Script | Purpose | Usage |
|--------|---------|-------|
| `install_all_native.sh` | Main installer | `./scripts/install_all_native.sh --help` |
| `install_mmseqs2.sh` | MMseqs2 installer | Auto-called by main installer |
| `convert_alphafold_db_to_mmseqs2_multistage.sh` | DB converter | Can be run standalone |
| `test_mmseqs2_zero_touch_e2e.sh` | Validation | `./scripts/test_mmseqs2_zero_touch_e2e.sh` |

## Documentation

- [Full Implementation Details](docs/MMSEQS2_ZERO_TOUCH_IMPLEMENTATION.md)
- [MMseqs2 Optimization Guide](docs/MMSEQS2_OPTIMIZATION_PLAN.md)
- [AlphaFold Optimization](docs/ALPHAFOLD_OPTIMIZATION_GUIDE.md)
- [Zero-Touch Quickstart](docs/ZERO_TOUCH_QUICKSTART.md)
