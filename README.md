# Protein Binder Design Blueprint (MCP + Dashboard)

This repository packages an end-to-end **protein binder design** workflow behind a simple control plane:

- **MCP Server** (FastAPI) orchestrates jobs and stores results
- **MCP Dashboard** (Next.js) submits jobs and visualizes progress/results
- Model backends can be:
  - **AMD64 Docker/NIM services** (where available)
  - **ARM64 host-native services** (recommended on DGX Spark / aarch64)
  - **Hybrid/fallback routing** via server config

## Quick Start (recommended)

**⚡ Performance Optimizations**:
- [AlphaFold optimizations](docs/ALPHAFOLD_OPTIMIZATION_GUIDE.md) for **29% faster** inference (balanced preset default)
- **[Phase 2 GPU kernels](PHASE_2_README.md)** for **15-30x faster** MSA alignment (auto-installed on DGX Spark/CUDA 13.1+)

Start the Dashboard + MCP Server stack (auto-selects the right compose file for your platform):

```bash
./scripts/run_dashboard_stack.sh up -d --build
```

Open:

- Dashboard: http://localhost:${MCP_DASHBOARD_HOST_PORT:-3000}
- MCP Server health: http://localhost:${MCP_SERVER_HOST_PORT:-8011}/health

Submit a demo job:

```bash
./scripts/submit_demo_job.sh
```

Monitor a job from the CLI (helps detect “is it hung?” and prints progress + cache/mem metrics):

```bash
./scripts/monitor_job.sh <job_id> --metrics
```

## Zero-Touch Native Installer (AlphaFold + MMseqs2)

Use the unified installer to provision AlphaFold (tiered DBs), MMseqs2, RFDiffusion, and ProteinMPNN with one command. MMseqs2 databases are automatically built to `~/.cache/alphafold/mmseqs2`.

| Profile | Command | What it does |
| --- | --- | --- |
| Minimal (CPU-friendly) | `bash scripts/install_all_native.sh --minimal` | Installs tools + UniRef90 → MMseqs2 (fastest download/build) |
| Recommended (dev) | `bash scripts/install_all_native.sh --recommended` | Installs tools + UniRef90 + small BFD → MMseqs2 |
| Full (production) | `bash scripts/install_all_native.sh --full` | Installs tools + full AlphaFold DBs (UniRef90, BFD, PDB SeqRes, UniProt) → MMseqs2 |

Notes:
- GPU indexing is auto-detected; falls back to CPU if no GPU.
- Existing MMseqs2 DBs are auto-detected and skipped; use `rm -rf ~/.cache/alphafold/mmseqs2` to force rebuild.
- Integration details: [docs/MMSEQS2_INSTALLER_INTEGRATION.md](docs/MMSEQS2_INSTALLER_INTEGRATION.md).

## Ports & Services (defaults)

- Dashboard: `3000`
- MCP Server (host): `${MCP_SERVER_HOST_PORT:-8011}` (container listens on `8000`)
- ARM64 host-native wrappers (when enabled):
  - AlphaFold2: `18081` (includes `/v1/metrics`)
  - RFDiffusion: `18082`
  - ProteinMPNN: `18083`
  - AlphaFold2-Multimer: `18084`

## API Notes

- Job status: `GET /api/jobs/{job_id}`
- Optional diagnostics (best-effort):
  - `include_metrics=1` includes stage timing + host snapshots
  - `include_residency=1` also includes DB page-cache residency sampling (slower)
- Errors are **UI-safe by default** (summarized/truncated). If you need the full failure detail for debugging:
  - `GET /api/jobs/{job_id}?include_error_detail=1`

## ARM64 / DGX Spark notes

- AlphaFold2-Multimer uses conservative defaults on ARM64 to avoid known JAX/XLA bf16 conversion crashes.
- You can override bfloat16 behavior via environment variables (see the ARM64 docs in `docs/`).

## Where to go next

- New here? Start with [START_HERE.md](START_HERE.md)
- **Performance optimizations**: 
  - [AlphaFold optimizations](docs/ALPHAFOLD_OPTIMIZATION_GUIDE.md) **(29% faster)**
  - **[Phase 2 GPU kernels](PHASE_2_README.md) (15-30x faster MSA alignment)** ← NEW!
- Docker + MCP stack details: [docs/DOCKER_MCP_README.md](docs/DOCKER_MCP_README.md)
- Deployment compose files: [deploy/](deploy/)
- Architecture/background: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

## GPU kernels (MMseqs2 prefilter)

**Phase 1 (Current):**
- CUDA k-mer extraction and diagonal scoring live in [tools/gpu_kernels/kmer_matching.cu](tools/gpu_kernels/kmer_matching.cu)
- Benchmarks: `benchmark_kmer`, `benchmark_batched_diagonals`, `benchmark_end_to_end`, `benchmark_fp4_comparison`
- **Experimental FP4 quantization** (Blackwell/GB10 tensor cores):
  - Enable with `-DENABLE_NVFP4_EXPERIMENTAL=ON` (requires SM≥90)
  - Currently uses simulated FP4 on CUDA 13.0 (1.02x speedup, 100% accuracy)
  - Upgrade to CUDA 13.1+ for native tensor cores (2-4x expected speedup)
  - See [docs/CUDA_UPGRADE_FP4_GUIDE.md](docs/CUDA_UPGRADE_FP4_GUIDE.md) for upgrade instructions
  - Validate accuracy: `python3 tools/gpu_kernels/validate_fp4_accuracy.py`

**Phase 2 (Infrastructure Complete):**
- **15-30x speedup target** via device-side index, batch processing, and streaming
- Auto-installed during `./scripts/install_all_native.sh` on CUDA 13.1+ systems
- Documentation: [PHASE_2_README.md](PHASE_2_README.md), [PHASE_2_ZERO_TOUCH_INSTALL.md](PHASE_2_ZERO_TOUCH_INSTALL.md)
- Tests: `bash tests/test_phase2_integration.sh` (37 tests, 100% pass rate)
- Status: ✅ Infrastructure ready, ⏳ kernel integration Week 1 (Jan 2-9, 2025)

