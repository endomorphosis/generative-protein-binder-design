# Protein Binder Design Blueprint (MCP + Dashboard)

This repository packages an end-to-end **protein binder design** workflow behind a simple control plane:

- **MCP Server** (FastAPI) orchestrates jobs and stores results
- **MCP Dashboard** (Next.js) submits jobs and visualizes progress/results
- Model backends can be:
  - **AMD64 Docker/NIM services** (where available)
  - **ARM64 host-native services** (recommended on DGX Spark / aarch64)
  - **Hybrid/fallback routing** via server config

## Quick Start (recommended)

**⚡ Performance**: Now includes [AlphaFold optimizations](docs/ALPHAFOLD_OPTIMIZATION_GUIDE.md) for **29% faster** inference (balanced preset default).

Start the Dashboard + MCP Server stack (auto-selects the right compose file for your platform):

```bash
./scripts/run_dashboard_stack.sh up -d --build
```

Open:

- Dashboard: http://localhost:${MCP_DASHBOARD_HOST_PORT:-3000}
- MCP Server health: http://localhost:${MCP_SERVER_HOST_PORT:-8011}/health

Notes:
- The compose stacks expose the MCP Server on `8011` by default.
- Some tooling/demos in this repo also use a **standalone MCP server** on `8010` (see “Run modes” below).

Submit a demo job:

```bash
./scripts/submit_demo_job.sh
```

Monitor a job from the CLI (helps detect "is it hung?" and prints progress + cache/mem metrics):

```bash
./scripts/monitor_job.sh <job_id> --metrics
```

## Zero-Touch Native Installer (AlphaFold + MMseqs2 + GPU Acceleration)

Use the unified installer to provision AlphaFold (tiered DBs), MMseqs2, RFDiffusion, and ProteinMPNN with one command. MMseqs2 databases are automatically built to `~/.cache/alphafold/mmseqs2`.

**🚀 NEW**: Automatically detects and configures GPU acceleration for 5-10x faster MSA generation!

| Profile | Command | What it does |
| --- | --- | --- |
| Minimal (CPU-friendly) | `bash scripts/install_all_native.sh --minimal` | Installs tools + UniRef90 → MMseqs2 (fastest download/build) |
| Recommended (dev) | `bash scripts/install_all_native.sh --recommended` | Installs tools + UniRef90 + small BFD → MMseqs2 + **GPU auto-config** |
| Full (production) | `bash scripts/install_all_native.sh --full` | Installs tools + full AlphaFold DBs (UniRef90, BFD, PDB SeqRes, UniProt) → MMseqs2 + **GPU auto-config** |

Notes:
- GPU detection and configuration is automatic - no manual setup needed
- If GPU detected: Creates GPU server scripts for 5-10x MSA speedup
- GPU indexing is auto-detected; falls back to CPU if no GPU.
- MMseqs2 databases are created in GPU-server mode (works with existing databases)
- For details: See [MMseqs2 GPU Quickstart](docs/MMSEQS2_GPU_QUICKSTART.md)
- Existing MMseqs2 DBs are auto-detected and skipped; use `rm -rf ~/.cache/alphafold/mmseqs2` to force rebuild.
- Integration details: [docs/MMSEQS2_INSTALLER_INTEGRATION.md](docs/MMSEQS2_INSTALLER_INTEGRATION.md).

## Ports & Services (defaults)

- Dashboard: `3000`
- MCP Server (stack host port): `${MCP_SERVER_HOST_PORT:-8011}` (container listens on `8000`)
- MCP Server (standalone/local container host port, optional): `8010` (container listens on `8000`)
- Model service ports depend on stack and environment variables, but commonly use:
  - AlphaFold2: `18081`
  - RFDiffusion: `18082`
  - ProteinMPNN: `18083`
  - AlphaFold2-Multimer: `18084`

In AMD64 NIM stacks these ports map to NIM containers. In ARM64/DGX Spark flows they often map to **host-native wrapper services**.

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
- Docs index (canonical navigation): [docs/INDEX.md](docs/INDEX.md)
- For AI agents & contributors: [docs/AGENTS.md](docs/AGENTS.md)
- **🔥 IMPORTANT FOR AI AGENTS**: [INSTITUTIONAL_KNOWLEDGE.md](INSTITUTIONAL_KNOWLEDGE.md) - Complete GPU/MMseqs2 optimization work (10x speedup achieved!)
- **GPU Integration Summary**: [INTEGRATION_COMPLETE.md](INTEGRATION_COMPLETE.md) - All GPU/CUDA 13.1/MMseqs2 work verified (34/34 checks)
- **Zero-Touch GPU Setup**: [ZERO_TOUCH_GPU_COMPLETE.md](ZERO_TOUCH_GPU_COMPLETE.md) - Automated GPU configuration details
- **Performance optimizations**: [docs/ALPHAFOLD_OPTIMIZATION_GUIDE.md](docs/ALPHAFOLD_OPTIMIZATION_GUIDE.md) **(29% faster)**
- **MMseqs2 GPU Guide**: [docs/MMSEQS2_GPU_QUICKSTART.md](docs/MMSEQS2_GPU_QUICKSTART.md) - User guide for 10x speedup
- Docker + MCP stack details: [docs/DOCKER_MCP_README.md](docs/DOCKER_MCP_README.md)
- Deployment compose files: [deploy/](deploy/)
- Architecture/background: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
