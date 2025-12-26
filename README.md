# Protein Binder Design Blueprint (MCP + Dashboard)

![A workflow diagram of the Protein Design Blueprint](docs/Protein_Design_Architecture_Diagram.png)

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

Submit a demo job:

```bash
./scripts/submit_demo_job.sh
```

Monitor a job from the CLI (helps detect “is it hung?” and prints progress + cache/mem metrics):

```bash
./scripts/monitor_job.sh <job_id> --metrics
```

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
- **Performance optimizations**: [docs/ALPHAFOLD_OPTIMIZATION_GUIDE.md](docs/ALPHAFOLD_OPTIMIZATION_GUIDE.md) **(29% faster)**
- Docker + MCP stack details: [docs/DOCKER_MCP_README.md](docs/DOCKER_MCP_README.md)
- Deployment compose files: [deploy/](deploy/)
- Architecture/background: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
