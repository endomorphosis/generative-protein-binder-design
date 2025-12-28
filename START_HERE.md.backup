# Start Here (non-ML friendly)

This project can run on **AMD64 (x86_64)** and **ARM64 (aarch64)**.

If you only want the web UI (Dashboard) and a working end-to-end demo, follow this page.

## 1) One-click start (VS Code)

1. Open this folder in VS Code.
2. Press `Ctrl+Shift+P` → run **Tasks: Run Task**.
3. Select **Stack: Start + Open Dashboard**.

That’s it — it will:
- auto-pick the correct stack for your computer (ARM64 vs AMD64)
- build containers (first run takes longer)
- open the Dashboard in your browser

## 2) One-command start (Terminal)

From the repo root:

```bash
./scripts/run_dashboard_stack.sh up -d --build
```

Open the Dashboard:
- http://localhost:3000

## 3) Quick health check (recommended)

If anything feels “stuck”, run:

```bash
./scripts/doctor_stack.sh
```

It prints a simple checklist (Docker OK, services OK, URLs OK) and the current service status.

## 4) Submit a demo job

This confirms the MCP server + dashboard pipeline wiring is working:

```bash
./scripts/submit_demo_job.sh
```

Then open the Dashboard and look for the new job.

## 5) (Optional) Zero-Touch Native Installer

If you want the native toolchain (AlphaFold, RFDiffusion, ProteinMPNN) plus MMseqs2 databases built automatically:

```bash
# Minimal DBs (fastest)
bash scripts/install_all_native.sh --minimal

# Recommended DBs (dev)
bash scripts/install_all_native.sh --recommended

# Full DBs (production)
bash scripts/install_all_native.sh --full
```

What it does:
- Installs tools into `~/miniforge3/envs/alphafold2`
- Downloads AlphaFold DBs for the chosen tier
- Builds MMseqs2 databases to `~/.cache/alphafold/mmseqs2` (GPU-accelerated when available)
- Skips already-built tiers automatically

## 5) Choose backend + fallback order

In the Dashboard header, click **Settings** to choose how model calls are routed:
- **NIM** (NVIDIA NIM services)
- **External** (any compatible REST services you run elsewhere)
- **Embedded** (last-resort: run inside the MCP server container; currently only supports ProteinMPNN when present)

Use **fallback** mode to try providers in order (recommended). These settings persist across restarts when using the provided Docker compose stacks.

## Common questions

### “Some services show not_ready or disabled”

That’s normal depending on platform and configuration:
- On **AMD64**, the NIM model services can run natively (best supported).
- On **ARM64**, the repo includes an ARM64-native stack, but some heavyweight models may require additional model downloads/configuration.

If you’re unsure what you’re seeing, run `./scripts/doctor_stack.sh` and share the output.

### “Port 3000 is already in use”

Run with a different dashboard port:

```bash
MCP_DASHBOARD_HOST_PORT=3005 ./scripts/run_dashboard_stack.sh up -d --build
```

## Where to go next

- MCP + Docker guide: [docs/DOCKER_MCP_README.md](docs/DOCKER_MCP_README.md)
- Platform guidance: [scripts/detect_platform.sh](scripts/detect_platform.sh)
