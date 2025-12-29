# Agent Guide (How to interact with this repo)

This guide is optimized for **automation and repeatability**: the same commands should work for humans and for AI agents.

## TL;DR workflows

### 1) Bring up the Dashboard + MCP Server (recommended)

```bash
./scripts/run_dashboard_stack.sh up -d --build
```

Then:
- Dashboard: `http://localhost:${MCP_DASHBOARD_HOST_PORT:-3000}`
- MCP Server health: `http://localhost:${MCP_SERVER_HOST_PORT:-8011}/health`

Submit a demo job:

```bash
./scripts/submit_demo_job.sh
```

### 2) Diagnose “it’s stuck”

```bash
./scripts/doctor_stack.sh
```

### 3) Tail logs

- Stack logs: `./scripts/run_dashboard_stack.sh logs -f --tail=200`
- Local MCP container logs (8010 mode): `docker logs -f mcp-server-local`

## Run modes and ports (know this first)

This repo commonly has **two MCP server modes**:

1) **Stack MCP server** (Docker Compose, used by Dashboard stacks)
   - Host port: `${MCP_SERVER_HOST_PORT:-8011}`
   - Container port: `8000`

2) **Local MCP server container** (single container, often used for MCP tool adapter demos)
   - Host port: `8010`
   - Container port: `8000`

The Dashboard code defaults to `http://localhost:8010` only when no `MCP_SERVER_URL` / `NEXT_PUBLIC_MCP_SERVER_URL` is set. In the compose stacks we set `MCP_SERVER_URL=http://mcp-server:8000`, so the Dashboard talks to the correct container automatically.

### How `run_dashboard_stack.sh` chooses a stack

`./scripts/run_dashboard_stack.sh` auto-selects a compose file:

- **AMD64/x86_64**
  - If host-native wrappers are already running and healthy on `18081/18082/18084`: uses `deploy/docker-compose-dashboard-host-native.yaml`
  - Else if `NGC_CLI_API_KEY` is set: uses `deploy/docker-compose-dashboard.yaml` (NIM model services)
  - Else: uses `deploy/docker-compose-dashboard-default.yaml` (control-plane only; configure providers in UI)

- **ARM64/aarch64**
  - Uses `deploy/docker-compose-dashboard-arm64-host-native.yaml` (routes to host-native wrappers)

You can force modes:

```bash
./scripts/run_dashboard_stack.sh --control-plane up -d --build
./scripts/run_dashboard_stack.sh --amd64 up -d
./scripts/run_dashboard_stack.sh --arm64 up -d --build
./scripts/run_dashboard_stack.sh --arm64-host-native up -d --build
./scripts/run_dashboard_stack.sh --host-native up -d --build
```

## Where to change things (repo map)

### Control plane

- `mcp-server/` — FastAPI server (jobs, routing config, SSE, MCP protocol)
- `mcp-dashboard/` — Next.js dashboard + API proxy routes (`/api/mcp/*`)
- `mcp-js-sdk/` — JS SDK used by the dashboard to call MCP tools/resources

### Model backends

- **NIM**: configured via runtime config + docker compose
- **Host-native wrappers (ARM64/DGX Spark)**: started by `scripts/start_everything.sh` and `scripts/run_arm64_native_model_services.sh`
- **Embedded**: runs inside the MCP server container (ProteinMPNN can be real when code+weights exist; AlphaFold/RFDiffusion embedding requires explicit runner configuration)

### Scripts you’ll use constantly

- `scripts/run_dashboard_stack.sh` — choose & run the correct compose stack
- `scripts/start_everything.sh` / `scripts/stop_everything.sh` — convenience wrappers (also start host-native services + watchdog)
- `scripts/submit_demo_job.sh` — smoke test job submission
- `scripts/monitor_job.sh` — polls job status + optional metrics
- `scripts/doctor_stack.sh` — checks ports, stack status, `/health`, `/api/services/status`

## MCP Server surface area (what the dashboard & agents call)

### REST endpoints (human-friendly)

- `GET /health`
- `GET /api/services/status`
- `POST /api/jobs` (create)
- `GET /api/jobs` (list)
- `GET /api/jobs/{job_id}` (poll)
  - Query params: `include_metrics=1`, `include_residency=1`, `include_error_detail=1`
- `GET /api/config` / `PUT /api/config` / `POST /api/config/reset` (backend routing/provider config)

### SSE

- `GET /sse` (job lifecycle events)
- `GET /mcp/sse` (alias)

### MCP protocol

- `POST /mcp` — JSON-RPC 2.0 endpoint (initialize/tools/list/tools/call/resources/list/resources/read)
- `GET /mcp/v1/tools`
- `GET /mcp/v1/resources`
- `GET /mcp/v1/resources/{job_id}`

## Dashboard proxy routes (important for debugging)

The dashboard generally calls server-side Next.js routes under `mcp-dashboard/app/api/mcp/*`, which then forward to the MCP server.

- If the UI can’t talk to the MCP server, check the dashboard container env:
  - `MCP_SERVER_URL` (preferred)
  - `NEXT_PUBLIC_MCP_SERVER_URL` (fallback)

## Tests / verification

- MCP server quick check: `./scripts/test-mcp-server.sh`
- Dashboard unit/e2e: see `mcp-dashboard/README.md`

## Editing conventions (for agents)

- Prefer editing *canonical docs* first: `README.md`, `START_HERE.md`, and `docs/INDEX.md`.
- When adding new flags/env vars, update:
  - the relevant script’s `usage()` text
  - `docs/QUICKSTART.md`
  - the top-level `README.md` ports/env section
- Avoid duplicating long instructions across many docs; link to the canonical guide instead.
