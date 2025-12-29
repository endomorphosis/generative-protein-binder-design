# Quick Start (canonical)

This repo provides a **Dashboard (Next.js)** and an **MCP Server (FastAPI)** that orchestrate protein binder design jobs.

There are multiple backend modes (NIM services, host-native services, embedded execution), but you can get to a working UI with one command.

If you get lost, start from [../START_HERE.md](../START_HERE.md) and the [docs index](INDEX.md).

## Prerequisites

- Docker Engine
- Docker Compose v2 (`docker compose ...`)
- (Optional) GPU + NVIDIA Container Toolkit (needed for the AMD64 NIM services stack)
- (Optional) NGC key (needed to pull NIM images)

## 1) Start the stack (recommended)

From repo root:

```bash
./scripts/run_dashboard_stack.sh up -d --build
```

Open:
- Dashboard: `http://localhost:${MCP_DASHBOARD_HOST_PORT:-3000}`
- MCP Server health: `http://localhost:${MCP_SERVER_HOST_PORT:-8011}/health`

### What stack did I get?

`run_dashboard_stack.sh` auto-selects the best compose file:

- **AMD64/x86_64**
  - If host-native wrappers are already healthy on `18081/18082/18084`: uses the *host-native* dashboard stack
  - Else if `NGC_CLI_API_KEY` is set: uses the *NIM* dashboard stack
  - Else: uses the *control-plane only* stack (Dashboard + MCP server, no model backends)

- **ARM64/aarch64**
  - Uses the *ARM64 host-native* dashboard stack (routes to host-native wrappers)

You can force a mode:

```bash
./scripts/run_dashboard_stack.sh --control-plane up -d --build
./scripts/run_dashboard_stack.sh --amd64 up -d
./scripts/run_dashboard_stack.sh --arm64 up -d --build
./scripts/run_dashboard_stack.sh --arm64-host-native up -d --build
./scripts/run_dashboard_stack.sh --host-native up -d --build
```

## 2) Submit a demo job

```bash
./scripts/submit_demo_job.sh
```

This calls `POST /api/jobs` on whichever MCP server is reachable (it prefers the stack server on `8011`).

Monitor progress:

```bash
./scripts/monitor_job.sh <job_id> --metrics
```

## 3) Quick diagnostics

If anything feels stuck:

```bash
./scripts/doctor_stack.sh
```

It checks:
- Docker daemon reachability
- expected ports
- stack status (`docker compose ps`)
- MCP Server `/health` and `/api/services/status`

## 4) (Optional) Enable the AMD64 NIM services stack

The NIM compose stack requires an NGC key.

```bash
export NGC_CLI_API_KEY="<YOUR_NGC_PERSONAL_RUN_KEY>"
echo "$NGC_CLI_API_KEY" | docker login nvcr.io --username='$oauthtoken' --password-stdin

export HOST_NIM_CACHE="$HOME/.cache/nim"
mkdir -p "$HOST_NIM_CACHE"
chmod -R 777 "$HOST_NIM_CACHE"

./scripts/run_dashboard_stack.sh --amd64 up -d --build
```

Notes:
- First start may take hours due to large model downloads.
- In this repo’s compose stacks, the NIM services are typically published on host ports `18081–18084` (to avoid common `8081–8084` collisions).

## 5) (Optional) Start host-native model wrappers (DGX Spark / ARM64)

For ARM64 systems, the “real inference” path is generally host-native services + the dashboard stack.

```bash
./scripts/start_everything.sh --arm64-host-native --provision --db-tier minimal
```

This:
- starts a memory watchdog (`outputs/memory-watchdog.log`)
- starts host-native model wrappers (`outputs/host-native-services.log`)
- starts Dashboard + MCP server via docker compose

## 6) (Optional) Run a single MCP server container on port 8010

Some demos/tools in this repo default to a standalone MCP server at `http://localhost:8010`.

```bash
docker build -t mcp-server:local ./mcp-server
docker rm -f mcp-server-local >/dev/null 2>&1 || true
docker run -d --name mcp-server-local -p 8010:8000 --restart unless-stopped mcp-server:local
curl -sS http://localhost:8010/health
```

## Common pitfalls

- **Dashboard can’t connect to server**: check the dashboard container env (`MCP_SERVER_URL` / `NEXT_PUBLIC_MCP_SERVER_URL`) and run `./scripts/doctor_stack.sh`.
- **You have `MCP_SERVER_URL` exported in your shell**: it can override scripts; `submit_demo_job.sh` prefers `8011` unless you force `MCP_SERVER_URL_FORCE=1`.
- **Port collisions**: change ports via env vars when starting the stack:

```bash
MCP_DASHBOARD_HOST_PORT=3005 MCP_SERVER_HOST_PORT=8012 ./scripts/run_dashboard_stack.sh up -d --build
```

## Next steps

- Deeper Docker overview: [DOCKER_MCP_README.md](DOCKER_MCP_README.md)
- System architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
- For agents/contributors: [AGENTS.md](AGENTS.md)
