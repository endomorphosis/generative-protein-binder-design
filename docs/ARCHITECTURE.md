# Architecture

This document describes how the repository fits together today (components, ports, and request flows). For “how to run”, start with `docs/QUICKSTART.md` and `docs/AGENTS.md`.

## Components (high level)

- **MCP Server** (`mcp-server/`): FastAPI app that exposes MCP endpoints, job orchestration endpoints, runtime configuration, and backend/provider routing.
- **MCP Dashboard** (`mcp-dashboard/`): Next.js/React UI. The browser talks to dashboard routes; the dashboard proxies most backend calls to the MCP server.
- **Model backends**: One or more services that actually run the heavy steps (NIM containers, “host-native” wrappers, or “embedded” implementations).
- **Scripts + compose** (`scripts/`, `deploy/`): Source of truth for stack selection and how things get started on different platforms.

## Run modes and ports

There are two common MCP server entrypoints you’ll see referenced throughout the repo:

### 1) Compose “stack” mode (recommended)

- MCP server is published on host **`8011`** by default (container listens on 8000).
- Dashboard is published on host **`3000`** by default.
- Model services are commonly published on host ports **`18081`–`18084`** (exact services depend on mode/platform).

### 2) Standalone MCP server container (optional)

- MCP server is published on host **`8010`** by default (container listens on 8000).
- Useful for local development or when you don’t want the full dashboard stack.

Environment variables used across the repo:

- `MCP_SERVER_HOST_PORT`: host port for the stack MCP server (defaults to 8011 in stack scripts/compose).
- `MCP_DASHBOARD_HOST_PORT`: host port for the dashboard (defaults to 3000).
- `MCP_SERVER_URL` / `NEXT_PUBLIC_MCP_SERVER_URL`: where the dashboard/proxies point for the MCP server.

## Request and data flow

### Dashboard-driven flow (typical)

1. **Browser → Dashboard**: The user interacts with the Next.js app.
2. **Dashboard → MCP server (proxy)**: The dashboard calls server-side API routes under `mcp-dashboard/app/api/mcp/*`, which then forward requests to the MCP server.
3. **MCP server → backends**: The MCP server selects a provider chain (single or fallback) and calls the configured backends.
4. **Results**: The MCP server writes job outputs and serves them via REST/MCP endpoints; the dashboard polls or streams progress.

### Direct API flow (headless/agent)

Agents or scripts can talk directly to the MCP server without going through the dashboard:

- REST-style job APIs under `/api/*`
- MCP protocol endpoints (`/mcp` JSON-RPC, `/mcp/v1/*` REST)
- Server-sent events (SSE) endpoints for streaming updates

## MCP server API surface (practical)

### Health & status

- `GET /health`: server liveness.
- `GET /api/services/status`: aggregated backend/provider health.
- `GET /api/gpu/status`: GPU visibility/status (when supported).

### Runtime config

- `GET /api/config`: current routing/provider config.
- `PUT /api/config`: update routing/provider config.
- `POST /api/config/reset`: reset to defaults.

Config persistence:

- Persisted to `MCP_CONFIG_PATH` when set (compose stacks mount this under `/config/`).
- Can be forced read-only via `MCP_CONFIG_READONLY=1`.

### Jobs

- `POST /api/jobs`: create a job.
- `GET /api/jobs`: list jobs.
- `GET /api/jobs/{job_id}`: job status/details.

### MCP protocol + streaming

- `POST /mcp`: MCP JSON-RPC endpoint.
- `GET /mcp/v1/tools`, `GET /mcp/v1/resources`: MCP REST helpers.
- `GET /sse` and `GET /mcp/sse`: SSE streaming endpoints.

## Backend routing model

The MCP server supports multiple provider types and a routing strategy:

- **Provider types** (conceptually):
  - `nim`: NVIDIA Inference Microservice endpoints (usually on host ports like 18081–18084 in this repo’s stacks).
  - `external`: arbitrary HTTP endpoints you provide.
  - `embedded`: local/packaged implementations that can be bootstrapped/downloaded.

- **Routing modes**:
  - `single`: always use one provider.
  - `fallback`: try providers in order until one succeeds.

This logic lives primarily in `mcp-server/runtime_config.py` (schema, persistence, env overrides) and `mcp-server/model_backends.py` (provider implementations + fallback behavior).

## Dashboard proxy routes (why they exist)

The dashboard runs in the browser, but the MCP server may be on a different origin/port. The dashboard therefore provides server-side proxy routes under `mcp-dashboard/app/api/mcp/*` for:

- Config: `/api/mcp/config`, `/api/mcp/config/reset`
- Status: `/api/mcp/services/status`
- Tools/resources: `/api/mcp/tools`, `/api/mcp/tools/call`, `/api/mcp/resources`, `/api/mcp/resources/read`
- Jobs: `/api/mcp/jobs`, `/api/mcp/jobs/status`
- Embedded bootstrap: `/api/mcp/embedded/bootstrap`

SSE is also proxied via the dashboard’s SSE routes.

## Deployment and stack selection

The compose files under `deploy/` define multiple deployment variants (dashboard-only, full, GPU-optimized, ARM64, etc.). In practice, the supported entrypoint is:

- `./scripts/run_dashboard_stack.sh up -d --build`

That script auto-selects the appropriate compose file for your platform and available backends (e.g., host-native wrappers vs NIMs vs control-plane).

## Where to make changes

- **MCP server endpoints**: `mcp-server/server.py`
- **Routing/config schema + persistence**: `mcp-server/runtime_config.py`
- **Backend/provider implementations**: `mcp-server/model_backends.py`
- **Dashboard settings UI** (routing/provider config UX): `mcp-dashboard/components/BackendSettings.tsx`
- **Dashboard proxy handlers**: `mcp-dashboard/app/api/mcp/*`
- **Stack selection logic**: `scripts/run_dashboard_stack.sh` (and diagnostics in `scripts/doctor_stack.sh`)

## Troubleshooting hooks

- Stack health checks:
      - `curl http://localhost:${MCP_SERVER_HOST_PORT:-8011}/health`
      - `curl http://localhost:${MCP_SERVER_HOST_PORT:-8011}/api/services/status`
- Diagnostics script: `./scripts/doctor_stack.sh`
- Container logs: `./scripts/run_dashboard_stack.sh logs -f --tail=200`

## Security notes (current default)

The stack is primarily geared toward local/dev and trusted-network use:

- No authentication by default on most endpoints.
- If exposing beyond localhost, put a reverse proxy in front and add auth/TLS, and restrict origins.
