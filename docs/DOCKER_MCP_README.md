# Docker Container and MCP Server Implementation

This directory contains the implementation of a complete Docker-based infrastructure for the Protein Binder Design workflow, including:

1. **MCP Server** - Model Context Protocol server for workflow management
2. **MCP Dashboard** - Web-based user interface for submitting jobs and viewing results
3. **Jupyter Container** - User-facing Jupyter notebook environment

## Architecture Overview

```
┌─────────────────┐
│  MCP Dashboard  │ (Port 3000)
│   (React/Next)  │
└────────┬────────┘
         │
         ▼
┌──────────────────────────────┐
│   MCP Server (FastAPI)       │
│  Host: ${MCP_SERVER_HOST_PORT:-8011}
│  Container: 8000             │
└────────┬─────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────┐
│ Model backends (selected by stack / configuration)        │
├──────────────────────────────────────────────────────────┤
│ AMD64 dashboard stacks (common defaults):                 │
│ • AlphaFold2         (18081)                              │
│ • RFDiffusion        (18082)                              │
│ • ProteinMPNN        (18083)                              │
│ • AlphaFold2-Multimer (18084)                             │
│                                                          │
│ ARM64 host-native wrappers (DGX Spark / aarch64):         │
│ • AlphaFold2         (18081, includes /v1/metrics)        │
│ • RFDiffusion        (18082)                              │
│ • ProteinMPNN        (18083)                              │
│ • AlphaFold2-Multimer (18084)                             │
└──────────────────────────────────────────────────────────┘

┌─────────────────┐
│ Jupyter Server  │ (Port 8888)
│   (Notebooks)   │
└─────────────────┘
```

## Quick Start

### Prerequisites

1. Docker and Docker Compose installed
2. NVIDIA GPU with drivers and nvidia-container-toolkit
3. NGC CLI API Key (for NIM services)

### Setup

1. Set your NGC API key:
```bash
export NGC_CLI_API_KEY=<your-key>
```

2. Create NIM cache directory:
```bash
mkdir -p ~/.cache/nim
chmod -R 777 ~/.cache/nim
export HOST_NIM_CACHE=~/.cache/nim
```

3. Start all services:
```bash
./scripts/run_dashboard_stack.sh up -d --build
```

### Accessing Services

- **MCP Dashboard**: http://localhost:3000
- **MCP Server API (host)**: http://localhost:${MCP_SERVER_HOST_PORT:-8011}
- **MCP Server Docs (host)**: http://localhost:${MCP_SERVER_HOST_PORT:-8011}/docs
- **Jupyter Notebook**: http://localhost:8888
- **Model services (typical)**: http://localhost:18081-18084
- **ARM64 native wrappers (DGX Spark)**: http://localhost:18081-18084

Note: when running the MCP server directly (no stack), some defaults and older examples use `8081-8084` for model services.

### Backend routing + fallback (NIM → external → embedded)

The MCP server supports a configurable provider/fallback pattern:
- **nim**: talk to NIM model services (default in the AMD64 stack)
- **external**: talk to any compatible REST services you run elsewhere
- **embedded**: last-resort execution inside the MCP server container (currently supports ProteinMPNN when present)

You can change this from the Dashboard via the **Settings** button (top-right). Under the hood the Dashboard updates the MCP server runtime config:
- `GET /api/config`
- `PUT /api/config`
- `POST /api/config/reset`

The Docker compose dashboard stacks mount a named volume and set `MCP_CONFIG_PATH=/config/mcp_config.json` so these settings persist across restarts. Set `MCP_CONFIG_READONLY=1` to disable runtime edits.

For embedded provider downloads, the stacks also mount a persistent `/models` volume, so any downloaded model assets (like ProteinMPNN source/weights) can be reused across restarts.

### AlphaFold DB options (local staged download vs external/NIM)

You have two supported ways to use **full** AlphaFold databases:

1) **Download locally (staged install)**
   - In the Dashboard **Settings → Embedded**, configure:
     - `AlphaFold DB URL (reduced / initial)`
     - `AlphaFold DB URL (full extras, optional)`
   - Enable `allow auto-download`, and either click **Download embedded assets** or enable background startup bootstrap via:
     - `MCP_EMBEDDED_AUTO_DOWNLOAD=1`
     - `MCP_BOOTSTRAP_ON_STARTUP=1`
   - The server downloads the reduced pack first and continues with the full extras pack in the background.
   - Progress appears in the existing **Service Status** banner via the `reason` field.

2) **Point AlphaFold to a model service that already has the DBs**
   - In the Dashboard **Settings → External URLs** (or NIM URLs), set the AlphaFold endpoint to a container/service you run elsewhere.
   - That service is responsible for hosting/mounting the full databases. This is the recommended approach when DBs live in a dedicated model-serving container or remote cluster.

### Multi-platform (one command)

Use the helper script below to start the correct dashboard stack for your machine:

```bash
./scripts/run_dashboard_stack.sh up -d --build
```

What it does:
- On **AMD64/x86_64**, starts the **NIM** dashboard stack ([deploy/docker-compose-dashboard.yaml](deploy/docker-compose-dashboard.yaml)).
- On **ARM64/aarch64**, starts the **ARM64-native** dashboard stack ([deploy/docker-compose-dashboard-arm64-native.yaml](deploy/docker-compose-dashboard-arm64-native.yaml)).

You can also force a mode:

```bash
./scripts/run_dashboard_stack.sh --amd64 up -d
./scripts/run_dashboard_stack.sh --arm64 up -d --build
```

To run the AMD64 NIM stack on ARM64 via emulation (qemu/binfmt), use:

```bash
./scripts/run_dashboard_stack.sh --emulated up -d
```

### DGX Spark: dashboard + server only (control plane)

When AlphaFold/RFDiffusion/Multimer run in separate containers or on remote infrastructure, you can run just the MCP server + dashboard on DGX Spark and configure provider URLs in the dashboard.

```bash
docker compose -f deploy/docker-compose-dashboard-dgx-spark.yaml up -d --build
```

Then use **Settings** to either:
- Point **External URLs** / **NIM URLs** at services that already host full databases, or
- Configure **Embedded downloads** to stage reduced → full AlphaFold DBs locally under `/models`.

### Publish multi-arch core images (MCP server + dashboard)

The model service images differ by architecture (NIM is AMD64-only; ARM64-native model containers are built from source), but the **core** images we own can be published as true multi-arch images:

```bash
REGISTRY=ghcr.io/hallucinate-llc TAG=latest PUSH=1 ./scripts/build_multiplatform_core_images.sh
```

This publishes:
- `ghcr.io/hallucinate-llc/mcp-server:latest`
- `ghcr.io/hallucinate-llc/mcp-dashboard:latest`

To use published images (instead of local builds) with the dashboard stacks:

```bash
MCP_SERVER_IMAGE=ghcr.io/hallucinate-llc/mcp-server:latest \
MCP_DASHBOARD_IMAGE=ghcr.io/hallucinate-llc/mcp-dashboard:latest \
./scripts/run_dashboard_stack.sh up -d
```


## MCP Server

The MCP (Model Context Protocol) Server provides a REST API for managing protein binder design workflows.

### Features

- **Job Management**: Create, list, and monitor design jobs
- **MCP Protocol**: Implements Model Context Protocol for tool and resource discovery
- **Background Processing**: Asynchronous job execution
- **Service Health Monitoring**: Check status of all NIM services

### API Endpoints

#### MCP Protocol Endpoints
- `GET /mcp/v1/tools` - List available tools
- `GET /mcp/v1/resources` - List available resources (job results)
- `GET /mcp/v1/resources/{job_id}` - Get specific resource

#### Job Management
- `POST /api/jobs` - Create a new design job
- `GET /api/jobs` - List all jobs
- `GET /api/jobs/{job_id}` - Get job status
- `DELETE /api/jobs/{job_id}` - Delete a job

The job status endpoint supports optional, best-effort diagnostics:

- `include_metrics=1`: stage timing and host-side metrics snapshots (when available)
- `include_residency=1`: also include AlphaFold DB page-cache residency sampling (slower)
- `include_error_detail=1`: include full error details (default responses contain a summarized/truncated error suitable for UIs)

#### Health & Monitoring
- `GET /health` - Server health check
- `GET /api/services/status` - Check NIM services status

### Example Usage

Create a new job:
```bash
curl -X POST http://localhost:${MCP_SERVER_HOST_PORT:-8011}/api/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV",
    "job_name": "My First Design",
    "num_designs": 5
  }'
```

Check job status:
```bash
curl "http://localhost:${MCP_SERVER_HOST_PORT:-8011}/api/jobs/{job_id}?include_metrics=1"
```

## MCP Dashboard

The MCP Dashboard is a modern web application built with React and Next.js that provides a user-friendly interface for the protein design workflow.

### Features

- **Job Submission**: Submit new protein design jobs with custom parameters
- **Real-time Monitoring**: Watch job progress in real-time
- **Results Visualization**: View and download results
- **Service Status**: Monitor health of all NIM services
- **Jupyter Integration**: Launch Jupyter notebooks directly from the dashboard

### Technology Stack

- Next.js 14 (React framework)
- TypeScript
- Tailwind CSS
- MCP JavaScript SDK
- Axios for API calls

### Development

To run the dashboard in development mode:

```bash
cd mcp-dashboard
npm install
npm run dev
```

The dashboard will be available at http://localhost:3000

### Building

```bash
cd mcp-dashboard
npm run build
npm start
```

## Jupyter Container

The Jupyter container provides an interactive notebook environment for exploring the protein design workflow.

### Features

- Pre-configured Jupyter Notebook
- Includes example notebook: `protein-binder-design.ipynb`
- Python packages for bioinformatics: Biopython, py3Dmol
- Data science tools: NumPy, Pandas, Matplotlib

### Usage

Access Jupyter at http://localhost:8888 (no token required in default configuration).

To run with a token for security:

```bash
docker compose -f ../deploy/docker-compose-full.yaml run -p 8888:8888 jupyter \
  jupyter notebook --ip=0.0.0.0 --NotebookApp.token='your-secure-token'
```

## Configuration

### Environment Variables

#### MCP Server
- `ALPHAFOLD_URL` - AlphaFold service URL (default: http://localhost:8081 when running locally; stack setups typically point this at http://localhost:18081)
- `RFDIFFUSION_URL` - RFDiffusion service URL (default: http://localhost:8082 when running locally; stack setups typically point this at http://localhost:18082)
- `PROTEINMPNN_URL` - ProteinMPNN service URL (default: http://localhost:8083 when running locally; stack setups typically point this at http://localhost:18083)
- `ALPHAFOLD_MULTIMER_URL` - AlphaFold-Multimer service URL (default: http://localhost:8084 when running locally; stack setups typically point this at http://localhost:18084)

#### MCP Dashboard
- `NEXT_PUBLIC_MCP_SERVER_URL` - MCP Server URL (often set explicitly; some dev tools default to http://localhost:8010 for a standalone MCP server)

#### NIM Services
- `NGC_CLI_API_KEY` - NGC API key (required)
- `HOST_NIM_CACHE` - Path to NIM cache directory (default: ~/.cache/nim)

## Deployment Options

### Option 1: Full Stack (Recommended for Production)
```bash
docker compose -f ../deploy/docker-compose-full.yaml up
```

Includes all services: NIMs, MCP Server, Dashboard, and Jupyter.

### Option 2: NIMs Only (Original)
```bash
cd deploy
docker compose up
```

### Option 2b: Dashboard Stack with Non-Conflicting Ports

This starts the dashboard + MCP server + all auxiliary model services together.
Model services are published on `18081-18084` to avoid collisions with other stacks that commonly use `8081-8084`.

Note: The NIM images used by this stack are `linux/amd64`. On ARM64 hosts, you will need emulation (binfmt/qemu) or use Option 2c.

```bash
# From repo root
export NGC_CLI_API_KEY=<your-key>
mkdir -p ~/.cache/nim
chmod -R 777 ~/.cache/nim
export HOST_NIM_CACHE=~/.cache/nim

docker compose -f deploy/docker-compose-dashboard.yaml up -d
```

Ports (defaults):
- Dashboard: `http://localhost:3000`
- MCP Server: `http://localhost:8011`
- Model services: `http://localhost:18081-18084`

Override ports if needed:

```bash
MCP_DASHBOARD_HOST_PORT=3005 MCP_SERVER_HOST_PORT=8015 docker compose -f deploy/docker-compose-dashboard.yaml up -d
```

Only starts the NIM services for direct API access.

### Option 2c: ARM64-Native Dashboard Stack (No Emulation)

This starts the dashboard + MCP server + ARM64-native model services built from source.
Model services are published on `18081-18083` (AlphaFold2-Multimer is not included).

```bash
mkdir -p ~/.cache/nim
chmod -R 777 ~/.cache/nim
export HOST_NIM_CACHE=~/.cache/nim

docker compose -f deploy/docker-compose-dashboard-arm64-native.yaml up -d --build
```

Notes:
- Mock/fallback model outputs are CI-only (enabled when `CI=1`).
- The ARM64 ProteinMPNN service includes the upstream ProteinMPNN code + weights and can run “real weights” in-container.
- The ARM64 AlphaFold2 and RFDiffusion services are CI-only API shims and will report `not_ready` in runtime. For real inference, configure External/NIM endpoints in the dashboard or use a native install.

### Option 3: Development Mode
```bash
# Start NIMs
cd deploy && docker compose up -d

# Run MCP Server locally
cd mcp-server
pip install -r requirements.txt
python server.py

# Run Dashboard locally
cd mcp-dashboard
npm install
npm run dev

# Run Jupyter locally
cd src
jupyter notebook
```

## Troubleshooting

### Dashboard can't connect to MCP Server

Check that the MCP Server is running:
```bash
curl http://localhost:${MCP_SERVER_HOST_PORT:-8011}/health

# If you're using the standalone MCP server container instead:
# curl http://localhost:8010/health
```

### MCP Server can't connect to NIMs

Check NIM service status:
```bash
curl http://localhost:18081/v1/health/ready
curl http://localhost:18082/v1/health/ready
curl http://localhost:18083/v1/health/ready
curl http://localhost:18084/v1/health/ready

# Some legacy/custom setups may publish 8081–8084 instead.
```

### Jupyter container exits immediately

Check logs:
```bash
docker compose -f ../deploy/docker-compose-full.yaml logs jupyter
```

Ensure volumes are properly mounted.

## Production Considerations

1. **Security**:
   - Enable Jupyter token authentication
   - Add HTTPS/TLS termination
   - Restrict CORS origins in MCP Server
   - Use secrets management for NGC_CLI_API_KEY

2. **Scaling**:
   - Use Redis/PostgreSQL for job storage instead of in-memory
   - Add message queue (e.g., Celery) for job processing
   - Deploy behind a load balancer

3. **Monitoring**:
   - Add Prometheus metrics
   - Implement structured logging
   - Set up alerting for service failures

## License

This implementation follows the same license as the parent project.

## Support

For issues or questions, please open an issue in the GitHub repository.
