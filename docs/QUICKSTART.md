# Quick Start Guide: MCP Server and Dashboard

This guide will help you get started with the new MCP Server and Dashboard implementation for the Protein Binder Design Blueprint.

## Prerequisites

- Docker and Docker Compose installed
- NVIDIA GPU with drivers and nvidia-container-toolkit (for NIM services)
- NGC CLI API Key (for NIM services)
- At least 8GB RAM for the MCP server and dashboard alone

## Option 1: Quick Start (MCP Server + Dashboard Only)

If you just want to test the MCP server and dashboard without the NIM services:

```bash
# Start MCP Server
cd mcp-server
python3 -m pip install -r requirements.txt
python3 server.py &

# In another terminal, start the Dashboard
## Prerequisites

- Docker and Docker Compose
- NVIDIA GPU + nvidia-container-toolkit (for accelerated services)
- (Optional) NGC API key if you use NIM images

## Option 1: Fastest (Docker stack, recommended)

One command to start MCP Server + Dashboard with the correct compose file auto-selected (ARM64/AMD64):

```bash
./scripts/run_dashboard_stack.sh up -d --build
```

Access:
- Dashboard: http://localhost:${MCP_DASHBOARD_HOST_PORT:-3000}
- MCP Server API: http://localhost:${MCP_SERVER_HOST_PORT:-8011}/health

Submit a demo job:
```bash
./scripts/submit_demo_job.sh
```

Check health quickly:
```bash
./scripts/doctor_stack.sh
```

## Option 2: Zero-Touch Native Install (AlphaFold + MMseqs2)

Install native toolchain and build MMseqs2 databases automatically to `~/.cache/alphafold/mmseqs2`:

```bash
# Minimal DBs (fastest)
   - View target structure and generated designs
   - Download results as JSON

5. **Launch Jupyter**:
   - Click "Open Jupyter Notebook" button
   - Explore the example notebook

## Using the MCP Server API

Notes:
- GPU indexing auto-detected; falls back to CPU.
- Already-built tiers are skipped; remove `~/.cache/alphafold/mmseqs2` to force rebuild.

## Using the Dashboard

### Create a job
```bash
curl -X POST http://localhost:8000/api/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": "MKFLKFSLLTAVLLSVVFAFSSCG",
    "job_name": "My Design",
    "num_designs": 5
  }'
```

### Get job status
```bash
curl http://localhost:8000/api/jobs/{job_id}
```

### List all jobs
```bash
curl http://localhost:8000/api/jobs
```

### Check service health
```bash
curl http://localhost:8000/api/services/status
```

## Testing

Run the included test script to verify everything is working:

```bash
./scripts/test-mcp-server.sh
```

## Troubleshooting

### Dashboard can't connect to MCP Server
- Check that the MCP server is running: `curl http://localhost:8000/health`
- Check environment variable: `NEXT_PUBLIC_MCP_SERVER_URL`

### MCP Server shows NIM service errors
- Ensure NIM services are running: `docker compose ps`
- Check individual service health: `curl http://localhost:8081/v1/health/ready`

### Docker build fails
- Ensure you have sufficient disk space (>1.5TB for full stack)
- Check Docker daemon is running: `docker ps`

### Jobs complete but show mock data
- This is expected when NIM services aren't running
- The MCP server will fall back to mock data for demonstration

## Next Steps

- Read the [Docker MCP guide](DOCKER_MCP_README.md)
- Explore the [example notebook](../src/protein-binder-design.ipynb)
- Check the [MCP Server API docs](http://localhost:8000/docs)
- Review [MMseqs2 installer integration](MMSEQS2_INSTALLER_INTEGRATION.md)

## Architecture

```
User Browser → Dashboard (React/Next.js) → MCP Server (FastAPI) → NIM Services
                      ↓
                Jupyter Notebook
```

## Support

For issues or questions, please open an issue on GitHub.
