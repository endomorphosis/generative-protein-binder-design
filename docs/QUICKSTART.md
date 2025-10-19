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
cd mcp-dashboard
npm install
npm run dev
```

Access:
- Dashboard: http://localhost:3000
- MCP Server API: http://localhost:8000/docs

## Option 2: Full Stack with Docker Compose

To run the complete stack including NIM services, MCP server, dashboard, and Jupyter:

### 1. Set up environment variables

```bash
export NGC_CLI_API_KEY=<your-ngc-api-key>
export HOST_NIM_CACHE=~/.cache/nim
mkdir -p ~/.cache/nim
chmod -R 777 ~/.cache/nim
```

### 2. Start all services

```bash
docker compose -f ../deploy/docker-compose-full.yaml up
```

This will start:
- AlphaFold2 (port 8081)
- RFDiffusion (port 8082)
- ProteinMPNN (port 8083)
- AlphaFold2-Multimer (port 8084)
- MCP Server (port 8000)
- MCP Dashboard (port 3000)
- Jupyter Notebook (port 8888)

**Note**: First run will download ~1.3TB of model data and may take several hours.

### 3. Access the services

- **MCP Dashboard**: http://localhost:3000
  - Submit new design jobs
  - Monitor job progress
  - View and download results
  
- **MCP Server API Documentation**: http://localhost:8000/docs
  - Interactive API documentation
  - Test endpoints directly
  
- **Jupyter Notebook**: http://localhost:8888
  - Interactive notebook environment
  - Example notebook available

## Using the Dashboard

1. **Check Service Status**: The top of the dashboard shows the status of all NIM services

2. **Submit a New Job**:
   - Enter a protein sequence (e.g., `MKFLKFSLLTAVLLSVVFAFSSCG`)
   - Optionally provide a job name
   - Set the number of designs (1-20)
   - Click "Start Design Job"

3. **Monitor Progress**:
   - Jobs appear in the middle panel
   - Click on a job to view details
   - Progress updates automatically every 5 seconds

4. **View Results**:
   - Completed jobs show results in the right panel
   - View target structure and generated designs
   - Download results as JSON

5. **Launch Jupyter**:
   - Click "Open Jupyter Notebook" button
   - Explore the example notebook

## Using the MCP Server API

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

- Read the [full documentation](DOCKER_MCP_README.md)
- Explore the [example notebook](src/protein-binder-design.ipynb)
- Check out the [MCP Server API docs](http://localhost:8000/docs)

## Architecture

```
User Browser → Dashboard (React/Next.js) → MCP Server (FastAPI) → NIM Services
                      ↓
                Jupyter Notebook
```

## Support

For issues or questions, please open an issue on GitHub.
