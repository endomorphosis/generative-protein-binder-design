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
┌─────────────────┐
│   MCP Server    │ (Port 8000)
│    (FastAPI)    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  NIM Services (Protein Design)      │
├─────────────────────────────────────┤
│ • AlphaFold2         (Port 8081)    │
│ • RFDiffusion        (Port 8082)    │
│ • ProteinMPNN        (Port 8083)    │
│ • AlphaFold2-Multimer (Port 8084)   │
└─────────────────────────────────────┘

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
docker compose -f ../deploy/docker-compose-full.yaml up
```

### Accessing Services

- **MCP Dashboard**: http://localhost:3000
- **MCP Server API**: http://localhost:8000
- **MCP Server Docs**: http://localhost:8000/docs
- **Jupyter Notebook**: http://localhost:8888
- **NIM Services**: http://localhost:8081-8084

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

#### Health & Monitoring
- `GET /health` - Server health check
- `GET /api/services/status` - Check NIM services status

### Example Usage

Create a new job:
```bash
curl -X POST http://localhost:8000/api/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "sequence": "MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV",
    "job_name": "My First Design",
    "num_designs": 5
  }'
```

Check job status:
```bash
curl http://localhost:8000/api/jobs/{job_id}
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
- `ALPHAFOLD_URL` - AlphaFold service URL (default: http://localhost:8081)
- `RFDIFFUSION_URL` - RFDiffusion service URL (default: http://localhost:8082)
- `PROTEINMPNN_URL` - ProteinMPNN service URL (default: http://localhost:8083)
- `ALPHAFOLD_MULTIMER_URL` - AlphaFold-Multimer service URL (default: http://localhost:8084)

#### MCP Dashboard
- `NEXT_PUBLIC_MCP_SERVER_URL` - MCP Server URL (default: http://localhost:8000)

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

Only starts the NIM services for direct API access.

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
curl http://localhost:8000/health
```

### MCP Server can't connect to NIMs

Check NIM service status:
```bash
curl http://localhost:8081/v1/health/ready
curl http://localhost:8082/v1/health/ready
curl http://localhost:8083/v1/health/ready
curl http://localhost:8084/v1/health/ready
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
