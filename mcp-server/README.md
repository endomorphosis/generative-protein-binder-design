# MCP Server

Model Context Protocol server for managing protein binder design workflows.

## Quick Start

### Using Docker
```bash
docker build -t mcp-server .
docker run -p 8000:8000 \
  -e ALPHAFOLD_URL=http://alphafold:8000 \
  -e RFDIFFUSION_URL=http://rfdiffusion:8000 \
  -e PROTEINMPNN_URL=http://proteinmpnn:8000 \
  -e ALPHAFOLD_MULTIMER_URL=http://alphafold-multimer:8000 \
  mcp-server
```

### Local Development
```bash
pip install -r requirements.txt
python server.py
```

The server will be available at http://localhost:8000

API documentation available at http://localhost:8000/docs

## Environment Variables

- `ALPHAFOLD_URL` - AlphaFold service endpoint (default: http://localhost:8081)
- `RFDIFFUSION_URL` - RFDiffusion service endpoint (default: http://localhost:8082)
- `PROTEINMPNN_URL` - ProteinMPNN service endpoint (default: http://localhost:8083)
- `ALPHAFOLD_MULTIMER_URL` - AlphaFold-Multimer service endpoint (default: http://localhost:8084)

## API Endpoints

See [DOCKER_MCP_README.md](../docs/DOCKER_MCP_README.md) for detailed API documentation.
