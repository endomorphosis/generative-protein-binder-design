# Jupyter User Container

Docker container with Jupyter Notebook for interactive protein design workflows.

## Quick Start

### Using Docker
```bash
docker build -t jupyter-protein-design .
docker run -p 8888:8888 -v $(pwd)/../src:/workspace/notebooks jupyter-protein-design
```

### Using Docker Compose
```bash
docker compose -f ../deploy/docker-compose-full.yaml up jupyter
```

Access Jupyter at http://localhost:8888

## Included Packages

- Jupyter Notebook / JupyterLab
- Biopython
- py3Dmol (for 3D visualization)
- NumPy, Pandas, Matplotlib
- HTTP clients (requests)

## Default Configuration

- No token/password required (for development)
- Listens on 0.0.0.0:8888
- Working directory: /workspace

## Security Note

For production use, enable token authentication:

```bash
docker run -p 8888:8888 jupyter-protein-design \
  jupyter notebook --NotebookApp.token='your-secure-token'
```

See [DOCKER_MCP_README.md](../docs/DOCKER_MCP_README.md) for more information.
