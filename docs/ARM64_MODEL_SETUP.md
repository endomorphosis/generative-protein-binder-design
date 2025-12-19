
# ARM64 Model Setup Guide

This guide explains how to provision and configure model assets for the ARM64-native dashboard stack.

Important: **Mock/fallback outputs are CI-only**. In normal runtime, missing models/weights should result in `not_ready` / failures.

Also note: the ARM64 Docker images for **AlphaFold2** and **RFDiffusion** in this repo are currently **CI-only API shims**. They do not ship a full AlphaFold/RFdiffusion toolchain and will not produce synthetic outputs at runtime.

## Setup (Real Models)

For real workflows on ARM64 (DGX Spark), you have two practical options:

1) **Native install on the host** (recommended for DGX Spark): install the tools + assets on the machine and run the MCP server in `MODEL_BACKEND=native`/`hybrid`.
2) **Remote provider**: run real model services elsewhere (NIM or your own containers in the cloud) and configure them via the MCP Dashboard (**Backend Settings** → External/NIM URLs).

If you're using the **ARM64 host-native dashboard stack** (deploy/docker-compose-dashboard-arm64-host-native.yaml), the simplest path is:

```bash
./scripts/provision_arm64_host_native_models.sh --db-tier reduced
./scripts/run_arm64_native_model_services.sh
./scripts/start_everything.sh --arm64-host-native
```

The provisioner uses the existing "complete" installers and will download weights/DBs and write integration env files under `tools/*/.env`.

Why this matters for ARM64: some upstream ML stacks assume x86_64-only prebuilt wheels/binaries (or are painful to build for aarch64 with CUDA). For scientists, the goal is to provide either a native installer or a remote endpoint with simple dashboard configuration.

To use real model weights for production workloads, you have two options:

### Option 1: Automatic Model Download (Best-effort)

The services automatically download models on first run. Simply:

1. **Create the model cache directory:**
   ```bash
   mkdir -p ~/.cache/nim
   chmod -R 777 ~/.cache/nim
   export HOST_NIM_CACHE=~/.cache/nim
   ```

2. **Start the stack:**
   ```bash
   ./scripts/run_dashboard_stack.sh up -d --build
   ```

   The containers will automatically download models on startup:
   - **RFDiffusion**: ~2-3GB (downloads automatically)
   - **AlphaFold2**: ~3-4GB (downloads automatically, but databases require ~2.2TB for full functionality)
   - **ProteinMPNN**: Included in repository

### Option 2: Manual Model Download

Use the provided script to download models before starting services:

```bash
# Run the initialization script
./scripts/init_models.sh

# Or download specific models
./scripts/download_models_arm64.sh
```

This script will:
- Download RFDiffusion models from IPD UW
- Download AlphaFold2 model parameters from Google Cloud
- Set up ProteinMPNN models (if not already present)
- Create proper directory structure

## Model Locations

Models are stored in the volume mounted at `/models` inside containers:

```
~/.cache/nim/
├── alphafold/
│   ├── params/
│   │   ├── params_model_1_ptm.npz
│   │   ├── params_model_2_ptm.npz
│   │   └── ...
│   └── .initialized
├── rfdiffusion/
│   ├── Base_ckpt.pt
│   ├── Complex_base_ckpt.pt
│   └── .initialized
└── proteinmpnn/
    ├── weights/
    └── .initialized
```

## Configuration Options

### Environment Variables

Set these in your environment or in [docker-compose-dashboard-arm64-native.yaml](../deploy/docker-compose-dashboard-arm64-native.yaml):
  
- **`HOST_NIM_CACHE`**: Host directory for model storage
  - Default: `~/.cache/nim`
  
- **`ALPHAFOLD_DATA_DIR`**: AlphaFold model directory inside container
  - Default: `/models/alphafold`
  
- **`RFDIFFUSION_MODEL_DIR`**: RFDiffusion model directory inside container
  - Default: `/models/rfdiffusion`
  
- **`PROTEINMPNN_MODEL_DIR`**: ProteinMPNN model directory inside container
  - Default: `/models/proteinmpnn`

### Service Readiness Logic

Each service is considered ready only when required assets are present (or the job is running under CI with `CI=1`).

## Checking Service Status

```bash
# Check overall health
curl http://localhost:8011/health

# Check detailed service status
curl http://localhost:8011/api/services/status | jq

# Check individual service health
curl http://localhost:18081/v1/health/ready  # AlphaFold2
curl http://localhost:18082/v1/health/ready  # RFDiffusion
curl http://localhost:18083/v1/health/ready  # ProteinMPNN
```

Service status will show:
- `"status": "ready"` - Service has models and is ready
- `"status": "not_ready"` - Models not found and mock disabled
- `"reason": "..."` - Explanation of current state

## Troubleshooting

### Services show "not_ready"

**Problem**: Services report 503 and "not_ready" status

**Solutions**:
1. Download models: `./scripts/init_models.sh`
3. Check model directory permissions: `chmod -R 777 ~/.cache/nim`
4. Verify volume mounts in docker-compose file

### Models not downloading automatically

**Problem**: Init script runs but models aren't present

**Solutions**:
1. Check internet connectivity
2. Verify `wget` or `curl` is installed: `docker exec <container> which wget`
3. Check disk space: `df -h ~/.cache/nim`
4. Run init script manually:
   ```bash
   docker exec -it protein-binder-dashboard-arm64-alphafold-1 /app/init_models.sh
   ```

### AlphaFold2 still not ready with models

**Note**: AlphaFold2 requires large genetic databases (~2.2TB) for full functionality. The model parameters alone (~3-4GB) are insufficient for production use.

**For production**: Follow the [official AlphaFold database setup](https://github.com/deepmind/alphafold#genetic-databases)

## Storage Requirements

- **Mock Mode**: <1GB (no models needed)
- **RFDiffusion**: ~2-3GB
- **AlphaFold2 (models only)**: ~3-4GB
- **AlphaFold2 (full databases)**: ~2.2TB
- **ProteinMPNN**: <100MB

## Advanced: Custom Model Locations

To use custom model directories:

1. Edit [docker-compose-dashboard-arm64-native.yaml](../deploy/docker-compose-dashboard-arm64-native.yaml):
   ```yaml
   volumes:
     - /my/custom/model/path:/models
   ```

2. Or set environment variable:
   ```bash
   export HOST_NIM_CACHE=/my/custom/model/path
   ```

## Performance Notes

- **Mock mode**: Instant response, suitable for UI testing and development
- **Real models**: GPU-accelerated inference, production-quality results
- **First run with auto-download**: May take 10-30 minutes depending on internet speed

## Next Steps

After setup:
1. Access the dashboard: http://localhost:3000
2. Check service status: http://localhost:8011/api/services/status
3. Run a test job through the UI or API
4. Monitor logs: `./scripts/run_dashboard_stack.sh logs -f`

For more information:
- [Architecture Documentation](../docs/ARCHITECTURE.md)
- [ARM64 Deployment Guide](../docs/ARM64_DEPLOYMENT.md)
- [Docker MCP README](../docs/DOCKER_MCP_README.md)
