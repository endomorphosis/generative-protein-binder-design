# NVIDIA BioNeMo Blueprint: Protein Binder Design

![A workflow diagram of the Protein Design Blueprint](docs/Protein_Design_Architecture_Diagram.png)

The NVIDIA BioNeMo Blueprint for protein binder design shows how generative AI and accelerated NIM microservices can be used to design binders to a target protein sequence smarter and faster. This workflow simplifies the process of _in silico_ protein binder design by automatically generating binder sequences and predicted structures for the binder and target.

This Blueprint takes as input a valid amino acid protein sequence. It utilizes the following models:

- **AlphaFold2**: A deep learning model for predicting protein structure from amino acid sequence, originally developed by DeepMind.
- **ProteinMPNN**: a deep learning model for predicting amino acid sequences for protein backbones.
- **RFDiffusion**: a generative model of protein backbones for protein binder design.
- **AlphaFold2-Multimer**: A deep learning model for predicting protein structure of multimers from a list of amino acid sequences, originally developed by DeepMind.

Once completed, this Blueprint outputs predicted multimer structures (in PDB format) for the target protein sequence and any generated peptide binders. These binder-target multimeric structures can then be assessed to find binders that effectively bind the target protein.

## System Requirements

The docker compose setup for this NIM Agent Blueprint requires the following specifications:
- At least 1300 GB (1.3 TB) of fast NVMe SSD space
- A modern CPU with at least 24 CPU cores
- At least 64 GB of RAM
- Two or more NVIDIA L40s, A100, or H100 GPUs

### Platform Support

This project supports both **AMD64** and **ARM64** architectures:

- **AMD64/x86_64** (Recommended): Native support for all NVIDIA NIM containers with optimal performance
- **ARM64/aarch64**: Supported with Docker emulation or native installation
  - Docker approach: AMD64 containers run via emulation (may have performance impact)
  - Native approach: Install tools directly on ARM64 for better performance

Run `./scripts/detect_platform.sh` to check your system and get platform-specific recommendations.

## Get Started

New here (or not an ML engineer)? Start with: [START_HERE.md](START_HERE.md)

### Quick Platform Check
```bash
./scripts/detect_platform.sh
```
This script will detect your system architecture and provide tailored recommendations.

### Documentation

#### General Setup
- [🚀 Local Setup Guide](docs/LOCAL_SETUP.md) - Comprehensive local development setup
- [📋 System Verification Report](docs/SYSTEM_VERIFICATION.md) - Check your system compatibility
- [Deploy with Docker Compose](deploy)
- [Deploy with Helm](protein-design-chart)
- [Deploy with MCP Server and Dashboard](docs/DOCKER_MCP_README.md) ⭐ **New!**
- [Deploy Natively on DGX Spark](docs/DGX_SPARK_NATIVE_DEPLOYMENT.md) ⭐ **New!**
- [Source code](src)

#### ARM64-Specific Guides
- [📘 **ARM64 Complete Guide**](docs/ARM64_COMPLETE_GUIDE.md) - **NEW: Comprehensive ARM64 support guide**
- [🔌 **ARM64 CUDA Fallback Guide**](docs/ARM64_CUDA_FALLBACK_GUIDE.md) - **NEW: Fallback solutions for ARM64 CUDA**
- [🚀 ARM64 Quick Start](docs/ARM64_QUICK_START.md) - Quick start for ARM64 systems
- [✅ ARM64 Completion Checklist](docs/ARM64_COMPLETION_CHECKLIST.md) - Step-by-step guide to complete ARM64 porting
- [🤖 ARM64 Automation Summary](docs/ARM64_AUTOMATION_SUMMARY.md) - Overview of automated ARM64 porting
- [🏗️ ARM64 Deployment Guide](docs/ARM64_DEPLOYMENT.md) - Complete guide for ARM64 deployment
- [⚙️ ARM64 Compatibility Guide](docs/ARM64_COMPATIBILITY.md) - Understanding ARM64 compatibility
- [🔧 ARM64 Native Installation](docs/ARM64_NATIVE_INSTALLATION.md) - Advanced: Install tools natively on ARM64

## Quick Start

### Automated Setup (Recommended)
```bash
./scripts/setup_local.sh
```

### Manual Setup
Deploy the blueprint using [Docker Compose](deploy) or [Helm](protein-design-chart)

### Option 1: Full Stack with MCP Server & Dashboard (Recommended)
Deploy the complete stack including MCP server, web dashboard, and Jupyter:
```bash
export NGC_CLI_API_KEY=<your-key>
export HOST_NIM_CACHE=~/.cache/nim
docker compose -f deploy/docker-compose-full.yaml up
```

Access the services:
- **MCP Dashboard**: http://localhost:3000 (Web UI for job submission and monitoring)
- **Jupyter Notebook**: http://localhost:8888 (Interactive notebooks)
- **MCP Server API**: http://localhost:8000/docs (API documentation)

### Option 2: Native Deployment on DGX Spark ⭐ **New!**
Run models directly on DGX Spark hardware without NIM containers:
```bash
# Install models natively (see docs/DGX_SPARK_NATIVE_DEPLOYMENT.md for details)
export MODEL_BACKEND=native
cd mcp-server
./start-native.sh
```

Benefits:
- ✅ 5-10x lower latency
- ✅ 3x higher throughput  
- ✅ 50% memory reduction
- ✅ Direct GPU control
- ✅ No container overhead

See [DGX Spark Native Deployment Guide](docs/DGX_SPARK_NATIVE_DEPLOYMENT.md) for complete installation and configuration instructions.

### Option 3: Hybrid Mode (Native + NIM Fallback)
Best for gradual migration - tries native execution first, falls back to NIM if unavailable:
```bash
export MODEL_BACKEND=hybrid
cd mcp-server
./start-hybrid.sh
```

### Option 4: NIMs Only
Deploy just the NIM services using [Docker Compose](deploy) or [Helm](protein-design-chart):
```bash
cd ./src
jupyter notebook
```

For detailed setup instructions, see [LOCAL_SETUP.md](LOCAL_SETUP.md)

## Set Up With Docker Compose

Navigate to the [deploy](deploy) directory to learn how to start up the NIMs.

## Set up with Helm

Follow the instructions in the [protein-design-chart](protein-design-chart) directory and deploy the Helm chart

## Notebook

An example of how to call each protein binder design step is located in [src/protein-binder-design.ipynb](src/protein-binder-design.ipynb)
