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

## Get Started

- [Deploy with Docker Compose](deploy)
- [Deploy with Helm](protein-design-chart)
- [Deploy with MCP Server and Dashboard](docs/DOCKER_MCP_README.md) ⭐ **New!**
- [Deploy Natively on DGX Spark](docs/DGX_SPARK_NATIVE_DEPLOYMENT.md) ⭐ **New!**
- [Source code](src)

## Quick Start

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

## Set Up With Docker Compose

Navigate to the [deploy](deploy) directory to learn how to start up the NIMs.

## Set up with Helm

Follow the instructions in the [protein-design-chart](protein-design-chart) directory and deploy the Helm chart

## Notebook

An example of how to call each protein binder design step is located in [src/protein-binder-design.ipynb](src/protein-binder-design.ipynb)
