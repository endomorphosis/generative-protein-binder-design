# ARM64 Native Installation Guide

## Overview

This guide provides detailed instructions for installing AlphaFold2, RFDiffusion, and ProteinMPNN natively on ARM64 systems. This approach avoids Docker container platform compatibility issues but requires significant technical expertise and time investment.

## NIM-Compatible Local Services (Recommended integration)

If you want the Docker dashboard/MCP server stack to route to your **host-native** AlphaFold2 + RFdiffusion installs (no shims), this repo includes small NIM-compatible HTTP wrappers.

- Start the host-native services:
    - `./scripts/run_arm64_native_model_services.sh`
    - This runs `native_services.alphafold_service` on `:18081` and `native_services.rfdiffusion_service` on `:18082`.
    - You must set:
        - `ALPHAFOLD_NATIVE_CMD` (must produce `{out_dir}/result.pdb`)
        - `RFDIFFUSION_NATIVE_CMD` (must produce `{out_dir}/design_{design_id}.pdb`)

- Start the dashboard stack that routes to those host-native services:
    - `./scripts/run_dashboard_stack.sh --arm64-host-native up -d --build`

This mode removes the “CI-only shim” containers from the critical path by pointing the MCP server at your real local installs via `http://host.docker.internal:18081` and `:18082`.

⚠️ **Warning:** Native installation is complex and may take several days to complete. It requires:
- Deep understanding of Python environments and dependency management
- Experience compiling software from source
- Familiarity with CUDA/GPU programming (if using GPU acceleration)
- Patience for debugging compatibility issues

## System Requirements

### Hardware Requirements
- **CPU:** Modern ARM64 processor (Apple Silicon M1/M2/M3 or ARM-based servers)
- **RAM:** Minimum 32GB (64GB+ recommended)
- **Storage:** At least 500GB free space for models and dependencies
- **GPU (Optional):** NVIDIA GPU with CUDA support or Apple Silicon GPU (via Metal)

### Software Prerequisites
- **OS:** Ubuntu 22.04 ARM64, macOS (Apple Silicon), or compatible ARM64 Linux distribution
- **Python:** 3.9 or 3.10 (some tools may not support Python 3.11+)
- **Conda/Mamba:** For environment management (highly recommended)
- **Build Tools:** gcc, g++, cmake, make
- **Git:** For cloning repositories

## Installation Steps

### Step 1: Prepare Your System

#### Install System Dependencies (Ubuntu/Debian ARM64)

```bash
# Update package list
sudo apt update && sudo apt upgrade -y

# Install build essentials and system libraries
sudo apt install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    libhdf5-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    pkg-config \
    python3-dev \
    python3-pip \
    python3-venv

# Install additional scientific computing libraries
sudo apt install -y \
    libffi-dev \
    libssl-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    llvm \
    libncurses5-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    liblzma-dev
```

#### Install System Dependencies (macOS with Homebrew)

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install required packages
brew install cmake wget git hdf5 openblas lapack gcc@11
brew install python@3.10
```

#### Install Conda/Mamba (Recommended for Environment Management)

```bash
# Install Miniforge for ARM64 (includes conda and mamba)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
bash Miniforge3-Linux-aarch64.sh -b -p $HOME/miniforge3

# For macOS Apple Silicon:
# wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
# bash Miniforge3-MacOSX-arm64.sh -b -p $HOME/miniforge3

# Add to PATH
echo 'export PATH="$HOME/miniforge3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Initialize conda
conda init bash
source ~/.bashrc
```

### Step 2: Install AlphaFold2 for ARM64

AlphaFold2 requires building from source for ARM64 architecture.

#### Clone AlphaFold Repository

```bash
# Create working directory
mkdir -p ~/protein-tools
cd ~/protein-tools

# Clone AlphaFold
git clone https://github.com/deepmind/alphafold.git
cd alphafold
```

#### Create Conda Environment

```bash
# Create new environment for AlphaFold
conda create -n alphafold python=3.10 -y
conda activate alphafold

# Install core dependencies
conda install -c conda-forge \
    numpy \
    scipy \
    pandas \
    matplotlib \
    jupyter \
    ipython \
    pytest \
    absl-py \
    -y
```

#### Install JAX for ARM64

JAX is a critical dependency for AlphaFold. For ARM64, you need to build from source or use ARM64-compatible wheels.

**Option A: Install Pre-built JAX (if available for your platform)**

```bash
# For CPU-only JAX on ARM64
pip install --upgrade pip
pip install --upgrade "jax[cpu]"
```

**Option B: Build JAX from Source (Advanced)**

```bash
# Install Bazel (build tool)
conda install -c conda-forge bazel -y

# Clone JAX repository
cd ~/protein-tools
git clone https://github.com/google/jax.git
cd jax

# Build and install JAX
python build/build.py --enable_cuda=false
pip install dist/*.whl

# Build and install jaxlib
cd ..
git clone https://github.com/google/jax.git jaxlib-source
cd jaxlib-source
python build/build.py --enable_cuda=false
pip install dist/*.whl
```

#### Install AlphaFold Dependencies

```bash
cd ~/protein-tools/alphafold

# Install Python dependencies
pip install -r docker/requirements.txt

# Install additional required packages
pip install tensorflow-cpu==2.12.0  # or tensorflow if GPU is available
pip install dm-haiku==0.0.10
pip install ml-collections==0.1.1
pip install biopython==1.81
pip install chex==0.1.82
pip install dm-tree==0.1.8
pip install immutabledict==2.2.3
pip install jax==0.4.13
pip install jaxlib==0.4.13
pip install numpy==1.24.3
```

#### Download AlphaFold Databases

AlphaFold requires large reference databases. This step requires significant disk space (~2.2TB).

```bash
# Create directory for databases
mkdir -p ~/alphafold_databases
cd ~/alphafold_databases

# Download databases using AlphaFold's script
# Note: This will take several hours
bash ~/protein-tools/alphafold/scripts/download_all_data.sh ~/alphafold_databases

# Alternatively, download minimal databases for testing:
# - UniRef90 (required, ~60GB)
# - MGnify (required, ~120GB)
# - BFD or Small BFD (required, ~1.7TB or ~17GB)
# - PDB70 (required, ~56GB)
# - PDB mmCIF files (required, ~200GB)
# - UniRef30 (optional, used for multimer)
# - UniProt (optional, used for multimer)
```

#### Configure AlphaFold

```bash
# Set environment variables
echo 'export ALPHAFOLD_DIR=~/protein-tools/alphafold' >> ~/.bashrc
echo 'export ALPHAFOLD_DATA_DIR=~/alphafold_databases' >> ~/.bashrc
source ~/.bashrc
```

#### Test AlphaFold Installation

```bash
cd ~/protein-tools/alphafold

# Run AlphaFold on a test sequence
python run_alphafold.py \
    --fasta_paths=example/query.fasta \
    --max_template_date=2020-05-14 \
    --model_preset=monomer \
    --db_preset=reduced_dbs \
    --data_dir=$ALPHAFOLD_DATA_DIR \
    --output_dir=./output \
    --use_gpu=false
```

### Step 3: Install RFDiffusion

RFDiffusion is used for generating protein backbones.

#### Clone RFDiffusion Repository

```bash
cd ~/protein-tools
git clone https://github.com/RosettaCommons/RFdiffusion.git
cd RFdiffusion
```

#### Create Conda Environment for RFDiffusion

```bash
# Create environment
conda create -n rfdiffusion python=3.10 -y
conda activate rfdiffusion

# Install PyTorch for ARM64
# For CPU-only:
pip install torch torchvision torchaudio

# For NVIDIA GPU (if compatible):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Install RFDiffusion Dependencies

```bash
# Install from requirements or environment file if available
if [ -f environment.yaml ]; then
    conda env update -f environment.yaml
elif [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

# Install additional dependencies
pip install \
    numpy \
    pandas \
    biopython \
    scipy \
    matplotlib \
    hydra-core \
    omegaconf \
    pyrosetta-installer
```

#### Install SE(3)-Transformer (Required for RFDiffusion)

```bash
cd ~/protein-tools
git clone https://github.com/FabianFuchsML/se3-transformer-public.git
cd se3-transformer-public
pip install -e .
```

#### Download RFDiffusion Model Weights

```bash
# Create directory for model weights
mkdir -p ~/protein-tools/RFdiffusion/models

# Download pre-trained weights
cd ~/protein-tools/RFdiffusion/models

# You may need to manually download weights from the RFDiffusion repository
# or use wget/curl if direct links are available
# Example (check RFDiffusion documentation for current URLs):
# wget http://files.ipd.uw.edu/pub/RFdiffusion/models/Base_ckpt.pt
# wget http://files.ipd.uw.edu/pub/RFdiffusion/models/Complex_base_ckpt.pt
```

#### Configure RFDiffusion

```bash
# Set environment variables
echo 'export RFDIFFUSION_DIR=~/protein-tools/RFdiffusion' >> ~/.bashrc
source ~/.bashrc
```

#### Test RFDiffusion Installation

```bash
cd ~/protein-tools/RFdiffusion

# Run a test generation
python scripts/run_inference.py \
    --config-name=base \
    inference.output_prefix=test_output \
    inference.num_designs=1 \
    'contigmap.contigs=[50-50]'
```

### Step 4: Install ProteinMPNN

ProteinMPNN is used for designing sequences given a protein backbone.

#### Clone ProteinMPNN Repository

```bash
cd ~/protein-tools
git clone https://github.com/dauparas/ProteinMPNN.git
cd ProteinMPNN
```

#### Create Conda Environment for ProteinMPNN

```bash
# Create environment
conda create -n proteinmpnn python=3.10 -y
conda activate proteinmpnn

# Install PyTorch (if not already installed)
pip install torch torchvision torchaudio
```

#### Install ProteinMPNN Dependencies

```bash
# Install dependencies
pip install \
    numpy \
    biopython \
    prody \
    matplotlib \
    pandas \
    scipy
```

#### Download ProteinMPNN Model Weights

```bash
# Model weights should be in the repository
# If not, download them manually
cd ~/protein-tools/ProteinMPNN

# Check for vanilla_model_weights directory
ls -la vanilla_model_weights/

# If not present, you may need to download from the repository releases
```

#### Test ProteinMPNN Installation

```bash
cd ~/protein-tools/ProteinMPNN

# Run a test sequence design
python protein_mpnn_run.py \
    --pdb_path ./examples/1BC8.pdb \
    --out_folder ./test_output \
    --num_seq_per_target 2 \
    --sampling_temp "0.1" \
    --seed 37 \
    --batch_size 1
```

### Step 5: Install AlphaFold2-Multimer

AlphaFold2-Multimer is included in the main AlphaFold installation but requires specific configuration.

```bash
# Activate AlphaFold environment
conda activate alphafold

# Ensure multimer-specific databases are downloaded
cd ~/alphafold_databases

# Download UniRef30 if not already downloaded
# wget -c http://wwwuser.gwdg.de/~compbiol/uniclust/2021_03/UniRef30_2021_03_hhsuite.tar.gz
# tar -xzvf UniRef30_2021_03_hhsuite.tar.gz

# Download UniProt for multimer
# wget -c ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
# gunzip uniprot_sprot.fasta.gz
```

#### Test AlphaFold2-Multimer

```bash
cd ~/protein-tools/alphafold

# Run multimer prediction
python run_alphafold.py \
    --fasta_paths=example/multimer.fasta \
    --max_template_date=2020-05-14 \
    --model_preset=multimer \
    --db_preset=reduced_dbs \
    --data_dir=$ALPHAFOLD_DATA_DIR \
    --output_dir=./output_multimer \
    --use_gpu=false
```

## Integration with the Blueprint Workflow

To use the natively installed tools with this blueprint's Jupyter notebook, you'll need to modify the API calls to use local Python functions instead of HTTP requests.

### Create Python Wrapper Scripts

Create wrapper scripts that mimic the NIM container APIs but call the local installations:

```bash
mkdir -p ~/protein-tools/wrappers
```

#### AlphaFold2 Wrapper (~/protein-tools/wrappers/alphafold_wrapper.py)

```python
#!/usr/bin/env python3
"""
AlphaFold2 local wrapper that mimics NIM API
"""
import sys
import os
import json
from pathlib import Path

# Add AlphaFold to path
sys.path.append(os.path.expanduser('~/protein-tools/alphafold'))

def predict_structure(sequence, output_dir):
    """Run AlphaFold prediction locally"""
    # Import AlphaFold modules
    from alphafold.run_alphafold import predict_structure_from_sequence
    
    # Configure paths
    data_dir = os.environ.get('ALPHAFOLD_DATA_DIR', os.path.expanduser('~/alphafold_databases'))
    
    # Run prediction
    results = predict_structure_from_sequence(
        sequence=sequence,
        output_dir=output_dir,
        data_dir=data_dir,
        model_preset='monomer',
        use_gpu=False
    )
    
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sequence', required=True)
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    
    results = predict_structure(args.sequence, args.output_dir)
    print(json.dumps(results))
```

#### RFDiffusion Wrapper (~/protein-tools/wrappers/rfdiffusion_wrapper.py)

```python
#!/usr/bin/env python3
"""
RFDiffusion local wrapper that mimics NIM API
"""
import sys
import os
import json

sys.path.append(os.path.expanduser('~/protein-tools/RFdiffusion'))

def generate_backbone(target_length, output_dir, num_designs=1):
    """Run RFDiffusion locally"""
    # Import RFDiffusion modules
    from run_inference import main as rf_main
    
    # Configure and run
    results = rf_main(
        contig=f'{target_length}-{target_length}',
        output_dir=output_dir,
        num_designs=num_designs
    )
    
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--length', type=int, required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_designs', type=int, default=1)
    args = parser.parse_args()
    
    results = generate_backbone(args.length, args.output_dir, args.num_designs)
    print(json.dumps(results))
```

#### ProteinMPNN Wrapper (~/protein-tools/wrappers/proteinmpnn_wrapper.py)

```python
#!/usr/bin/env python3
"""
ProteinMPNN local wrapper that mimics NIM API
"""
import sys
import os
import json

sys.path.append(os.path.expanduser('~/protein-tools/ProteinMPNN'))

def design_sequence(pdb_path, output_dir, num_sequences=1):
    """Run ProteinMPNN locally"""
    # Import ProteinMPNN modules
    from protein_mpnn_run import main as mpnn_main
    
    # Configure and run
    results = mpnn_main(
        pdb_path=pdb_path,
        out_folder=output_dir,
        num_seq_per_target=num_sequences,
        sampling_temp=0.1
    )
    
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb_path', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_sequences', type=int, default=1)
    args = parser.parse_args()
    
    results = design_sequence(args.pdb_path, args.output_dir, args.num_sequences)
    print(json.dumps(results))
```

### Modify Jupyter Notebook

Update the notebook to use local installations instead of API calls. Replace HTTP requests with subprocess calls to wrapper scripts:

```python
# Instead of:
# response = requests.post('http://localhost:18081/v1/predict', json=data)
# (Some legacy/custom setups may publish 8081 instead.)

# Use:
import subprocess
import json

result = subprocess.run(
    ['python', '~/protein-tools/wrappers/alphafold_wrapper.py',
     '--sequence', sequence,
     '--output_dir', output_dir],
    capture_output=True,
    text=True
)
response_data = json.loads(result.stdout)
```

## Troubleshooting

### Common Issues and Solutions

#### JAX Installation Issues on ARM64

**Problem:** JAX wheels not available for ARM64
**Solution:** Build JAX from source or use CPU-only version

```bash
pip install --upgrade "jax[cpu]"
```

#### CUDA/GPU Issues on ARM64

**Problem:** NVIDIA CUDA not working on ARM64
**Solution:** Use CPU-only versions or investigate CUDA ARM64 support

```bash
# Use CPU versions of dependencies
export CUDA_VISIBLE_DEVICES=""
```

#### Memory Issues

**Problem:** Out of memory errors during model execution
**Solution:** Reduce batch sizes, use model checkpointing, or add swap space

```bash
# Add swap space (Linux)
sudo fallocate -l 64G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### Missing Model Weights

**Problem:** Model weights not found
**Solution:** Manually download from official sources and place in correct directories

```bash
# Check RFDiffusion GitHub releases for model weights
# Check AlphaFold documentation for database downloads
```

#### Python Version Conflicts

**Problem:** Incompatible Python versions between tools
**Solution:** Use separate conda environments for each tool

```bash
conda activate alphafold    # For AlphaFold
conda activate rfdiffusion  # For RFDiffusion
conda activate proteinmpnn  # For ProteinMPNN
```

### Performance Optimization

#### Use Mamba Instead of Conda

Mamba is faster for dependency resolution:

```bash
conda install mamba -c conda-forge
mamba install <packages>  # Use mamba instead of conda
```

#### Enable Multi-threading

Set environment variables for parallel processing:

```bash
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
```

#### Use Local Caching

Cache computed results to avoid recomputation:

```bash
export ALPHAFOLD_CACHE_DIR=~/.cache/alphafold
mkdir -p $ALPHAFOLD_CACHE_DIR
```

#### Optional: Pre-warm AlphaFold DBs in RAM (Evictable)

On Linux, large read-heavy databases benefit from the kernel page cache.
You can pre-warm the key AlphaFold DB files so the first run after boot is faster.

This does **not** lock/pin memory; the cache is automatically evicted under memory pressure.

```bash
bash scripts/warm_alphafold_page_cache.sh --data-dir ~/.cache/alphafold --min-mem-gb 6
```

If you want extra protection against transient memory spikes, enabling swap (or zram-backed swap) is a good mitigation.

#### Continuous Memory Safety (Recommended for Non-Technical Users)

When using the ARM64 host-native mode, `scripts/start_everything.sh` starts a small background watchdog that:

- Monitors `MemAvailable` continuously.
- Under memory pressure, evicts AlphaFold DB file pages from the Linux page cache (best-effort).
- Does **not** lock memory and does **not** drop global caches.

You can also run a read-only report to help users understand what they are seeing:

```bash
bash scripts/check_memory_safety.sh
```

## Validation and Testing

### Verify All Components

```bash
# Test AlphaFold
conda activate alphafold
python -c "import alphafold; print('AlphaFold OK')"

# Test RFDiffusion
conda activate rfdiffusion
python -c "import torch; print('RFDiffusion environment OK')"

# Test ProteinMPNN
conda activate proteinmpnn
python -c "import torch; print('ProteinMPNN environment OK')"
```

### Run End-to-End Test

```bash
# Create test directory
mkdir -p ~/protein-tools/test
cd ~/protein-tools/test

# Run complete workflow
# 1. Generate backbone with RFDiffusion
# 2. Design sequence with ProteinMPNN
# 3. Predict structure with AlphaFold
# 4. Predict complex with AlphaFold-Multimer
```

## Maintenance and Updates

### Update Tools

```bash
# Update AlphaFold
cd ~/protein-tools/alphafold
git pull origin main
conda activate alphafold
pip install -r docker/requirements.txt --upgrade

# Update RFDiffusion
cd ~/protein-tools/RFdiffusion
git pull origin main
conda activate rfdiffusion
pip install -r requirements.txt --upgrade

# Update ProteinMPNN
cd ~/protein-tools/ProteinMPNN
git pull origin main
conda activate proteinmpnn
pip install -r requirements.txt --upgrade
```

### Database Updates

AlphaFold databases should be updated periodically:

```bash
cd ~/alphafold_databases
# Re-run download script or manually update databases
bash ~/protein-tools/alphafold/scripts/download_all_data.sh .
```

## Alternative: Docker with Rosetta/Emulation

If native installation proves too complex, consider using Docker with platform emulation as a middle-ground approach. See [ARM64_COMPATIBILITY.md](ARM64_COMPATIBILITY.md) for details.

## Additional Resources

- [AlphaFold GitHub](https://github.com/deepmind/alphafold)
- [RFDiffusion GitHub](https://github.com/RosettaCommons/RFdiffusion)
- [ProteinMPNN GitHub](https://github.com/dauparas/ProteinMPNN)
- [JAX Installation Guide](https://github.com/google/jax#installation)
- [PyTorch ARM64 Installation](https://pytorch.org/get-started/locally/)
- [Conda for ARM64](https://github.com/conda-forge/miniforge)

## Support and Community

For issues specific to ARM64 installations:
1. Check tool-specific GitHub issues
2. ARM64/Apple Silicon community forums
3. BioConda community for package availability

## Time and Skill Estimates

**Estimated Installation Time:**
- Experienced users: 2-3 days
- Intermediate users: 4-7 days
- Beginners: 1-2 weeks (with debugging)

**Required Skill Level:**
- Python package management: Advanced
- Linux/macOS command line: Intermediate-Advanced
- Debugging: Advanced
- Patience: Expert level

**Recommendation:** Unless you have significant experience with bioinformatics software installation and debugging, we strongly recommend using the Docker-based approach with platform emulation or cloud-based instances with x86_64 architecture.
