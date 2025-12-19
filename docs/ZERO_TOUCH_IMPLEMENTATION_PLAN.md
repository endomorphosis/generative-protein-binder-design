# AlphaFold2 + RFDiffusion Zero-Touch Local Implementation Workplan

## Executive Summary

Based on repository analysis, the current state shows:
- ✅ **Architecture exists**: Native backend framework with pluggable model support
- ✅ **ARM64 support exists**: Scripts and documentation for ARM64 deployment
- ⚠️ **Model provisioning**: Partial automation exists but needs completion
- ❌ **Zero-touch installation**: Not yet implemented for AlphaFold2/RFDiffusion

## Current State Assessment

### What Works Today
1. **MCP Server Framework**: Full backend routing (NIM/Native/Hybrid)
2. **Native Service Wrappers**: FastAPI services for AlphaFold2 and RFDiffusion
3. **Model Download Scripts**: init_models.sh and download_models_arm64.sh (partial)
4. **Docker Orchestration**: Multiple compose files for different deployment modes
5. **Documentation**: Comprehensive ARM64 and native installation guides

### What's Missing
1. **Automated Model Installation**: Scripts download stubs but don't install full toolchains
2. **Database Provisioning**: AlphaFold2 requires ~2.2TB databases (not automated)
3. **Dependency Management**: No automated setup for native Python environments
4. **End-to-End Testing**: No validation that full pipeline works locally
5. **Zero-Touch Experience**: Manual intervention still required

## Implementation Strategy

### Phase 1: Automated Toolchain Installation (Priority: HIGH)

#### Task 1.1: Create AlphaFold2 Zero-Touch Installer
**Location**: `scripts/install_alphafold2_complete.sh`

```bash
#!/bin/bash
# Complete AlphaFold2 installation with all dependencies

# 1. Detect architecture and OS
# 2. Install system dependencies (apt/brew)
# 3. Create conda environment
# 4. Install JAX (CPU or GPU based on hardware)
# 5. Install AlphaFold2 from source
# 6. Download model parameters (~3GB)
# 7. Download reduced databases (~50GB) or full (~2.2TB)
# 8. Configure environment variables
# 9. Validate installation with test sequence
```

**Key Features**:
- Platform detection (x86_64/ARM64, Linux/macOS)
- GPU detection (NVIDIA CUDA/Apple Metal/CPU-only)
- Progressive database download (minimal → reduced → full)
- Validation steps at each stage
- Rollback on failure

#### Task 1.2: Create RFDiffusion Zero-Touch Installer
**Location**: `scripts/install_rfdiffusion_complete.sh`

```bash
#!/bin/bash
# Complete RFDiffusion installation

# 1. Detect architecture and OS
# 2. Install PyTorch (CPU/CUDA/Metal)
# 3. Clone RFDiffusion repository
# 4. Install SE(3)-Transformer dependency
# 5. Download model weights (~2-3GB)
# 6. Configure environment
# 7. Validate with test backbone generation
```

#### Task 1.3: Unified Installer Script
**Location**: `scripts/install_all_native.sh`

```bash
#!/bin/bash
# One-command installation of entire stack

# Options:
#   --minimal: Model params only (~5GB)
#   --reduced: Reduced databases (~50GB)
#   --full: Complete databases (~2.3TB)
#   --cpu-only: Skip GPU dependencies
#   --platform=arm64|x86_64: Force architecture

# Installs:
# 1. AlphaFold2 (with database options)
# 2. RFDiffusion
# 3. ProteinMPNN
# 4. Native service wrappers
# 5. MCP server dependencies
```

### Phase 2: Database Provisioning Strategy (Priority: HIGH)

#### Challenge: AlphaFold2 Databases (~2.2TB)

**Solution 1: Tiered Database Approach**
```
Tier 1 - Minimal (5GB):
  - Model parameters only
  - Good for: Testing, CI/CD
  - Accuracy: Demo quality

Tier 2 - Reduced (50GB):
  - Small BFD, reduced_dbs preset
  - Good for: Development, small proteins
  - Accuracy: 70-80% of full

Tier 3 - Full (2.2TB):
  - Complete databases
  - Good for: Production, research
  - Accuracy: 100% (state-of-the-art)
```

**Implementation**:
```bash
# scripts/download_alphafold_databases.sh
#!/bin/bash
TIER=${1:-minimal}  # minimal | reduced | full

case $TIER in
  minimal)
    # Download only model parameters
    download_alphafold_params
    ;;
  reduced)
    # Download reduced databases
    download_small_bfd
    download_reduced_uniref
    ;;
  full)
    # Download complete databases
    run_official_download_script
    ;;
esac
```

**Solution 2: Remote Database Access**
- Mount shared network storage (NFS/SMB)
- Use cloud storage (S3/GCS) with caching
- Database-as-a-Service (centralized lab server)

#### Task 2.1: Implement Database Tiering
**Files**:
- `scripts/download_alphafold_databases.sh`
- `native_services/alphafold_service.py` (add db_preset parameter)
- `mcp-server/backends/native_backend.py` (route to appropriate preset)

#### Task 2.2: Network Database Support
**Files**:
- `scripts/mount_shared_databases.sh`
- `docs/SHARED_DATABASE_SETUP.md`

### Phase 3: Native Service Implementation (Priority: MEDIUM)

#### Current State
- ✅ Wrapper services exist: `native_services/alphafold_service.py`, `rfdiffusion_service.py`
- ✅ Use environment variables: `ALPHAFOLD_NATIVE_CMD`, `RFDIFFUSION_NATIVE_CMD`
- ❌ No direct Python API integration (relies on subprocess calls)

#### Task 3.1: Direct Python API Integration

**Option A: Keep Subprocess Approach (Simpler)**
```python
# native_services/alphafold_service.py

def get_alphafold_cmd():
    """Generate command based on installation type"""
    install_type = env_str("ALPHAFOLD_INSTALL_TYPE", "docker")
    
    if install_type == "native":
        # Use native conda environment
        conda_env = env_str("ALPHAFOLD_CONDA_ENV", "alphafold")
        base_cmd = f"conda run -n {conda_env} python {ALPHAFOLD_DIR}/run_alphafold.py"
    elif install_type == "docker":
        # Use Docker image
        base_cmd = "docker run --rm alphafold:latest"
    
    return base_cmd + " --fasta={fasta} --output_dir={out_dir} ..."
```

**Option B: Import Python Modules Directly (Faster, more complex)**
```python
# native_services/alphafold_native.py

import sys
import os
sys.path.insert(0, os.environ['ALPHAFOLD_PATH'])

from alphafold.model import model as alphafold_model
from alphafold.data import pipeline as alphafold_pipeline

class AlphaFoldNative:
    def __init__(self):
        self.model_runner = alphafold_model.RunModel(...)
        self.data_pipeline = alphafold_pipeline.DataPipeline(...)
    
    def predict(self, sequence: str) -> str:
        features = self.data_pipeline.process(sequence)
        prediction = self.model_runner.predict(features)
        return self.to_pdb(prediction)
```

**Recommendation**: Start with subprocess (Option A), migrate to direct API (Option B) for performance.

#### Task 3.2: Auto-Detection and Configuration

```python
# scripts/detect_native_installations.py

def detect_alphafold():
    """Detect if AlphaFold2 is installed and where"""
    
    # Check conda environments
    conda_envs = list_conda_environments()
    if 'alphafold' in conda_envs:
        return {
            'type': 'conda',
            'env': 'alphafold',
            'path': get_conda_env_path('alphafold'),
            'status': 'ready' if validate_alphafold() else 'incomplete'
        }
    
    # Check system Python
    try:
        import alphafold
        return {
            'type': 'system',
            'path': alphafold.__file__,
            'status': 'ready'
        }
    except ImportError:
        pass
    
    # Check Docker
    if check_docker_image('alphafold'):
        return {
            'type': 'docker',
            'image': 'alphafold:latest',
            'status': 'ready'
        }
    
    return {'type': 'none', 'status': 'not_installed'}
```

### Phase 4: Integration and Testing (Priority: MEDIUM)

#### Task 4.1: End-to-End Validation

**Script**: `scripts/validate_native_installation.sh`
```bash
#!/bin/bash
# Validate complete installation

echo "Testing AlphaFold2..."
result=$(curl -X POST http://localhost:18081/v1/structure \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKTAYIAKQRQISFVKSHFSRQ"}')

if [[ $result == *"pdb"* ]]; then
  echo "✓ AlphaFold2 working"
else
  echo "✗ AlphaFold2 failed"
  exit 1
fi

echo "Testing RFDiffusion..."
# Similar test...

echo "Testing end-to-end pipeline..."
# Run full workflow...
```

#### Task 4.2: CI/CD Integration
- Add native installation tests to GitHub Actions
- Test on ARM64 and x86_64 runners
- Cache installations to speed up tests

### Phase 5: Documentation and UX (Priority: LOW)

#### Task 5.1: Quick Start Guide
**File**: `docs/ZERO_TOUCH_QUICKSTART.md`

```markdown
# Zero-Touch Installation Quick Start

## One-Command Installation

### Minimal (5GB, CPU-only, ~15 minutes)
```bash
./scripts/install_all_native.sh --minimal --cpu-only
```

### Recommended (50GB, GPU, ~1 hour)
```bash
./scripts/install_all_native.sh --reduced --gpu
```

### Production (2.3TB, GPU, ~6 hours)
```bash
./scripts/install_all_native.sh --full --gpu
```

## Verification
```bash
./scripts/validate_native_installation.sh
```

## Usage
```bash
# Start services
./scripts/run_arm64_native_model_services.sh

# Start dashboard
./scripts/run_dashboard_stack.sh --arm64-host-native up
```
```

#### Task 5.2: Troubleshooting Guide
**File**: `docs/NATIVE_TROUBLESHOOTING.md`
- Common installation issues
- Database download problems
- GPU detection failures
- Environment conflicts

## Implementation Timeline

### Week 1: Foundation
- [ ] Task 1.1: AlphaFold2 zero-touch installer
- [ ] Task 1.2: RFDiffusion zero-touch installer  
- [ ] Task 2.1: Database tiering implementation

### Week 2: Integration
- [ ] Task 1.3: Unified installer script
- [ ] Task 3.1: Native service implementation (Option A)
- [ ] Task 3.2: Auto-detection and configuration

### Week 3: Testing & Polish
- [ ] Task 4.1: End-to-end validation
- [ ] Task 2.2: Network database support
- [ ] Task 5.1: Quick start guide

### Week 4: Production Ready
- [ ] Task 4.2: CI/CD integration
- [ ] Task 5.2: Troubleshooting guide
- [ ] Performance benchmarking
- [ ] Final documentation review

## Technical Decisions

### Database Strategy (CRITICAL)
**Recommendation**: Tiered approach with clear communication
- Default to "reduced" databases (50GB) for most users
- Provide "minimal" for CI/CD and testing
- Document "full" for production research

**Rationale**: 2.2TB is prohibitive for most users. Reduced databases provide 70-80% accuracy with 2% of storage.

### Installation Method
**Recommendation**: Conda-based installation
- Platform-agnostic (works on ARM64 and x86_64)
- Isolated environments prevent conflicts
- Easy to uninstall/update

### Service Architecture
**Recommendation**: Keep subprocess-based approach initially
- Simpler to implement and maintain
- Works with any installation method
- Easier to debug
- Can migrate to direct API later for performance

### ARM64 Strategy
**Recommendation**: Native installation preferred over Docker emulation
- Better performance (5-10x faster)
- No emulation overhead
- Direct hardware access

## Success Criteria

1. **Zero Manual Intervention**
   - ✅ User runs single command
   - ✅ Script detects platform/hardware
   - ✅ Installs all dependencies
   - ✅ Downloads model weights
   - ✅ Configures services
   - ✅ Validates installation

2. **Works Out of Box**
   - ✅ Services start successfully
   - ✅ Dashboard connects to services
   - ✅ Test job completes end-to-end
   - ✅ Results are scientifically valid

3. **Platform Support**
   - ✅ x86_64 Linux (CUDA)
   - ✅ ARM64 Linux (CPU/CUDA)
   - ✅ ARM64 macOS (Metal/CPU)

4. **Performance**
   - ✅ Native faster than NIM containers
   - ✅ Reduced databases acceptable accuracy
   - ✅ Multi-GPU utilization

## Risk Mitigation

### Risk: AlphaFold2 Build Failures on ARM64
**Mitigation**: 
- Provide pre-built Docker images as fallback
- Document common build issues and fixes
- Support CPU-only mode

### Risk: Database Download Timeouts
**Mitigation**:
- Resumable downloads (wget -c)
- Provide torrent/alternative mirrors
- Support mounting external storage

### Risk: Version Conflicts
**Mitigation**:
- Use isolated conda environments
- Pin specific package versions
- Provide lockfiles

### Risk: GPU Compatibility Issues
**Mitigation**:
- Auto-detect GPU capabilities
- Fallback to CPU mode
- Support multiple CUDA versions

## Next Steps

1. **Immediate**: Review this plan with stakeholders
2. **Week 1 Start**: Begin Task 1.1 (AlphaFold2 installer)
3. **Parallel**: Start Task 2.1 (Database tiering)
4. **Testing**: Set up test environments (ARM64 + x86_64)

## Questions to Resolve

1. **Database Tier Default**: Should we default to "minimal" or "reduced"?
   - Recommendation: "reduced" for better out-of-box experience

2. **GPU Requirement**: Should GPU be required or optional?
   - Recommendation: Optional, with CPU fallback

3. **Network Databases**: Should we prioritize shared database mounting?
   - Recommendation: Yes, for enterprise/lab deployments

4. **Docker vs Native**: Should Docker remain an option?
   - Recommendation: Yes, keep both paths (hybrid approach)

## References

- **AlphaFold2**: https://github.com/deepmind/alphafold
- **RFDiffusion**: https://github.com/RosettaCommons/RFdiffusion
- **Current Docs**: `docs/ARM64_NATIVE_INSTALLATION.md`, `docs/DGX_SPARK_NATIVE_DEPLOYMENT.md`
- **Model Downloads**: `scripts/init_models.sh`, `scripts/download_models_arm64.sh`
