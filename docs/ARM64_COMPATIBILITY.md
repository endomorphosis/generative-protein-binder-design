# ARM64 System Compatibility Notice

⚠️ **Important:** Your system is running on ARM64 architecture (aarch64), but the NVIDIA NIM containers are currently built for AMD64/x86_64 architecture. This creates a platform compatibility issue.

## Current System Status

**Detected Architecture:** ARM64 (aarch64)  
**Container Architecture:** AMD64 (x86_64)  
**GPU Available:** 1x NVIDIA GB10  

## Known Issues

1. **Platform Mismatch Warning:** You'll see warnings about platform mismatches when running containers
2. **Performance Impact:** Running AMD64 containers on ARM64 may cause performance degradation
3. **Potential Compatibility Issues:** Some containers may fail to start or run properly

## Recommended Solutions

### Option 1: Use Docker Platform Emulation (Current Approach)
The Docker Compose files have been modified to explicitly specify `platform: linux/amd64` which will use emulation:

```bash
# Start all services (may have compatibility issues)
docker compose up -d

# OR use the single GPU manager for sequential operation
./run_single_gpu.sh
```

**Pros:** May work with emulation  
**Cons:** Performance impact, potential compatibility issues

### Option 2: Native Installation (Complex but Possible)
For those with significant technical expertise, you can install the tools directly on your ARM64 system. This approach avoids Docker platform compatibility issues but requires substantial time and effort.

**Installation Complexity:** High  
**Time Required:** 2-7 days depending on experience  
**Technical Expertise:** Advanced

See the comprehensive guide: **[ARM64 Native Installation Guide](ARM64_NATIVE_INSTALLATION.md)**

This guide includes:
- Building AlphaFold2 from source for ARM64
- Installing RFDiffusion natively with ARM64-compatible dependencies
- Setting up ProteinMPNN manually
- Creating wrapper scripts to integrate with the blueprint workflow
- Troubleshooting common ARM64-specific issues

**Pros:**
- No Docker platform mismatch issues
- Better performance than emulation
- Full control over the environment
- Can be optimized for your specific ARM64 system

**Cons:**
- Very time-consuming setup (days, not hours)
- Requires advanced technical skills
- Complex dependency management
- May require building multiple packages from source
- Debugging can be challenging
- Limited community support for ARM64 builds

### Option 3: Cloud-Based Alternative
For optimal performance, consider using cloud instances with AMD64 architecture and multiple GPUs:

- NVIDIA NGC Cloud instances
- AWS EC2 instances with NVIDIA GPUs (g4dn, g5, p3, p4 families)
- Google Cloud Platform with NVIDIA GPUs
- Azure NC-series instances

### Option 4: Alternative Approaches
If Docker emulation, native installation, and cloud options don't meet your needs:

1. **ARM64 Compatible Alternatives:** Look for ARM64-native protein folding tools
2. **VM with x86_64 Emulation:** Use QEMU to run a full x86_64 virtual machine
3. **Remote Development:** Use cloud-based development environments with x86_64 architecture

## Troubleshooting Tips

### If Containers Fail to Start:
1. Check Docker logs: `docker compose logs [service_name]`
2. Verify GPU access: `docker run --rm --gpus all --platform linux/amd64 nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi`
3. Try running services one at a time: `./run_single_gpu.sh`

### If Performance is Poor:
1. Monitor GPU usage: `watch nvidia-smi`
2. Check system resources: `htop`
3. Consider reducing model complexity or using smaller datasets

## Single GPU Optimizations

Since you have only 1 GPU instead of the recommended 4, the setup has been modified to:

1. **Share GPU 0:** All services now use the same GPU (`device_ids: ['0']`)
2. **Sequential Execution:** Use `run_single_gpu.sh` to run one service at a time
3. **Reduced Resource Conflicts:** Services can be started individually to avoid GPU memory conflicts

## Next Steps

1. **Try the current setup:** `docker compose up -d` or `./run_single_gpu.sh`
2. **Monitor for issues:** Check logs and performance
3. **If problems persist:** Consider cloud alternatives or native installation

For the best experience with this workflow, we recommend using an AMD64 system with multiple NVIDIA GPUs as specified in the original requirements.