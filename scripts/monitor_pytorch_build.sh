#!/bin/bash
# PyTorch ARM64 Build Monitor Dashboard
# Created: October 18, 2025
# 
# This script monitors the progress of ARM64 PyTorch builds and provides
# a real-time dashboard with build status, disk usage, and progress tracking.

clear
echo "============================================================"
echo "          PyTorch ARM64 Build Monitor Dashboard"
echo "============================================================"
echo "Build started: $(date)"
echo "Process ID: $(ps aux | grep build_pytorch_arm64_blackwell.sh | grep -v grep | awk '{print $2}' | head -1)"
echo "Log file: ~/pytorch_gb10_blackwell_build.log"
echo ""

while true; do
    clear
    echo "============================================================"
    echo "          PyTorch ARM64 Build Monitor Dashboard"
    echo "============================================================"
    echo "Last updated: $(date)"
    echo ""
    
    # Process status
    echo "📊 PROCESS STATUS:"
    if ps aux | grep build_pytorch_arm64_blackwell.sh | grep -v grep > /dev/null; then
        echo "  ✅ Build process: RUNNING"
        PID=$(ps aux | grep build_pytorch_arm64_blackwell.sh | grep -v grep | awk '{print $2}' | head -1)
        echo "  🔢 Process ID: $PID"
    else
        echo "  ⚠️  Build process: NOT FOUND"
    fi
    
    # Disk usage
    echo ""
    echo "💾 DISK USAGE:"
    if [ -d ~/pytorch_build_gb10_fixed ]; then
        SIZE=$(du -sh ~/pytorch_build_gb10_fixed | cut -f1)
        echo "  📁 Build directory: $SIZE"
    else
        echo "  📁 Build directory: Not created"
    fi
    
    # Build progress
    echo ""
    echo "🏗️  BUILD PROGRESS:"
    if [ -f ~/pytorch_gb10_blackwell_build.log ]; then
        echo "  📄 Last 3 lines of build log:"
        tail -3 ~/pytorch_gb10_blackwell_build.log | sed 's/^/     /'
    else
        echo "  📄 Build log: Not available"
    fi
    
    # Current phase detection
    echo ""
    echo "🔍 CURRENT PHASE:"
    if [ -f ~/pytorch_gb10_blackwell_build.log ]; then
        if tail -50 ~/pytorch_gb10_blackwell_build.log | grep -q "Cloning into"; then
            echo "  📥 Repository cloning"
        elif tail -50 ~/pytorch_gb10_blackwell_build.log | grep -q "cmake"; then
            echo "  ⚙️  CMake configuration"
        elif tail -50 ~/pytorch_gb10_blackwell_build.log | grep -q "Building wheel"; then
            echo "  🛠️  Building PyTorch wheel"
        elif tail -50 ~/pytorch_gb10_blackwell_build.log | grep -q "Installing"; then
            echo "  📦 Installing PyTorch"
        elif tail -50 ~/pytorch_gb10_blackwell_build.log | grep -q "Verifying"; then
            echo "  ✅ Verifying installation"
        elif tail -50 ~/pytorch_gb10_blackwell_build.log | grep -q "Build completed"; then
            echo "  🎉 Build completed successfully!"
            break
        else
            echo "  🔄 In progress..."
        fi
    else
        echo "  ❓ Status unknown"
    fi
    
    echo ""
    echo "============================================================"
    echo "Press Ctrl+C to exit monitoring"
    echo "Commands:"
    echo "  tail -f ~/pytorch_gb10_blackwell_build.log  # Follow build log"
    echo "  ./scripts/build_pytorch_arm64_blackwell.sh  # Start new build"
    echo "============================================================"
    
    sleep 10
done