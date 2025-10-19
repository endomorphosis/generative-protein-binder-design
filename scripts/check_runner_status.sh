#!/bin/bash

echo "=== GitHub Actions Self-Hosted Runner Status ==="
echo "Generated: $(date)"
echo "Hostname: $(hostname)"
echo "Architecture: $(uname -m)"
echo

echo "=== Runner Process Status ==="
if pgrep -f "Runner.Listener" > /dev/null; then
    echo "✓ GitHub Actions Runner is RUNNING"
    echo "Process details:"
    ps aux | grep "Runner.Listener" | grep -v grep
else
    echo "✗ GitHub Actions Runner is NOT RUNNING"
fi
echo

echo "=== Runner Configuration ==="
if [ -f "/home/barberb/actions-runner/.runner" ]; then
    echo "Runner configuration file found:"
    cat /home/barberb/actions-runner/.runner | jq . 2>/dev/null || cat /home/barberb/actions-runner/.runner
else
    echo "No runner configuration found"
fi
echo

echo "=== System Resources ==="
echo "Memory:"
free -h
echo
echo "Disk Space:"
df -h . | head -2
echo
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv 2>/dev/null || echo "GPU not available"
echo

echo "=== Docker Status ==="
if command -v docker >/dev/null 2>&1; then
    echo "Docker Version: $(docker --version)"
    echo "Docker Status: $(docker info --format '{{.ServerVersion}}' 2>/dev/null || echo 'Docker not accessible')"
else
    echo "Docker not installed"
fi
echo

echo "=== Python Environment ==="
python3 --version
if [ -d ".venv" ]; then
    echo "Virtual environment exists: .venv"
    source .venv/bin/activate 2>/dev/null && echo "Virtual environment activated" || echo "Failed to activate venv"
fi
echo

echo "=== Workflows Available ==="
if [ -d ".github/workflows" ]; then
    echo "Available workflows:"
    ls -la .github/workflows/*.yml
else
    echo "No workflows directory found"
fi
echo

echo "=== Runner Labels Expected ==="
echo "This runner should appear with these labels in GitHub:"
echo "- self-hosted"
echo "- ARM64" 
echo "- Linux"
echo "- gpu"
echo "- nvidia"
echo "- protein-design"
echo

echo "=== Next Steps ==="
echo "1. Check runner status at: https://github.com/hallucinate-llc/generative-protein-binder-design/settings/actions/runners"
echo "2. The runner should be visible and online"
echo "3. Test workflows by pushing commits or using GitHub web interface"
echo "4. Monitor runner logs: tail -f /home/barberb/actions-runner/_diag/Runner_*.log"
echo

echo "✓ Runner status check complete!"