#!/bin/bash
# Simple ARM64 workflow trigger via git push
# This works around GitHub API branch restrictions

set -e

echo "================================================"
echo "  ARM64 Workflow Trigger (Git Push Method)"
echo "================================================"
echo ""

echo "This method works by creating a small commit to trigger workflows"
echo "that activate on push events. This bypasses GitHub API branch restrictions."
echo ""

echo "Available trigger methods:"
echo ""
echo "1. Runner Connection Test (triggers on any push)"
echo "   → Tests ARM64 runner connectivity and system resources"
echo "   → Quick validation of GPU, Python, Docker access"
echo ""
echo "2. Jupyter Notebook Test (triggers on notebook changes)"
echo "   → Full scientific computing environment test"
echo "   → May trigger if we touch notebook files"
echo ""

read -p "Proceed with triggering Runner Connection Test? (y/n): " confirm

if [ "$confirm" != "y" ]; then
    echo "Cancelled."
    exit 0
fi

# Create a simple timestamp file to trigger the workflow
TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S %Z")
TRIGGER_FILE="workflow_trigger_$(date +%s).txt"

echo "ARM64 Workflow Trigger" > "$TRIGGER_FILE"
echo "Timestamp: $TIMESTAMP" >> "$TRIGGER_FILE"
echo "Method: Git push trigger" >> "$TRIGGER_FILE"
echo "Branch: $(git branch --show-current)" >> "$TRIGGER_FILE"
echo "Commit: $(git rev-parse --short HEAD)" >> "$TRIGGER_FILE"
echo "" >> "$TRIGGER_FILE"
echo "This file was created to trigger the Runner Connection Test workflow" >> "$TRIGGER_FILE"
echo "which validates ARM64 system resources and runner connectivity." >> "$TRIGGER_FILE"

echo ""
echo "Creating commit to trigger workflow..."

git add "$TRIGGER_FILE"
git commit -m "Trigger ARM64 Runner Connection Test

- Test ARM64 runner connectivity via push trigger
- Validate system resources (CPU, Memory, GPU, Storage)
- Verify Python environment and Docker access
- Generated at: $TIMESTAMP"

echo ""
echo "Pushing to trigger workflow..."
git push origin HEAD

echo ""
echo "✓ Commit pushed successfully!"
echo ""
echo "The Runner Connection Test workflow should now be triggered."
echo ""
echo "Monitor progress:"
echo "  → Check runner logs: tail -f /home/barberb/actions-runner/_diag/Runner_*.log"
echo "  → Check work directory: ls -la /home/barberb/actions-runner/_work/"
echo "  → GitHub Actions: https://github.com/hallucinate-llc/generative-protein-binder-design/actions"
echo ""
echo "The workflow will:"
echo "  1. Test ARM64 architecture detection"
echo "  2. Check system resources (CPU, RAM, Storage)"
echo "  3. Validate GPU access"
echo "  4. Test Python environment"
echo "  5. Check Docker connectivity"
echo ""

# Clean up trigger file after a delay
sleep 2
rm -f "$TRIGGER_FILE"
echo "✓ Trigger file cleaned up"
echo ""
echo "Workflow should be running on your ARM64 runner now! 🚀"