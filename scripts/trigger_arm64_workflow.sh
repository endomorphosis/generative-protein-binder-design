#!/bin/bash
# Helper script to trigger the ARM64 completion workflow
# This script provides an easy way to start the ARM64 porting completion process

set -e

echo "================================================"
echo "  ARM64 Workflow Trigger Helper"
echo "================================================"
echo ""

# Check if gh CLI is available
if ! command -v gh >/dev/null 2>&1; then
    echo "Error: GitHub CLI (gh) is not installed"
    echo ""
    echo "To install GitHub CLI:"
    echo "  Ubuntu/Debian: sudo apt install gh"
    echo "  macOS: brew install gh"
    echo "  Or visit: https://cli.github.com/"
    echo ""
    echo "After installing, authenticate with: gh auth login"
    echo ""
    exit 1
fi

# Check if authenticated
if ! gh auth status >/dev/null 2>&1; then
    echo "Error: Not authenticated with GitHub CLI"
    echo ""
    echo "Please authenticate with: gh auth login"
    echo ""
    exit 1
fi

echo "Available ARM64 Workflows:"
echo ""
echo "1. ARM64 Validation Workflow (arm64-validation.yml)"
echo "   → ARM64 compatibility check and validation"
echo "   → Platform detection, Docker testing, Python validation"
echo "   → Estimated time: 10-15 minutes"
echo ""
echo "2. Jupyter Notebook Test (jupyter-test.yml)"
echo "   → Test scientific computing environment on ARM64"
echo "   → Python packages, GPU access, notebook execution"
echo "   → Estimated time: 5-10 minutes"
echo ""
echo "3. Runner Connection Test (runner-test.yml)"
echo "   → Quick ARM64 runner connectivity test"
echo "   → System info, GPU check, environment validation"
echo "   → Estimated time: 2-5 minutes"
echo ""
echo "4. Native Installation Test (native-install-test.yml) [Manual dispatch only]"
echo "   → Test native ARM64 installation of protein tools"
echo "   → AlphaFold2, RFDiffusion, ProteinMPNN testing"
echo "   → Estimated time: 30-60 minutes"
echo ""

read -p "Which workflow would you like to trigger? (1-4, or q to quit): " choice

case $choice in
    1)
        echo ""
        echo "Triggering ARM64 Validation Workflow..."
        echo ""
        read -p "Test mode (quick/full/docker-only/native-only, default: full): " test_mode
        test_mode=${test_mode:-full}
        
        echo ""
        echo "Configuration:"
        echo "  - Test mode: $test_mode"
        echo ""
        
        read -p "Proceed with triggering workflow? (y/n): " confirm
        if [ "$confirm" = "y" ]; then
            gh workflow run arm64-validation.yml -f test_mode=$test_mode
            
            echo ""
            echo "✓ Workflow triggered successfully!"
            echo ""
            echo "Monitor progress with:"
            echo "  gh run watch"
            echo "  gh run list --workflow=arm64-validation.yml --limit 5"
            echo ""
        else
            echo "Cancelled."
        fi
        ;;
        
    2)
        echo ""
        echo "Triggering Jupyter Notebook Test..."
        echo ""
        read -p "Notebook path (default: src/protein-binder-design.ipynb): " notebook_path
        notebook_path=${notebook_path:-src/protein-binder-design.ipynb}
        
        echo ""
        echo "Configuration:"
        echo "  - Notebook: $notebook_path"
        echo ""
        
        read -p "Proceed with triggering workflow? (y/n): " confirm
        if [ "$confirm" = "y" ]; then
            gh workflow run "Jupyter Notebook Test" -f notebook_path="$notebook_path"
            
            echo ""
            echo "✓ Workflow triggered successfully!"
            echo ""
            echo "Monitor progress with:"
            echo "  gh run watch"
            echo "  gh run list --workflow='Jupyter Notebook Test' --limit 5"
            echo ""
        else
            echo "Cancelled."
        fi
        ;;
        
    3)
        echo ""
        echo "Triggering Runner Connection Test..."
        echo ""
        echo "This will test basic ARM64 runner connectivity and system resources."
        echo ""
        
        read -p "Proceed with triggering workflow? (y/n): " confirm
        if [ "$confirm" = "y" ]; then
            gh workflow run "Runner Connection Test"
            
            echo ""
            echo "✓ Workflow triggered successfully!"
            echo ""
            echo "Monitor progress with:"
            echo "  gh run watch"
            echo "  gh run list --workflow='Runner Connection Test' --limit 5"
            echo ""
        else
            echo "Cancelled."
        fi
        ;;
        
    4)
        echo ""
        echo "Note: Native Installation Test requires manual dispatch from GitHub web interface"
        echo "This workflow tests installation of protein design tools on ARM64."
        echo ""
        echo "To run this workflow:"
        echo "1. Visit: https://github.com/hallucinate-llc/generative-protein-binder-design/actions"
        echo "2. Find 'Native Installation Test' workflow"
        echo "3. Click 'Run workflow'"
        echo "4. Select component: alphafold2, rfdiffusion, proteinmpnn, or all"
        echo ""
        echo "This workflow is not available via CLI because it's on a feature branch."
        echo ""
        ;;
        
    q|Q)
        echo "Exiting..."
        exit 0
        ;;
        
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac

echo "================================================"
echo "  Next Steps"
echo "================================================"
echo ""
echo "1. Monitor the workflow execution:"
echo "   → gh run watch"
echo "   → https://github.com/hallucinate-llc/generative-protein-binder-design/actions"
echo ""
echo "2. Download results when complete:"
echo "   → gh run list --workflow=<workflow-name> --limit 5"
echo "   → gh run download <run-id>"
echo ""
echo "3. For more help:"
echo "   → ./continue_arm64_port.sh"
echo "   → Review ARM64_COMPLETION_CHECKLIST.md"
echo ""
