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
echo "1. ARM64 Complete Porting Workflow (arm64-complete-port.yml)"
echo "   → Full validation and testing on ARM64"
echo "   → Platform detection, validation, Python testing, Docker testing"
echo "   → Estimated time: 15-40 minutes"
echo ""
echo "2. ARM64 Validation Workflow (arm64-validation.yml)"
echo "   → Quick ARM64 compatibility check"
echo "   → Platform detection and basic validation"
echo "   → Estimated time: 5-10 minutes"
echo ""
echo "3. Protein Design Pipeline (protein-design-pipeline.yml)"
echo "   → Full protein design workflow"
echo "   → Can be run on ARM64 with use_native=true"
echo "   → Estimated time: Variable based on workload"
echo ""

read -p "Which workflow would you like to trigger? (1-3, or q to quit): " choice

case $choice in
    1)
        echo ""
        echo "Triggering ARM64 Complete Porting Workflow..."
        echo ""
        read -p "Run full pipeline tests? (y/n, default: y): " run_pipeline
        run_pipeline=${run_pipeline:-y}
        
        read -p "Run all validation tests? (y/n, default: y): " run_validation
        run_validation=${run_validation:-y}
        
        if [ "$run_pipeline" = "y" ]; then
            run_pipeline_flag="true"
        else
            run_pipeline_flag="false"
        fi
        
        if [ "$run_validation" = "y" ]; then
            run_validation_flag="true"
        else
            run_validation_flag="false"
        fi
        
        echo ""
        echo "Configuration:"
        echo "  - Run full pipeline: $run_pipeline_flag"
        echo "  - Run validation tests: $run_validation_flag"
        echo ""
        
        read -p "Proceed with triggering workflow? (y/n): " confirm
        if [ "$confirm" = "y" ]; then
            gh workflow run arm64-complete-port.yml \
                -f run_full_pipeline=$run_pipeline_flag \
                -f run_validation_tests=$run_validation_flag
            
            echo ""
            echo "✓ Workflow triggered successfully!"
            echo ""
            echo "Monitor progress with:"
            echo "  gh run watch"
            echo "  gh run list --workflow=arm64-complete-port.yml --limit 5"
            echo ""
            echo "Or visit: https://github.com/hallucinate-llc/generative-protein-binder-design/actions"
            echo ""
        else
            echo "Cancelled."
        fi
        ;;
        
    2)
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
        
    3)
        echo ""
        echo "Triggering Protein Design Pipeline..."
        echo ""
        read -p "Use native ARM64 execution? (y/n, default: y): " use_native
        use_native=${use_native:-y}
        
        read -p "Target protein PDB ID (default: 7BZ5): " target_protein
        target_protein=${target_protein:-7BZ5}
        
        read -p "Number of designs (default: 10): " num_designs
        num_designs=${num_designs:-10}
        
        echo ""
        echo "Configuration:"
        echo "  - Use native: $use_native"
        echo "  - Target protein: $target_protein"
        echo "  - Number of designs: $num_designs"
        echo ""
        
        read -p "Proceed with triggering workflow? (y/n): " confirm
        if [ "$confirm" = "y" ]; then
            gh workflow run protein-design-pipeline.yml \
                -f use_native=$use_native \
                -f target_protein=$target_protein \
                -f num_designs=$num_designs
            
            echo ""
            echo "✓ Workflow triggered successfully!"
            echo ""
            echo "Monitor progress with:"
            echo "  gh run watch"
            echo "  gh run list --workflow=protein-design-pipeline.yml --limit 5"
            echo ""
        else
            echo "Cancelled."
        fi
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
