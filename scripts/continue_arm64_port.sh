#!/bin/bash
# Script to continue ARM64 porting process
# This script helps execute the remaining steps from ARM64_AUTOMATION_SUMMARY.md

set -e

echo "================================================"
echo "  Continue ARM64 Porting Process"
echo "================================================"
echo ""

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored status
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "[i] $1"
}

# Check if we're on ARM64
ARCH=$(uname -m)
echo "Current Architecture: $ARCH"
echo ""

if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    print_status "Running on ARM64 - can execute ARM64 workflows"
    IS_ARM64=true
else
    print_warning "Running on $ARCH - ARM64 workflows require ARM64 runner"
    IS_ARM64=false
fi
echo ""

echo "================================================"
echo "  ARM64 Porting Status Check"
echo "================================================"
echo ""

# Check if verification passes
print_info "Running verification script..."
if ./verify_arm64_port.sh > /tmp/verify_result.txt 2>&1; then
    print_status "All ARM64 porting verifications passed"
else
    print_error "Verification failed - see /tmp/verify_result.txt for details"
    exit 1
fi
echo ""

echo "================================================"
echo "  Next Steps to Complete ARM64 Porting"
echo "================================================"
echo ""

echo "Based on ARM64_AUTOMATION_SUMMARY.md, the following steps remain:"
echo ""

echo "1. ${YELLOW}Monitor Workflow Execution${NC}"
echo "   The arm64-complete-port.yml workflow should be running on an ARM64 runner."
echo ""
echo "   To check status:"
if command -v gh >/dev/null 2>&1; then
    echo "   → gh run list --workflow=arm64-complete-port.yml --limit 5"
    echo "   → gh run watch  # Watch the latest run"
    echo ""
    print_info "Running workflow status check..."
    gh run list --workflow=arm64-complete-port.yml --limit 5 2>/dev/null || print_warning "No recent workflow runs found or gh CLI not configured"
else
    echo "   → Visit: https://github.com/hallucinate-llc/generative-protein-binder-design/actions"
    print_warning "GitHub CLI (gh) not installed - use web interface"
fi
echo ""

echo "2. ${YELLOW}Download and Validate Results${NC}"
echo "   Once the workflow completes, download the completion report:"
echo ""
echo "   → gh run list --workflow=arm64-complete-port.yml"
echo "   → gh run download <run-id> --name arm64-completion-report-<run_number>"
echo "   → cat arm64_completion_report.txt"
echo ""

echo "3. ${YELLOW}Run Full Pipeline Testing${NC}"
echo "   Test the complete protein design pipeline with real targets:"
echo ""
if [ "$IS_ARM64" = true ]; then
    echo "   → gh workflow run protein-design-pipeline.yml -f use_native=true -f target_protein=7BZ5 -f num_designs=10"
    print_status "You can run this on your ARM64 system"
else
    echo "   → This requires an ARM64 runner or system"
    print_warning "Trigger via GitHub Actions on ARM64 runner"
fi
echo ""

echo "4. ${YELLOW}Production Deployment${NC}"
echo "   After successful testing, deploy to production:"
echo ""
echo "   → Review ARM64_DEPLOYMENT.md for deployment options"
echo "   → Set up automated scheduling for regular testing"
echo "   → Monitor performance and optimize resource allocation"
echo ""

echo "================================================"
echo "  Quick Actions"
echo "================================================"
echo ""

# Provide quick action menu
echo "What would you like to do?"
echo ""
echo "  a) Run platform detection"
echo "  b) Run verification script"
echo "  c) Check workflow status (requires gh CLI)"
echo "  d) List available workflows"
echo "  e) Show ARM64 deployment guide"
echo "  f) Trigger arm64-complete-port workflow"
echo "  g) Exit and show summary"
echo ""

read -p "Enter your choice (a-g): " choice

case $choice in
    a)
        echo ""
        print_info "Running platform detection..."
        ./detect_platform.sh
        ;;
    b)
        echo ""
        print_info "Running verification script..."
        ./verify_arm64_port.sh
        ;;
    c)
        if command -v gh >/dev/null 2>&1; then
            echo ""
            print_info "Checking workflow status..."
            gh run list --workflow=arm64-complete-port.yml --limit 10
        else
            print_error "GitHub CLI (gh) not installed"
            echo "Install with: sudo apt install gh"
        fi
        ;;
    d)
        echo ""
        print_info "Available workflows:"
        ls -1 .github/workflows/*.yml | sed 's|.github/workflows/||'
        ;;
    e)
        echo ""
        print_info "ARM64 Deployment Guide:"
        cat ARM64_DEPLOYMENT.md | head -100
        echo ""
        echo "(See ARM64_DEPLOYMENT.md for full guide)"
        ;;
    f)
        if command -v gh >/dev/null 2>&1; then
            echo ""
            print_info "Triggering arm64-complete-port workflow..."
            if [ "$IS_ARM64" = true ]; then
                gh workflow run arm64-complete-port.yml \
                    -f run_full_pipeline=true \
                    -f run_validation_tests=true
                print_status "Workflow triggered! Check status with: gh run watch"
            else
                print_warning "This will queue for ARM64 runner"
                gh workflow run arm64-complete-port.yml \
                    -f run_full_pipeline=true \
                    -f run_validation_tests=true
                print_status "Workflow triggered! It will run when ARM64 runner is available"
            fi
        else
            print_error "GitHub CLI (gh) not installed"
            echo "Install with: sudo apt install gh"
            echo "Or trigger manually at: https://github.com/hallucinate-llc/generative-protein-binder-design/actions"
        fi
        ;;
    g|*)
        echo ""
        ;;
esac

echo ""
echo "================================================"
echo "  Summary"
echo "================================================"
echo ""

echo "ARM64 Porting Status:"
echo "  ✓ Infrastructure setup complete"
echo "  ✓ Workflows deployed"
echo "  ✓ Documentation complete"
echo "  ✓ Verification scripts passing"
echo ""

echo "Remaining Tasks:"
echo "  → Monitor arm64-complete-port workflow execution"
echo "  → Validate completion report"
echo "  → Run full pipeline testing"
echo "  → Deploy to production"
echo ""

echo "Key Documentation:"
echo "  → ARM64_AUTOMATION_SUMMARY.md - This automation guide"
echo "  → ARM64_NEXT_STEPS.md - Detailed next steps"
echo "  → ARM64_DEPLOYMENT.md - Deployment guide"
echo "  → ARM64_WORKFLOW_STATUS.md - Workflow status documentation"
echo ""

echo "For more information, review:"
echo "  → ARM64_AUTOMATION_SUMMARY.md"
echo "  → https://github.com/hallucinate-llc/generative-protein-binder-design/actions"
echo ""

print_status "ARM64 porting infrastructure is ready!"
echo ""
