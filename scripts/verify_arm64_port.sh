#!/bin/bash
# Verification script for ARM64 porting
# This script verifies that all ARM64 porting changes are in place and valid

set -e

echo "================================================"
echo "  ARM64 Porting Verification"
echo "================================================"
echo

ERRORS=0

# Function to check if file exists
check_file() {
    if [ -f "$1" ]; then
        echo "[OK] $1 exists"
    else
        echo "[FAIL] $1 missing"
        ERRORS=$((ERRORS + 1))
    fi
}

# Function to check if file is executable
check_executable() {
    if [ -x "$1" ]; then
        echo "[OK] $1 is executable"
    else
        echo "[FAIL] $1 not executable"
        ERRORS=$((ERRORS + 1))
    fi
}

# Function to validate YAML
check_yaml() {
    if python3 -c "import yaml; yaml.safe_load(open('$1'))" 2>/dev/null; then
        echo "[OK] $1 is valid YAML"
    else
        echo "[FAIL] $1 has YAML errors"
        ERRORS=$((ERRORS + 1))
    fi
}

# Function to validate Docker Compose
check_docker_compose() {
    if NGC_CLI_API_KEY=dummy docker compose -f "$1" config > /dev/null 2>&1; then
        echo "[OK] $1 is valid Docker Compose"
    else
        echo "[FAIL] $1 has Docker Compose errors"
        ERRORS=$((ERRORS + 1))
    fi
}

echo "=== Checking New Files ==="
check_file ".github/workflows/arm64-validation.yml"
check_file "ARM64_DEPLOYMENT.md"
check_file "ARM64_PORTING_SUMMARY.md"
check_file "detect_platform.sh"
check_file ".gitignore"
echo

echo "=== Checking File Permissions ==="
check_executable "detect_platform.sh"
echo

echo "=== Validating Workflows ==="
check_yaml ".github/workflows/arm64-validation.yml"
check_yaml ".github/workflows/runner-test.yml"
check_yaml ".github/workflows/system-health.yml"
check_yaml ".github/workflows/docker-test.yml"
check_yaml ".github/workflows/native-install-test.yml"
check_yaml ".github/workflows/jupyter-test.yml"
check_yaml ".github/workflows/protein-design-pipeline.yml"
echo

echo "=== Validating Docker Compose Files ==="
check_docker_compose "deploy/docker-compose.yaml"
check_docker_compose "deploy/docker-compose-single-gpu.yaml"
echo

echo "=== Checking Platform Annotations ==="
if grep -q "platform: linux/amd64.*NVIDIA NIM" deploy/docker-compose.yaml; then
    echo "[OK] docker-compose.yaml has platform annotations with comments"
else
    echo "[FAIL] docker-compose.yaml missing proper platform annotations"
    ERRORS=$((ERRORS + 1))
fi

if grep -q "platform: linux/amd64.*NVIDIA NIM" deploy/docker-compose-single-gpu.yaml; then
    echo "[OK] docker-compose-single-gpu.yaml has platform annotations with comments"
else
    echo "[FAIL] docker-compose-single-gpu.yaml missing proper platform annotations"
    ERRORS=$((ERRORS + 1))
fi
echo

echo "=== Checking Documentation Links ==="
if grep -q "ARM64_DEPLOYMENT.md" README.md; then
    echo "[OK] README.md links to ARM64_DEPLOYMENT.md"
else
    echo "[FAIL] README.md missing ARM64_DEPLOYMENT.md link"
    ERRORS=$((ERRORS + 1))
fi

if grep -q "detect_platform.sh" README.md; then
    echo "[OK] README.md mentions detect_platform.sh"
else
    echo "[FAIL] README.md missing detect_platform.sh reference"
    ERRORS=$((ERRORS + 1))
fi

if grep -q "arm64-validation.yml" .github/workflows/README.md; then
    echo "[OK] Workflow README documents arm64-validation.yml"
else
    echo "[FAIL] Workflow README missing arm64-validation.yml"
    ERRORS=$((ERRORS + 1))
fi
echo

echo "=== Testing Platform Detection Script ==="
if ./detect_platform.sh > /tmp/platform_test.log 2>&1; then
    echo "[OK] detect_platform.sh runs successfully"
    # Check if it detected architecture
    if grep -q "System Architecture:" /tmp/platform_test.log; then
        echo "[OK] Platform detection works"
    else
        echo "[FAIL] Platform detection incomplete"
        ERRORS=$((ERRORS + 1))
    fi
else
    echo "[FAIL] detect_platform.sh failed to run"
    ERRORS=$((ERRORS + 1))
fi
echo

echo "=== Checking Workflow Structure ==="
# Check that arm64-validation workflow has all required jobs
if python3 -c "
import yaml
with open('.github/workflows/arm64-validation.yml') as f:
    data = yaml.safe_load(f)
    jobs = data.get('jobs', {})
    if 'validate-arm64' in jobs:
        print('[OK] arm64-validation.yml has validate-arm64 job')
    else:
        print('[FAIL] arm64-validation.yml missing validate-arm64 job')
        exit(1)
" 2>/dev/null; then
    :
else
    ERRORS=$((ERRORS + 1))
fi

# Check for platform detection step
if grep -q "Platform Detection" .github/workflows/arm64-validation.yml; then
    echo "[OK] arm64-validation.yml has Platform Detection step"
else
    echo "[FAIL] arm64-validation.yml missing Platform Detection step"
    ERRORS=$((ERRORS + 1))
fi
echo

echo "=== Summary ==="
echo "Total checks performed: Multiple"
echo "Errors found: $ERRORS"
echo

if [ $ERRORS -eq 0 ]; then
    echo "================================================"
    echo "  ✓ ALL CHECKS PASSED"
    echo "================================================"
    echo
    echo "The ARM64 porting is complete and verified!"
    echo
    echo "Next steps:"
    echo "1. Test on actual ARM64 hardware: gh workflow run arm64-validation.yml"
    echo "2. Run full pipeline: gh workflow run protein-design-pipeline.yml -f use_native=true"
    echo "3. Check system health: gh workflow run system-health.yml"
    echo
    exit 0
else
    echo "================================================"
    echo "  ✗ VERIFICATION FAILED"
    echo "================================================"
    echo
    echo "Please fix the errors above and run this script again."
    echo
    exit 1
fi
