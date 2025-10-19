#!/bin/bash

# Test script for MCP Server and Dashboard
# This script validates the basic functionality of the implemented components

set -e

echo "========================================"
echo "MCP Server and Dashboard Test Suite"
echo "========================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Check if MCP server is running
echo "Test 1: MCP Server Health Check"
if curl -s http://localhost:8000/health | grep -q "healthy"; then
    echo -e "${GREEN}✓ MCP Server is healthy${NC}"
else
    echo -e "${RED}✗ MCP Server health check failed${NC}"
    exit 1
fi
echo ""

# Test 2: Check MCP protocol endpoints
echo "Test 2: MCP Protocol Tools Endpoint"
if curl -s http://localhost:8000/mcp/v1/tools | grep -q "design_protein_binder"; then
    echo -e "${GREEN}✓ MCP tools endpoint is working${NC}"
else
    echo -e "${RED}✗ MCP tools endpoint failed${NC}"
    exit 1
fi
echo ""

# Test 3: Create a test job
echo "Test 3: Create a protein design job"
JOB_RESPONSE=$(curl -s -X POST http://localhost:8000/api/jobs \
  -H "Content-Type: application/json" \
  -d '{"sequence": "MKFLKFSLLTAVLLSVVFAFSSCG", "job_name": "Test Job", "num_designs": 2}')

JOB_ID=$(echo $JOB_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['job_id'])")

if [ ! -z "$JOB_ID" ]; then
    echo -e "${GREEN}✓ Job created successfully: $JOB_ID${NC}"
else
    echo -e "${RED}✗ Job creation failed${NC}"
    exit 1
fi
echo ""

# Test 4: Check job status
echo "Test 4: Check job status"
sleep 2
JOB_STATUS=$(curl -s http://localhost:8000/api/jobs/$JOB_ID | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])")

if [ "$JOB_STATUS" == "completed" ] || [ "$JOB_STATUS" == "running" ]; then
    echo -e "${GREEN}✓ Job status retrieved: $JOB_STATUS${NC}"
else
    echo -e "${YELLOW}⚠ Job status: $JOB_STATUS${NC}"
fi
echo ""

# Test 5: List all jobs
echo "Test 5: List all jobs"
JOB_COUNT=$(curl -s http://localhost:8000/api/jobs | python3 -c "import sys, json; print(len(json.load(sys.stdin)))")

if [ "$JOB_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓ Found $JOB_COUNT job(s)${NC}"
else
    echo -e "${RED}✗ No jobs found${NC}"
    exit 1
fi
echo ""

# Test 6: Check service status
echo "Test 6: Check NIM services status"
if curl -s http://localhost:8000/api/services/status | grep -q "alphafold"; then
    echo -e "${GREEN}✓ Service status endpoint is working${NC}"
    echo -e "${YELLOW}Note: NIM services may show errors if not running${NC}"
else
    echo -e "${RED}✗ Service status endpoint failed${NC}"
    exit 1
fi
echo ""

# Summary
echo "========================================"
echo -e "${GREEN}All tests passed!${NC}"
echo "========================================"
echo ""
echo "You can now:"
echo "  - Access the MCP Dashboard at http://localhost:3000"
echo "  - Access the MCP Server API docs at http://localhost:8000/docs"
echo "  - Access Jupyter Notebook at http://localhost:8888"
echo ""
