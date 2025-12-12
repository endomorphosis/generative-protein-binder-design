#!/usr/bin/env python3
"""
Comprehensive MCP Tool Validation and Schema Testing
Verifies all MCP server tools work correctly with proper schema validation
"""

import asyncio
import httpx
import json
from typing import Dict, Any, List
from datetime import datetime

MCP_SERVER_URL = "http://localhost:8010"

class MCPToolValidator:
    """Validates MCP tools against their declared schemas"""
    
    def __init__(self):
        self.client = None
        self.tools_schema = {}
        self.validation_results = []
    
    async def init(self):
        """Initialize async client and fetch tool schemas"""
        self.client = httpx.AsyncClient(timeout=30.0)
        await self._fetch_tools_schema()
    
    async def close(self):
        """Close async client"""
        if self.client:
            await self.client.aclose()
    
    async def _fetch_tools_schema(self):
        """Fetch tool schemas from MCP server"""
        response = await self.client.get(f"{MCP_SERVER_URL}/mcp/v1/tools")
        if response.status_code == 200:
            data = response.json()
            for tool in data.get("tools", []):
                self.tools_schema[tool["name"]] = tool
    
    def _validate_schema(self, tool_name: str, inputs: Dict[str, Any]) -> tuple[bool, str]:
        """Validate inputs against tool schema"""
        if tool_name not in self.tools_schema:
            return False, f"Tool {tool_name} not found in schema"
        
        schema = self.tools_schema[tool_name]
        input_schema = schema.get("inputSchema", {})
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])
        
        # Check required fields
        for field in required:
            if field not in inputs:
                return False, f"Missing required field: {field}"
        
        # Check field types (simplified)
        for field, value in inputs.items():
            if field in properties:
                field_schema = properties[field]
                expected_type = field_schema.get("type")
                
                if expected_type:
                    python_type = {
                        "string": str,
                        "integer": int,
                        "number": float,
                        "boolean": bool,
                        "array": list,
                        "object": dict
                    }.get(expected_type)
                    
                    if python_type and not isinstance(value, python_type):
                        return False, f"Field {field} expects {expected_type}, got {type(value).__name__}"
        
        return True, "Schema validation passed"
    
    async def test_design_protein_binder_schema(self):
        """Test design_protein_binder tool schema"""
        test_name = "design_protein_binder - Schema Validation"
        
        # Valid input
        valid_input = {
            "sequence": "MKTAYIAKQRQISFVK",
            "job_name": "test_job",
            "num_designs": 5
        }
        
        is_valid, msg = self._validate_schema("design_protein_binder", valid_input)
        self._log_result(test_name, "PASS" if is_valid else "FAIL", msg)
        
        # Test with minimal required fields
        test_name = "design_protein_binder - Minimal Input"
        minimal_input = {
            "sequence": "MKTAYIAKQRQISFVK"
        }
        
        is_valid, msg = self._validate_schema("design_protein_binder", minimal_input)
        self._log_result(test_name, "PASS" if is_valid else "FAIL", msg)
        
        # Test invalid input (missing required field)
        test_name = "design_protein_binder - Missing Required Field"
        invalid_input = {
            "job_name": "test"
        }
        
        is_valid, msg = self._validate_schema("design_protein_binder", invalid_input)
        self._log_result(test_name, "PASS" if not is_valid else "FAIL", f"Correctly rejected: {msg}")
        
        # Test invalid type
        test_name = "design_protein_binder - Invalid Type"
        invalid_type = {
            "sequence": 12345,  # Should be string
            "num_designs": "five"  # Should be integer
        }
        
        is_valid, msg = self._validate_schema("design_protein_binder", invalid_type)
        result_text = f"Correctly rejected: {msg}"
        self._log_result(test_name, "PASS" if not is_valid else "FAIL", result_text)
    
    async def test_get_job_status_schema(self):
        """Test get_job_status tool schema"""
        test_name = "get_job_status - Schema Validation"
        
        valid_input = {
            "job_id": "job_12345"
        }
        
        is_valid, msg = self._validate_schema("get_job_status", valid_input)
        self._log_result(test_name, "PASS" if is_valid else "FAIL", msg)
        
        # Test missing required field
        test_name = "get_job_status - Missing Required Field"
        invalid_input = {}
        
        is_valid, msg = self._validate_schema("get_job_status", invalid_input)
        self._log_result(test_name, "PASS" if not is_valid else "FAIL", f"Correctly rejected: {msg}")
    
    async def test_list_jobs_schema(self):
        """Test list_jobs tool schema"""
        test_name = "list_jobs - Schema Validation"
        
        # list_jobs requires no parameters
        valid_input = {}
        
        is_valid, msg = self._validate_schema("list_jobs", valid_input)
        self._log_result(test_name, "PASS" if is_valid else "FAIL", msg)
    
    async def test_tool_execution(self):
        """Test actual tool execution"""
        print("\n" + "="*80)
        print("TESTING ACTUAL TOOL EXECUTION")
        print("="*80 + "\n")
        
        # Test 1: Create a job
        test_name = "Execute design_protein_binder"
        try:
            response = await self.client.post(
                f"{MCP_SERVER_URL}/api/jobs",
                json={
                    "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLG",
                    "job_name": "validation_test",
                    "num_designs": 2
                }
            )
            
            if response.status_code == 200:
                job_data = response.json()
                job_id = job_data["job_id"]
                self._log_result(test_name, "PASS", f"Created job {job_id}")
                
                # Test 2: Get job status
                test_name = "Execute get_job_status"
                status_response = await self.client.get(
                    f"{MCP_SERVER_URL}/api/jobs/{job_id}"
                )
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    self._log_result(
                        test_name,
                        "PASS",
                        f"Retrieved status: {status_data['status']}"
                    )
                    
                    # Test 3: List jobs
                    test_name = "Execute list_jobs"
                    list_response = await self.client.get(f"{MCP_SERVER_URL}/api/jobs")
                    
                    if list_response.status_code == 200:
                        jobs = list_response.json()
                        self._log_result(
                            test_name,
                            "PASS",
                            f"Retrieved {len(jobs)} jobs"
                        )
                    else:
                        self._log_result(test_name, "FAIL", f"Status {list_response.status_code}")
                else:
                    self._log_result(test_name, "FAIL", f"Status {status_response.status_code}")
            else:
                self._log_result(test_name, "FAIL", f"Status {response.status_code}")
        except Exception as e:
            self._log_result(test_name, "ERROR", str(e))
    
    async def test_mcp_protocol_compliance(self):
        """Test MCP protocol compliance"""
        print("\n" + "="*80)
        print("TESTING MCP PROTOCOL COMPLIANCE")
        print("="*80 + "\n")
        
        # Test tools endpoint format
        test_name = "MCP /mcp/v1/tools endpoint format"
        try:
            response = await self.client.get(f"{MCP_SERVER_URL}/mcp/v1/tools")
            if response.status_code == 200:
                data = response.json()
                
                # Check for required structure
                if "tools" in data and isinstance(data["tools"], list):
                    tools = data["tools"]
                    
                    # Validate each tool has required fields
                    all_valid = True
                    for tool in tools:
                        required_fields = {"name", "description", "inputSchema"}
                        if not required_fields.issubset(set(tool.keys())):
                            all_valid = False
                            break
                    
                    if all_valid:
                        self._log_result(test_name, "PASS", f"Format correct, {len(tools)} tools found")
                    else:
                        self._log_result(test_name, "FAIL", "Tool missing required fields")
                else:
                    self._log_result(test_name, "FAIL", "Invalid tools endpoint format")
            else:
                self._log_result(test_name, "FAIL", f"Status {response.status_code}")
        except Exception as e:
            self._log_result(test_name, "ERROR", str(e))
        
        # Test resources endpoint format
        test_name = "MCP /mcp/v1/resources endpoint format"
        try:
            response = await self.client.get(f"{MCP_SERVER_URL}/mcp/v1/resources")
            if response.status_code == 200:
                data = response.json()
                
                if "resources" in data and isinstance(data["resources"], list):
                    resources = data["resources"]
                    
                    # Validate each resource has required fields
                    all_valid = True
                    for resource in resources:
                        required_fields = {"uri", "name", "description", "mimeType"}
                        if not required_fields.issubset(set(resource.keys())):
                            all_valid = False
                            break
                    
                    if all_valid or len(resources) == 0:
                        self._log_result(test_name, "PASS", f"Format correct, {len(resources)} resources found")
                    else:
                        self._log_result(test_name, "FAIL", "Resource missing required fields")
                else:
                    self._log_result(test_name, "FAIL", "Invalid resources endpoint format")
            else:
                self._log_result(test_name, "FAIL", f"Status {response.status_code}")
        except Exception as e:
            self._log_result(test_name, "ERROR", str(e))
    
    def _log_result(self, test_name: str, status: str, details: str):
        """Log test result"""
        self.validation_results.append({
            "test": test_name,
            "status": status,
            "details": details
        })
        
        color = {
            "PASS": "\033[92m",
            "FAIL": "\033[91m",
            "ERROR": "\033[93m",
            "SKIP": "\033[96m",
        }.get(status, "\033[0m")
        reset = "\033[0m"
        
        print(f"{color}[{status}]{reset} {test_name}")
        print(f"      {details}\n")
    
    def print_summary(self):
        """Print validation summary"""
        print("\n" + "="*80)
        print("MCP TOOL VALIDATION SUMMARY")
        print("="*80)
        
        status_counts = {}
        for result in self.validation_results:
            status = result["status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print(f"\nTotal tests: {len(self.validation_results)}")
        for status in ["PASS", "FAIL", "ERROR", "SKIP"]:
            count = status_counts.get(status, 0)
            if count > 0:
                print(f"  {status}: {count}")
        
        # Show passing rate
        pass_count = status_counts.get("PASS", 0)
        total_count = len(self.validation_results)
        pass_rate = (pass_count / total_count * 100) if total_count > 0 else 0
        
        print(f"\nPass Rate: {pass_rate:.1f}%")
        
        if pass_rate == 100:
            print("\n✓ All tests passed!")
        
        print("\n" + "="*80)


async def main():
    """Run comprehensive validation"""
    validator = MCPToolValidator()
    await validator.init()
    
    try:
        print("\n" + "="*80)
        print("COMPREHENSIVE MCP TOOL VALIDATION")
        print("="*80)
        print(f"Server: {MCP_SERVER_URL}\n")
        
        # Test schema validation
        print("=== SCHEMA VALIDATION ===\n")
        await validator.test_design_protein_binder_schema()
        await validator.test_get_job_status_schema()
        await validator.test_list_jobs_schema()
        
        # Test actual execution
        await validator.test_tool_execution()
        
        # Test MCP protocol compliance
        await validator.test_mcp_protocol_compliance()
        
        # Print summary
        validator.print_summary()
        
    finally:
        await validator.close()


if __name__ == "__main__":
    asyncio.run(main())
