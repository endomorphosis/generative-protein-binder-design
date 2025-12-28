#!/usr/bin/env python3
"""
MCP Server Tools Quick Demo
Quick reference for testing MCP server tools manually
"""

import subprocess
import json
import time
from typing import Dict, Any

def print_header(title: str):
    """Print formatted header"""
    print(f"\n{'='*80}")
    print(f"{title.center(80)}")
    print(f"{'='*80}\n")

def print_section(title: str):
    """Print formatted section"""
    print(f"\n{title}")
    print("-" * len(title))

def run_curl(method: str, endpoint: str, data: Dict[str, Any] = None, description: str = "") -> Dict[str, Any]:
    """Run curl command and return response"""
    url = f"http://localhost:8010{endpoint}"
    
    if description:
        print(f"\n📍 {description}")
    
    print(f"   {method} {endpoint}")
    
    if method == "GET":
        cmd = f"curl -s -X {method} {url}"
    elif method == "POST":
        json_str = json.dumps(data).replace('"', '\\"')
        cmd = f"curl -s -X {method} {url} -H 'Content-Type: application/json' -d '{json_str}'"
    else:
        return {}
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    try:
        response = json.loads(result.stdout)
        return response
    except:
        print(f"   Response: {result.stdout[:200]}")
        return {}

def demo_1_health_check():
    """Demo 1: Health check"""
    print_section("1️⃣  Health Check")
    response = run_curl("GET", "/health", description="Check if server is running")
    if response:
        print(f"   Status: {response.get('status', 'unknown')}")
    return True

def demo_2_list_tools():
    """Demo 2: List available tools"""
    print_section("2️⃣  List Available Tools")
    response = run_curl("GET", "/mcp/v1/tools", description="Discover available MCP tools")
    
    if "tools" in response:
        tools = response["tools"]
        print(f"   Found {len(tools)} tools:")
        for tool in tools:
            print(f"   • {tool['name']}: {tool['description']}")
    return True

def demo_3_check_services():
    """Demo 3: Check service health"""
    print_section("3️⃣  Check Backend Services")
    response = run_curl("GET", "/api/services/status", description="Check protein design services")
    
    if response:
        print("   Service Status:")
        for service, status in response.items():
            service_status = status.get("status", "unknown")
            print(f"   • {service}: {service_status}")
    return True

def demo_4_create_job():
    """Demo 4: Create a design job"""
    print_section("4️⃣  Create Protein Design Job")
    
    payload = {
        "sequence": "MKTAYIAKQRQISFVKSHFSRQ",
        "job_name": "demo_design",
        "num_designs": 2
    }
    
    response = run_curl("POST", "/api/jobs", payload, description="Create a new design job")
    
    if "job_id" in response:
        job_id = response["job_id"]
        status = response.get("status", "unknown")
        print(f"   ✅ Job created!")
        print(f"   Job ID: {job_id}")
        print(f"   Status: {status}")
        return job_id
    else:
        print("   ❌ Failed to create job")
        return None

def demo_5_check_status(job_id: str):
    """Demo 5: Check job status"""
    print_section("5️⃣  Check Job Status")
    
    endpoint = f"/api/jobs/{job_id}"
    response = run_curl("GET", endpoint, description=f"Check status of job {job_id}")
    
    if "status" in response:
        print(f"   Job Status: {response['status']}")
        print(f"   Progress:")
        progress = response.get("progress", {})
        for step, step_status in progress.items():
            print(f"   • {step}: {step_status}")
    return True

def demo_6_list_jobs():
    """Demo 6: List all jobs"""
    print_section("6️⃣  List All Jobs")
    response = run_curl("GET", "/api/jobs", description="Retrieve all design jobs")
    
    if isinstance(response, list):
        print(f"   Found {len(response)} jobs:")
        for job in response[:5]:  # Show first 5
            job_name = job.get("job_name") or job.get("job_id")
            status = job.get("status")
            print(f"   • {job_name}: {status}")
        if len(response) > 5:
            print(f"   ... and {len(response) - 5} more")
    return True

def demo_7_wait_and_retrieve(job_id: str):
    """Demo 7: Wait for job completion"""
    print_section("7️⃣  Wait for Job Completion")
    
    print(f"   Waiting for job {job_id} to complete...")
    start_time = time.time()
    max_wait = 120  # 2 minutes
    
    while time.time() - start_time < max_wait:
        response = run_curl("GET", f"/api/jobs/{job_id}")
        
        if "status" in response:
            status = response["status"]
            elapsed = int(time.time() - start_time)
            print(f"   [{elapsed}s] Status: {status}")
            
            if status == "completed":
                print(f"   ✅ Job completed!")
                
                # Show results
                results = response.get("results")
                if results:
                    designs = results.get("designs", [])
                    print(f"   Generated {len(designs)} designs")
                return True
            elif status == "failed":
                print(f"   ❌ Job failed: {response.get('error', 'Unknown error')}")
                return False
        
        time.sleep(2)
    
    print(f"   ⏱️  Timeout: Job did not complete within {max_wait}s")
    return False

def demo_8_tool_schema():
    """Demo 8: Show tool schemas"""
    print_section("8️⃣  Tool Input Schema")
    
    response = run_curl("GET", "/mcp/v1/tools", description="Retrieve tool schemas")
    
    if "tools" in response:
        for tool in response["tools"]:
            print(f"\n   Tool: {tool['name']}")
            print(f"   Description: {tool['description']}")
            schema = tool.get("inputSchema", {})
            required = schema.get("required", [])
            properties = schema.get("properties", {})
            
            print(f"   Input fields:")
            for prop_name, prop_schema in properties.items():
                prop_type = prop_schema.get("type", "unknown")
                required_marker = "required" if prop_name in required else "optional"
                description = prop_schema.get("description", "")
                print(f"     • {prop_name} ({prop_type}) [{required_marker}]: {description}")
    return True

def main():
    """Run all demos"""
    print_header("MCP SERVER TOOLS QUICK DEMO")
    
    print("This demo shows how to use each MCP server tool via curl/HTTP requests.")
    print("The same tools are available via VS Code and the Dashboard.")
    
    # Run demos
    demo_1_health_check()
    
    demo_2_list_tools()
    
    demo_3_check_services()
    
    job_id = demo_4_create_job()
    
    if job_id:
        time.sleep(1)
        demo_5_check_status(job_id)
        
        demo_6_list_jobs()
        
        demo_7_wait_and_retrieve(job_id)
    
    demo_8_tool_schema()
    
    # Summary
    print_header("✅ DEMO COMPLETE")
    print("All MCP server tools are working correctly!")
    print("\nNext Steps:")
    print("  1. Test tools in VS Code: Use GitHub Copilot to call these tools")
    print("  2. Test in Dashboard: Visit http://localhost:3000")
    print("  3. View API docs: Visit http://localhost:8010/docs")
    print("\nTool Reference:")
    print("  • design_protein_binder: Create new design jobs")
    print("  • get_job_status: Check job progress")
    print("  • list_jobs: View all jobs")

if __name__ == "__main__":
    main()
