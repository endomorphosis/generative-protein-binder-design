#!/usr/bin/env python3
"""
VS Code Integration Test for MCP Server Tools
Simulates tool usage patterns as they would appear in VS Code via GitHub Copilot
"""

import asyncio
import httpx
import json
from typing import Dict, Any, List
from datetime import datetime

MCP_SERVER_URL = "http://localhost:8010"

class VSCodeMCPIntegration:
    """Simulates VS Code's use of MCP tools"""
    
    def __init__(self):
        self.client = None
        self.current_job_id = None
        self.conversation_history = []
    
    async def init(self):
        """Initialize async client"""
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def close(self):
        """Close async client"""
        if self.client:
            await self.client.aclose()
    
    async def log_to_conversation(self, role: str, content: str):
        """Log message to conversation history"""
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.conversation_history.append(message)
        print(f"\n[{role.upper()}]: {content[:100]}...")
    
    async def scenario_1_simple_design(self):
        """Scenario 1: User asks for a simple protein design"""
        print("\n" + "="*80)
        print("SCENARIO 1: Simple Protein Design Request")
        print("="*80)
        
        await self.log_to_conversation(
            "user",
            "Design a binder protein for the human ACE2 protein sequence"
        )
        
        # VS Code would call design_protein_binder tool
        sequence = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"
        
        job_response = await self.client.post(
            f"{MCP_SERVER_URL}/api/jobs",
            json={
                "sequence": sequence[:100],  # Use shorter sequence for faster test
                "job_name": "VSCode-ACE2-Design",
                "num_designs": 3
            }
        )
        
        if job_response.status_code == 200:
            job_data = job_response.json()
            self.current_job_id = job_data["job_id"]
            
            await self.log_to_conversation(
                "assistant",
                f"I've started a protein design job. Job ID: {self.current_job_id}. "
                f"The job will design 3 binder proteins for your target sequence. "
                f"Status: {job_data['status']}"
            )
            
            print(f"✓ Job created successfully: {self.current_job_id}")
            return True
        else:
            print(f"✗ Failed to create job: {job_response.status_code}")
            return False
    
    async def scenario_2_check_progress(self):
        """Scenario 2: User checks progress"""
        print("\n" + "="*80)
        print("SCENARIO 2: Check Job Progress")
        print("="*80)
        
        if not self.current_job_id:
            print("No active job. Run scenario 1 first.")
            return False
        
        await self.log_to_conversation(
            "user",
            f"What's the status of job {self.current_job_id}?"
        )
        
        # VS Code calls get_job_status tool
        status_response = await self.client.get(
            f"{MCP_SERVER_URL}/api/jobs/{self.current_job_id}"
        )
        
        if status_response.status_code == 200:
            job_data = status_response.json()
            status = job_data["status"]
            progress = job_data["progress"]
            
            progress_text = "\n".join([
                f"  • {step}: {status}" 
                for step, status in progress.items()
            ])
            
            await self.log_to_conversation(
                "assistant",
                f"Job Status: {status}\nProgress:\n{progress_text}"
            )
            
            print(f"✓ Job status retrieved: {status}")
            print(f"  Progress:\n{progress_text}")
            return True
        else:
            print(f"✗ Failed to get job status: {status_response.status_code}")
            return False
    
    async def scenario_3_list_all_jobs(self):
        """Scenario 3: User asks for list of jobs"""
        print("\n" + "="*80)
        print("SCENARIO 3: List All Jobs")
        print("="*80)
        
        await self.log_to_conversation(
            "user",
            "Show me all the protein design jobs I've created"
        )
        
        # VS Code calls list_jobs tool
        list_response = await self.client.get(f"{MCP_SERVER_URL}/api/jobs")
        
        if list_response.status_code == 200:
            jobs = list_response.json()
            
            job_summary = "\n".join([
                f"  • {job['job_name'] or job['job_id']}: {job['status']}"
                for job in jobs
            ])
            
            await self.log_to_conversation(
                "assistant",
                f"You have {len(jobs)} protein design jobs:\n{job_summary}"
            )
            
            print(f"✓ Retrieved {len(jobs)} jobs")
            for job in jobs:
                print(f"  - {job['job_name'] or job['job_id']}: {job['status']}")
            return True
        else:
            print(f"✗ Failed to list jobs: {list_response.status_code}")
            return False
    
    async def scenario_4_wait_for_completion(self):
        """Scenario 4: Monitor job until completion"""
        print("\n" + "="*80)
        print("SCENARIO 4: Wait for Job Completion")
        print("="*80)
        
        if not self.current_job_id:
            print("No active job. Run scenario 1 first.")
            return False
        
        await self.log_to_conversation(
            "user",
            f"Wait for job {self.current_job_id} to complete and show me the results"
        )
        
        max_wait = 120  # 2 minutes
        elapsed = 0
        
        while elapsed < max_wait:
            status_response = await self.client.get(
                f"{MCP_SERVER_URL}/api/jobs/{self.current_job_id}"
            )
            
            if status_response.status_code != 200:
                print(f"✗ Failed to get job status")
                return False
            
            job_data = status_response.json()
            status = job_data["status"]
            
            print(f"  Status: {status} (elapsed: {elapsed}s)")
            
            if status in ["completed", "failed"]:
                if status == "completed":
                    await self.log_to_conversation(
                        "assistant",
                        f"Job completed successfully! "
                        f"Generated {len(job_data.get('results', {}).get('designs', []))} designs."
                    )
                    print(f"✓ Job completed successfully")
                    return True
                else:
                    error = job_data.get("error", "Unknown error")
                    await self.log_to_conversation(
                        "assistant",
                        f"Job failed: {error}"
                    )
                    print(f"✗ Job failed: {error}")
                    return False
            
            await asyncio.sleep(2)
            elapsed += 2
        
        print(f"✗ Job did not complete within {max_wait} seconds")
        return False
    
    async def scenario_5_service_health_check(self):
        """Scenario 5: Check backend services health"""
        print("\n" + "="*80)
        print("SCENARIO 5: Check Backend Services")
        print("="*80)
        
        await self.log_to_conversation(
            "user",
            "Are all the protein design services running?"
        )
        
        # Check service status
        health_response = await self.client.get(
            f"{MCP_SERVER_URL}/api/services/status"
        )
        
        if health_response.status_code == 200:
            services = health_response.json()
            
            service_info = []
            all_healthy = True
            
            for service_name, service_status in services.items():
                status = service_status.get("status", "unknown")
                service_info.append(f"  • {service_name}: {status}")
                if status != "ready":
                    all_healthy = False
            
            service_text = "\n".join(service_info)
            
            if all_healthy:
                await self.log_to_conversation(
                    "assistant",
                    f"All services are healthy and ready:\n{service_text}"
                )
            else:
                await self.log_to_conversation(
                    "assistant",
                    f"Some services are not healthy:\n{service_text}"
                )
            
            print(f"✓ Service health check completed")
            print(service_text)
            return True
        else:
            print(f"✗ Failed to check service health: {health_response.status_code}")
            return False
    
    async def scenario_6_tool_discovery(self):
        """Scenario 6: Discover available tools"""
        print("\n" + "="*80)
        print("SCENARIO 6: Tool Discovery (MCP Protocol)")
        print("="*80)
        
        await self.log_to_conversation(
            "user",
            "What tools are available for protein design?"
        )
        
        # Call MCP tools endpoint
        tools_response = await self.client.get(f"{MCP_SERVER_URL}/mcp/v1/tools")
        
        if tools_response.status_code == 200:
            data = tools_response.json()
            tools = data.get("tools", [])
            
            tool_descriptions = []
            for tool in tools:
                tool_descriptions.append(f"  • {tool['name']}: {tool['description']}")
            
            tool_text = "\n".join(tool_descriptions)
            
            await self.log_to_conversation(
                "assistant",
                f"Available tools:\n{tool_text}"
            )
            
            print(f"✓ Found {len(tools)} tools:")
            print(tool_text)
            return True
        else:
            print(f"✗ Failed to discover tools: {tools_response.status_code}")
            return False
    
    def print_conversation_summary(self):
        """Print the conversation as it would appear in VS Code"""
        print("\n" + "="*80)
        print("VS CODE CONVERSATION HISTORY")
        print("="*80)
        
        for msg in self.conversation_history:
            role = msg["role"].upper()
            content = msg["content"]
            
            if role == "USER":
                print(f"\n👤 USER:\n{content}\n")
            else:
                print(f"🤖 ASSISTANT:\n{content}\n")
        
        print("\n" + "="*80)


async def main():
    """Run all scenarios"""
    vscode = VSCodeMCPIntegration()
    await vscode.init()
    
    try:
        print("\n" + "="*80)
        print("VS CODE INTEGRATION TEST FOR MCP SERVER TOOLS")
        print("="*80)
        print(f"Server: {MCP_SERVER_URL}\n")
        
        # Run scenarios
        results = {}
        
        results["Tool Discovery"] = await vscode.scenario_6_tool_discovery()
        results["Service Health"] = await vscode.scenario_5_service_health_check()
        results["Simple Design"] = await vscode.scenario_1_simple_design()
        results["Check Progress"] = await vscode.scenario_2_check_progress()
        results["List Jobs"] = await vscode.scenario_3_list_all_jobs()
        results["Wait Completion"] = await vscode.scenario_4_wait_for_completion()
        
        # Print summary
        print("\n" + "="*80)
        print("TEST RESULTS SUMMARY")
        print("="*80)
        
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        
        for scenario, result in results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"{status}: {scenario}")
        
        print(f"\nTotal: {passed}/{total} scenarios passed")
        
        # Print conversation
        vscode.print_conversation_summary()
        
    finally:
        await vscode.close()


if __name__ == "__main__":
    asyncio.run(main())
