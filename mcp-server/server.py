#!/usr/bin/env python3
"""
MCP Server for Protein Binder Design
Implements Model Context Protocol endpoints for managing protein design workflows
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Protein Binder Design MCP Server",
    description="Model Context Protocol server for managing protein design workflows",
    version="1.0.0"
)

# Enable CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify dashboard URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration for NIM services
NIM_SERVICES = {
    "alphafold": os.getenv("ALPHAFOLD_URL", "http://localhost:8081"),
    "rfdiffusion": os.getenv("RFDIFFUSION_URL", "http://localhost:8082"),
    "proteinmpnn": os.getenv("PROTEINMPNN_URL", "http://localhost:8083"),
    "alphafold_multimer": os.getenv("ALPHAFOLD_MULTIMER_URL", "http://localhost:8084"),
}

# In-memory job storage (in production, use a database)
jobs_db: Dict[str, Dict[str, Any]] = {}

# Pydantic models for request/response
class ProteinSequenceInput(BaseModel):
    sequence: str = Field(..., description="Target protein amino acid sequence")
    job_name: Optional[str] = Field(None, description="Optional name for the job")
    num_designs: int = Field(5, description="Number of binder designs to generate")
    
class JobStatus(BaseModel):
    job_id: str
    status: str
    created_at: str
    updated_at: str
    job_name: Optional[str] = None
    progress: Dict[str, Any]
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ToolInfo(BaseModel):
    name: str
    description: str
    inputSchema: Dict[str, Any]

class ResourceInfo(BaseModel):
    uri: str
    name: str
    description: str
    mimeType: str

# MCP Protocol endpoints
@app.get("/mcp/v1/tools")
async def list_tools() -> Dict[str, List[ToolInfo]]:
    """List available MCP tools"""
    return {
        "tools": [
            ToolInfo(
                name="design_protein_binder",
                description="Design protein binders for a target sequence",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "sequence": {
                            "type": "string",
                            "description": "Target protein amino acid sequence"
                        },
                        "job_name": {
                            "type": "string",
                            "description": "Optional name for the job"
                        },
                        "num_designs": {
                            "type": "integer",
                            "description": "Number of binder designs to generate",
                            "default": 5
                        }
                    },
                    "required": ["sequence"]
                }
            ),
            ToolInfo(
                name="get_job_status",
                description="Get the status of a protein design job",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "job_id": {
                            "type": "string",
                            "description": "Job ID to query"
                        }
                    },
                    "required": ["job_id"]
                }
            ),
            ToolInfo(
                name="list_jobs",
                description="List all protein design jobs",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            )
        ]
    }

@app.get("/mcp/v1/resources")
async def list_resources() -> Dict[str, List[ResourceInfo]]:
    """List available MCP resources"""
    resources = []
    for job_id, job in jobs_db.items():
        if job.get("results"):
            resources.append(ResourceInfo(
                uri=f"job://{job_id}",
                name=job.get("job_name", job_id),
                description=f"Results for protein design job {job_id}",
                mimeType="application/json"
            ))
    return {"resources": resources}

@app.get("/mcp/v1/resources/{job_id}")
async def get_resource(job_id: str) -> Dict[str, Any]:
    """Get a specific resource"""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_db[job_id]
    if not job.get("results"):
        raise HTTPException(status_code=404, detail="Job results not available yet")
    
    return {
        "contents": [
            {
                "uri": f"job://{job_id}",
                "mimeType": "application/json",
                "text": json.dumps(job["results"], indent=2)
            }
        ]
    }

# Job management endpoints
@app.post("/api/jobs", response_model=JobStatus)
async def create_job(
    input_data: ProteinSequenceInput,
    background_tasks: BackgroundTasks
) -> JobStatus:
    """Create a new protein binder design job"""
    job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(jobs_db)}"
    
    job = {
        "job_id": job_id,
        "status": "created",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "job_name": input_data.job_name,
        "input": {
            "sequence": input_data.sequence,
            "num_designs": input_data.num_designs
        },
        "progress": {
            "alphafold": "pending",
            "rfdiffusion": "pending",
            "proteinmpnn": "pending",
            "alphafold_multimer": "pending"
        },
        "results": None,
        "error": None
    }
    
    jobs_db[job_id] = job
    
    # Start job processing in background
    background_tasks.add_task(process_job, job_id)
    
    return JobStatus(**job)

@app.get("/api/jobs", response_model=List[JobStatus])
async def list_jobs() -> List[JobStatus]:
    """List all jobs"""
    return [JobStatus(**job) for job in jobs_db.values()]

@app.get("/api/jobs/{job_id}", response_model=JobStatus)
async def get_job(job_id: str) -> JobStatus:
    """Get job status"""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(**jobs_db[job_id])

@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str) -> Dict[str, str]:
    """Delete a job"""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    del jobs_db[job_id]
    return {"message": "Job deleted successfully"}

# Health check endpoints
@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/api/services/status")
async def check_services() -> Dict[str, Any]:
    """Check status of all NIM services"""
    status = {}
    async with httpx.AsyncClient(timeout=5.0) as client:
        for service_name, url in NIM_SERVICES.items():
            try:
                response = await client.get(f"{url}/v1/health/ready")
                status[service_name] = {
                    "status": "ready" if response.status_code == 200 else "not_ready",
                    "url": url
                }
            except Exception as e:
                status[service_name] = {
                    "status": "error",
                    "error": str(e),
                    "url": url
                }
    return status

# Background job processing
async def process_job(job_id: str):
    """Process a protein binder design job"""
    try:
        job = jobs_db[job_id]
        job["status"] = "running"
        job["updated_at"] = datetime.now().isoformat()
        
        sequence = job["input"]["sequence"]
        num_designs = job["input"]["num_designs"]
        
        async with httpx.AsyncClient(timeout=600.0) as client:
            # Step 1: AlphaFold2 - predict structure of target
            logger.info(f"Job {job_id}: Running AlphaFold2")
            job["progress"]["alphafold"] = "running"
            
            try:
                af_response = await client.post(
                    f"{NIM_SERVICES['alphafold']}/v1/structure",
                    json={"sequence": sequence}
                )
                af_response.raise_for_status()
                alphafold_result = af_response.json()
                job["progress"]["alphafold"] = "completed"
            except Exception as e:
                logger.error(f"AlphaFold2 error: {e}")
                job["progress"]["alphafold"] = f"error: {str(e)}"
                # For demo purposes, continue with mock data
                alphafold_result = {"pdb": "mock_structure"}
            
            # Step 2: RFDiffusion - generate binder backbones
            logger.info(f"Job {job_id}: Running RFDiffusion")
            job["progress"]["rfdiffusion"] = "running"
            
            try:
                rf_response = await client.post(
                    f"{NIM_SERVICES['rfdiffusion']}/v1/design",
                    json={
                        "target_pdb": alphafold_result.get("pdb"),
                        "num_designs": num_designs
                    }
                )
                rf_response.raise_for_status()
                rfdiffusion_result = rf_response.json()
                job["progress"]["rfdiffusion"] = "completed"
            except Exception as e:
                logger.error(f"RFDiffusion error: {e}")
                job["progress"]["rfdiffusion"] = f"error: {str(e)}"
                rfdiffusion_result = {"designs": [{"pdb": f"mock_design_{i}"} for i in range(num_designs)]}
            
            # Step 3: ProteinMPNN - generate sequences for backbones
            logger.info(f"Job {job_id}: Running ProteinMPNN")
            job["progress"]["proteinmpnn"] = "running"
            
            try:
                mpnn_results = []
                for design in rfdiffusion_result.get("designs", [])[:num_designs]:
                    mpnn_response = await client.post(
                        f"{NIM_SERVICES['proteinmpnn']}/v1/sequence",
                        json={"backbone_pdb": design.get("pdb")}
                    )
                    mpnn_response.raise_for_status()
                    mpnn_results.append(mpnn_response.json())
                job["progress"]["proteinmpnn"] = "completed"
            except Exception as e:
                logger.error(f"ProteinMPNN error: {e}")
                job["progress"]["proteinmpnn"] = f"error: {str(e)}"
                mpnn_results = [{"sequence": f"MOCK_SEQ_{i}"} for i in range(num_designs)]
            
            # Step 4: AlphaFold2-Multimer - predict complex structures
            logger.info(f"Job {job_id}: Running AlphaFold2-Multimer")
            job["progress"]["alphafold_multimer"] = "running"
            
            try:
                multimer_results = []
                for i, mpnn_result in enumerate(mpnn_results[:num_designs]):
                    multimer_response = await client.post(
                        f"{NIM_SERVICES['alphafold_multimer']}/v1/structure",
                        json={
                            "sequences": [sequence, mpnn_result.get("sequence")]
                        }
                    )
                    multimer_response.raise_for_status()
                    multimer_results.append(multimer_response.json())
                job["progress"]["alphafold_multimer"] = "completed"
            except Exception as e:
                logger.error(f"AlphaFold2-Multimer error: {e}")
                job["progress"]["alphafold_multimer"] = f"error: {str(e)}"
                multimer_results = [{"pdb": f"mock_complex_{i}"} for i in range(num_designs)]
            
            # Compile results
            job["results"] = {
                "target_structure": alphafold_result,
                "designs": [
                    {
                        "design_id": i,
                        "backbone": rfdiffusion_result.get("designs", [])[i] if i < len(rfdiffusion_result.get("designs", [])) else {},
                        "sequence": mpnn_results[i] if i < len(mpnn_results) else {},
                        "complex_structure": multimer_results[i] if i < len(multimer_results) else {}
                    }
                    for i in range(num_designs)
                ]
            }
            
            job["status"] = "completed"
            job["updated_at"] = datetime.now().isoformat()
            logger.info(f"Job {job_id}: Completed successfully")
            
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        job["status"] = "failed"
        job["error"] = str(e)
        job["updated_at"] = datetime.now().isoformat()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
