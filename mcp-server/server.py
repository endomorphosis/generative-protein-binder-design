#!/usr/bin/env python3
"""
MCP Server for Protein Binder Design
Implements Model Context Protocol endpoints for managing protein design workflows

Supports multiple backends:
- NIM Backend: NVIDIA NIM containers (default)
- Native Backend: Direct model execution on DGX Spark
- Hybrid Backend: Native with NIM fallback
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
import logging

# Import model backend abstraction
from model_backends import BackendManager, allow_mock_outputs
from runtime_config import RuntimeConfigManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Runtime config + backend manager (supports NIM/external/embedded + fallback)
config_manager = RuntimeConfigManager()
backend_manager = BackendManager(config_manager)

app = FastAPI(
    title="Protein Binder Design MCP Server",
    description=f"Model Context Protocol server for managing protein design workflows\nBackend: {os.getenv('MODEL_BACKEND', 'nim')}",
    version="1.0.0"
)


@app.get("/api/config")
async def get_runtime_config() -> Dict[str, Any]:
    """Get MCP server runtime config (used by the dashboard settings UI)."""
    return config_manager.get().model_dump()


@app.put("/api/config")
async def update_runtime_config(request: Request) -> Dict[str, Any]:
    """Update MCP server runtime config.

    Accepts a partial config object; merges into current config and persists
    to MCP_CONFIG_PATH if set.
    """
    patch = await request.json()
    try:
        updated = config_manager.update(patch)
        # Force backend rebuild on next use.
        _ = backend_manager.get()
        return updated.model_dump()
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/config/reset")
async def reset_runtime_config() -> Dict[str, Any]:
    """Reset runtime config to defaults."""
    try:
        updated = config_manager.reset_to_defaults()
        _ = backend_manager.get()
        return updated.model_dump()
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc))

# Enable CORS for dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify dashboard URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job storage (in production, use a database)
jobs_db: Dict[str, Dict[str, Any]] = {}

# Simple in-memory SSE broadcaster
subscribers: List[asyncio.Queue] = []

async def broadcast_event(event: Dict[str, Any]):
    data = event
    for q in list(subscribers):
        try:
            await q.put(data)
        except Exception:
            # ignore subscriber errors
            pass

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
            ),
            ToolInfo(
                name="delete_job",
                description="Delete a protein design job",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "job_id": {
                            "type": "string",
                            "description": "Job ID to delete"
                        }
                    },
                    "required": ["job_id"]
                }
            ),
            ToolInfo(
                name="check_services",
                description="Check status of all backend services (NIM/native/hybrid)",
                inputSchema={
                    "type": "object",
                    "properties": {}
                }
            ),
            ToolInfo(
                name="predict_structure",
                description="Predict structure from sequence (AlphaFold2 backend)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "sequence": {
                            "type": "string",
                            "description": "Protein amino acid sequence"
                        }
                    },
                    "required": ["sequence"]
                }
            ),
            ToolInfo(
                name="design_binder_backbone",
                description="Generate binder backbones from a target PDB (RFDiffusion backend)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "target_pdb": {
                            "type": "string",
                            "description": "Target protein PDB content (string)"
                        },
                        "num_designs": {
                            "type": "integer",
                            "description": "Number of backbones to generate",
                            "default": 5
                        }
                    },
                    "required": ["target_pdb"]
                }
            ),
            ToolInfo(
                name="generate_sequence",
                description="Generate binder sequence from a backbone PDB (ProteinMPNN backend)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "backbone_pdb": {
                            "type": "string",
                            "description": "Backbone PDB content (string)"
                        }
                    },
                    "required": ["backbone_pdb"]
                }
            ),
            ToolInfo(
                name="predict_complex",
                description="Predict complex structure from sequences (AlphaFold2-Multimer backend)",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "sequences": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of chain sequences"
                        }
                    },
                    "required": ["sequences"]
                }
            )
        ]
    }


def _jsonrpc_error(_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": _id, "error": {"code": code, "message": message}}


def _jsonrpc_result(_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": _id, "result": result}


def _mcp_initialize_result() -> Dict[str, Any]:
    # Keep in sync with tools/mcp_stdio_adapter.py
    return {
        "protocolVersion": "2024-11-05",
        "capabilities": {
            "tools": {},
            "resources": {},
            "logging": {},
        },
        "serverInfo": {"name": "protein-binder-mcp-server", "version": "1.0.0"},
    }


@app.post("/mcp")
async def mcp_jsonrpc(request: Request) -> Dict[str, Any]:
    """JSON-RPC 2.0 endpoint for MCP clients that use HTTP transport.

    This is in addition to the REST-style MCP endpoints under /mcp/v1/*.
    """
    try:
        message = await request.json()
    except Exception:
        return _jsonrpc_error(None, -32700, "Parse error")

    if not isinstance(message, dict):
        return _jsonrpc_error(None, -32600, "Invalid Request")

    msg_id = message.get("id")
    method = message.get("method")
    params = message.get("params") or {}

    try:
        model_backend = backend_manager.get()
        if method == "initialize":
            return _jsonrpc_result(msg_id, _mcp_initialize_result())

        if method == "tools/list":
            tools = (await list_tools()).get("tools", [])
            return _jsonrpc_result(msg_id, {"tools": tools})

        if method == "tools/call":
            name = params.get("name")
            arguments = params.get("arguments") or {}

            if name == "design_protein_binder":
                input_data = ProteinSequenceInput(**arguments)
                job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(jobs_db)}"
                job = {
                    "job_id": job_id,
                    "status": "created",
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "job_name": input_data.job_name,
                    "input": {"sequence": input_data.sequence, "num_designs": input_data.num_designs},
                    "progress": {
                        "alphafold": "pending",
                        "rfdiffusion": "pending",
                        "proteinmpnn": "pending",
                        "alphafold_multimer": "pending",
                    },
                    "results": None,
                    "error": None,
                }
                jobs_db[job_id] = job
                try:
                    asyncio.create_task(broadcast_event({"type": "job.created", "job": job}))
                except Exception:
                    pass
                asyncio.create_task(process_job(job_id))
                return _jsonrpc_result(
                    msg_id,
                    {
                        "content": [{"type": "text", "text": json.dumps(JobStatus(**job).model_dump(), indent=2)}],
                        "isError": False,
                    },
                )

            if name == "get_job_status":
                job_id = arguments.get("job_id")
                if not job_id or job_id not in jobs_db:
                    return _jsonrpc_error(msg_id, -32004, "Job not found")
                job = JobStatus(**jobs_db[job_id]).model_dump()
                return _jsonrpc_result(
                    msg_id,
                    {"content": [{"type": "text", "text": json.dumps(job, indent=2)}], "isError": False},
                )

            if name == "list_jobs":
                jobs = [JobStatus(**j).model_dump() for j in jobs_db.values()]
                return _jsonrpc_result(
                    msg_id,
                    {"content": [{"type": "text", "text": json.dumps(jobs, indent=2)}], "isError": False},
                )

            if name == "delete_job":
                job_id = arguments.get("job_id")
                if not job_id or job_id not in jobs_db:
                    return _jsonrpc_error(msg_id, -32004, "Job not found")
                deleted = jobs_db.pop(job_id)
                try:
                    asyncio.create_task(broadcast_event({"type": "job.deleted", "job": deleted}))
                except Exception:
                    pass
                return _jsonrpc_result(
                    msg_id,
                    {
                        "content": [{"type": "text", "text": json.dumps({"deleted": job_id}, indent=2)}],
                        "isError": False,
                    },
                )

            if name == "check_services":
                status = await model_backend.check_health()
                return _jsonrpc_result(
                    msg_id,
                    {"content": [{"type": "text", "text": json.dumps(status, indent=2)}], "isError": False},
                )

            if name == "predict_structure":
                sequence = arguments.get("sequence")
                if not sequence:
                    return _jsonrpc_error(msg_id, -32602, "Missing sequence")
                result = await model_backend.predict_structure(sequence)
                return _jsonrpc_result(
                    msg_id,
                    {"content": [{"type": "text", "text": json.dumps(result, indent=2)}], "isError": False},
                )

            if name == "design_binder_backbone":
                target_pdb = arguments.get("target_pdb")
                if not target_pdb:
                    return _jsonrpc_error(msg_id, -32602, "Missing target_pdb")
                num_designs = int(arguments.get("num_designs") or 5)
                result = await model_backend.design_binder_backbone(target_pdb, num_designs)
                return _jsonrpc_result(
                    msg_id,
                    {"content": [{"type": "text", "text": json.dumps(result, indent=2)}], "isError": False},
                )

            if name == "generate_sequence":
                backbone_pdb = arguments.get("backbone_pdb")
                if not backbone_pdb:
                    return _jsonrpc_error(msg_id, -32602, "Missing backbone_pdb")
                result = await model_backend.generate_sequence(backbone_pdb)
                return _jsonrpc_result(
                    msg_id,
                    {"content": [{"type": "text", "text": json.dumps(result, indent=2)}], "isError": False},
                )

            if name == "predict_complex":
                sequences = arguments.get("sequences")
                if not sequences or not isinstance(sequences, list):
                    return _jsonrpc_error(msg_id, -32602, "Missing sequences")
                result = await model_backend.predict_complex(sequences)
                return _jsonrpc_result(
                    msg_id,
                    {"content": [{"type": "text", "text": json.dumps(result, indent=2)}], "isError": False},
                )

            return _jsonrpc_error(msg_id, -32601, f"Unknown tool: {name}")

        if method == "resources/list":
            resources = (await list_resources()).get("resources", [])
            return _jsonrpc_result(msg_id, {"resources": resources})

        if method == "resources/read":
            uri = params.get("uri") or params.get("path")
            if not uri:
                return _jsonrpc_error(msg_id, -32602, "Missing resource uri")
            if uri.startswith("job://"):
                job_id = uri.replace("job://", "")
            else:
                job_id = uri
            contents = (await get_resource(job_id)).get("contents", [])
            return _jsonrpc_result(msg_id, {"contents": contents})

        if method in {"shutdown", "exit"}:
            return _jsonrpc_result(msg_id, None)

        return _jsonrpc_error(msg_id, -32601, f"Method not found: {method}")
    except Exception as exc:
        return _jsonrpc_error(msg_id, -32603, str(exc))

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
    
    # notify subscribers about new job
    try:
        asyncio.create_task(broadcast_event({"type": "job.created", "job": job}))
    except Exception:
        pass
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
    """Check status of all backend services"""
    return await backend_manager.get().check_health()


@app.get('/sse')
async def sse_endpoint(request: Request):
    async def event_generator():
        q: asyncio.Queue = asyncio.Queue()
        subscribers.append(q)
        try:
            # immediate handshake
            yield "data: ready\n\n"
            while True:
                try:
                    data = await asyncio.wait_for(q.get(), timeout=15.0)
                    yield f"data: {json.dumps(data)}\n\n"
                except asyncio.TimeoutError:
                    # heartbeat to keep connection alive
                    yield "data: ping\n\n"
                except asyncio.CancelledError:
                    break
        finally:
            try:
                subscribers.remove(q)
            except ValueError:
                pass

    return StreamingResponse(event_generator(), media_type='text/event-stream')


@app.get('/mcp/sse')
async def mcp_sse_endpoint(request: Request):
    """Alias of /sse for clients expecting an MCP-namespaced SSE endpoint."""
    return await sse_endpoint(request)

# Background job processing
async def process_job(job_id: str):
    """Process a protein binder design job using configured backend"""
    try:
        model_backend = backend_manager.get()
        job = jobs_db[job_id]
        job["status"] = "running"
        job["updated_at"] = datetime.now().isoformat()
        try:
            asyncio.create_task(broadcast_event({"type": "job.updated", "job": job}))
        except Exception:
            pass
        
        sequence = job["input"]["sequence"]
        num_designs = job["input"]["num_designs"]
        
        # Step 1: AlphaFold2 - predict structure of target
        logger.info(f"Job {job_id}: Running AlphaFold2")
        job["progress"]["alphafold"] = "running"
        
        try:
            alphafold_result = await model_backend.predict_structure(sequence)
            job["progress"]["alphafold"] = "completed"
            logger.info(f"Job {job_id}: AlphaFold2 completed")
        except Exception as e:
            logger.error(f"AlphaFold2 error: {e}")
            job["progress"]["alphafold"] = f"error: {str(e)}"
            if allow_mock_outputs():
                # For CI/demo only
                alphafold_result = {"pdb": "mock_structure"}
            else:
                job["status"] = "failed"
                job["error"] = f"AlphaFold2 failed: {e}"
                job["updated_at"] = datetime.now().isoformat()
                try:
                    asyncio.create_task(broadcast_event({"type": "job.failed", "job": job}))
                except Exception:
                    pass
                return
        try:
            asyncio.create_task(broadcast_event({"type": "job.updated", "job": job}))
        except Exception:
            pass
        
        # Step 2: RFDiffusion - generate binder backbones
        logger.info(f"Job {job_id}: Running RFDiffusion")
        job["progress"]["rfdiffusion"] = "running"
        
        try:
            rfdiffusion_result = await model_backend.design_binder_backbone(
                alphafold_result.get("pdb", ""),
                num_designs
            )
            job["progress"]["rfdiffusion"] = "completed"
            logger.info(f"Job {job_id}: RFDiffusion completed")
        except Exception as e:
            logger.error(f"RFDiffusion error: {e}")
            job["progress"]["rfdiffusion"] = f"error: {str(e)}"
            if allow_mock_outputs():
                rfdiffusion_result = {"designs": [{"pdb": f"mock_design_{i}"} for i in range(num_designs)]}
            else:
                job["status"] = "failed"
                job["error"] = f"RFDiffusion failed: {e}"
                job["updated_at"] = datetime.now().isoformat()
                try:
                    asyncio.create_task(broadcast_event({"type": "job.failed", "job": job}))
                except Exception:
                    pass
                return
        try:
            asyncio.create_task(broadcast_event({"type": "job.updated", "job": job}))
        except Exception:
            pass
        
        # Step 3: ProteinMPNN - generate sequences for backbones
        logger.info(f"Job {job_id}: Running ProteinMPNN")
        job["progress"]["proteinmpnn"] = "running"
        
        try:
            mpnn_results = []
            for design in rfdiffusion_result.get("designs", [])[:num_designs]:
                mpnn_result = await model_backend.generate_sequence(design.get("pdb", ""))
                mpnn_results.append(mpnn_result)
            job["progress"]["proteinmpnn"] = "completed"
            logger.info(f"Job {job_id}: ProteinMPNN completed")
        except Exception as e:
            logger.error(f"ProteinMPNN error: {e}")
            job["progress"]["proteinmpnn"] = f"error: {str(e)}"
            if allow_mock_outputs():
                mpnn_results = [{"sequence": f"MOCK_SEQ_{i}"} for i in range(num_designs)]
            else:
                job["status"] = "failed"
                job["error"] = f"ProteinMPNN failed: {e}"
                job["updated_at"] = datetime.now().isoformat()
                try:
                    asyncio.create_task(broadcast_event({"type": "job.failed", "job": job}))
                except Exception:
                    pass
                return
        try:
            asyncio.create_task(broadcast_event({"type": "job.updated", "job": job}))
        except Exception:
            pass
        
        # Step 4: AlphaFold2-Multimer - predict complex structures
        logger.info(f"Job {job_id}: Running AlphaFold2-Multimer")
        job["progress"]["alphafold_multimer"] = "running"
        
        try:
            multimer_results = []
            for i, mpnn_result in enumerate(mpnn_results[:num_designs]):
                multimer_result = await model_backend.predict_complex([
                    sequence,
                    mpnn_result.get("sequence", "")
                ])
                multimer_results.append(multimer_result)
            job["progress"]["alphafold_multimer"] = "completed"
            logger.info(f"Job {job_id}: AlphaFold2-Multimer completed")
        except Exception as e:
            logger.error(f"AlphaFold2-Multimer error: {e}")
            job["progress"]["alphafold_multimer"] = f"error: {str(e)}"
            if allow_mock_outputs():
                multimer_results = [{"pdb": f"mock_complex_{i}"} for i in range(num_designs)]
            else:
                job["status"] = "failed"
                job["error"] = f"AlphaFold2-Multimer failed: {e}"
                job["updated_at"] = datetime.now().isoformat()
                try:
                    asyncio.create_task(broadcast_event({"type": "job.failed", "job": job}))
                except Exception:
                    pass
                return
        try:
            asyncio.create_task(broadcast_event({"type": "job.updated", "job": job}))
        except Exception:
            pass
        
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
        try:
            asyncio.create_task(broadcast_event({"type": "job.completed", "job": job}))
        except Exception:
            pass
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        job["status"] = "failed"
        job["error"] = str(e)
        job["updated_at"] = datetime.now().isoformat()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
