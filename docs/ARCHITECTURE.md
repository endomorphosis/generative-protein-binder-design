# Architecture Documentation

## System Architecture

The Protein Binder Design system with MCP Server and Dashboard consists of multiple interconnected components:

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Layer                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌───────────────┐              ┌───────────────┐               │
│  │  Web Browser  │              │  Web Browser  │               │
│  │  (Dashboard)  │              │  (Jupyter)    │               │
│  └───────┬───────┘              └───────┬───────┘               │
│          │                               │                       │
│          │ HTTP/REST                     │ HTTP                  │
│          │                               │                       │
└──────────┼───────────────────────────────┼───────────────────────┘
           │                               │
           ▼                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Application Layer                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌───────────────────┐         ┌───────────────────┐            │
│  │  MCP Dashboard    │         │  Jupyter Server   │            │
│  │  (Next.js/React)  │         │  (Python/Jupyter) │            │
│  │  Port: 3000       │         │  Port: 8888       │            │
│  └─────────┬─────────┘         └───────────────────┘            │
│            │                                                      │
│            │ MCP Protocol                                         │
│            │ REST API                                             │
│            │                                                      │
│            ▼                                                      │
│  ┌───────────────────┐                                           │
│  │   MCP Server      │                                           │
│  │   (FastAPI)       │                                           │
│  │   Port: 8000      │                                           │
│  │                   │                                           │
│  │  Components:      │                                           │
│  │  • Job Manager    │                                           │
│  │  • MCP Protocol   │                                           │
│  │  • Workflow       │                                           │
│  │    Orchestrator   │                                           │
│  └─────────┬─────────┘                                           │
│            │                                                      │
└────────────┼──────────────────────────────────────────────────────┘
             │
             │ HTTP/REST
             │
             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    NIM Service Layer                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ AlphaFold2   │  │ RFDiffusion  │  │ ProteinMPNN  │          │
│  │ (NIM)        │  │ (NIM)        │  │ (NIM)        │          │
│  │ Port: 8081   │  │ Port: 8082   │  │ Port: 8083   │          │
│  │ GPU: 0       │  │ GPU: 1       │  │ GPU: 2       │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                   │
│  ┌──────────────┐                                                │
│  │ AlphaFold2   │                                                │
│  │ Multimer     │                                                │
│  │ (NIM)        │                                                │
│  │ Port: 8084   │                                                │
│  │ GPU: 3       │                                                │
│  └──────────────┘                                                │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. MCP Dashboard (Frontend)
- **Technology**: Next.js 14, React 18, TypeScript, Tailwind CSS
- **Purpose**: User interface for job submission and result visualization
- **Key Features**:
  - Real-time job monitoring
  - Interactive form for protein sequence input
  - Service health status display
  - Result visualization and download
  - Jupyter notebook launcher

### 2. MCP Server (Backend)
- **Technology**: FastAPI, Python 3.11, Uvicorn
- **Purpose**: Workflow orchestration and job management
- **Key Features**:
  - RESTful API for job management
  - Model Context Protocol (MCP) implementation
  - Background job processing
  - Service health monitoring
  - Job state management

### 3. Jupyter Server
- **Technology**: Jupyter Notebook, Python 3.11
- **Purpose**: Interactive notebook environment
- **Key Features**:
  - Pre-configured environment with bioinformatics packages
  - Example notebooks
  - Direct access to NIM services

### 4. NIM Services
- **AlphaFold2**: Predicts protein structure from sequence
- **RFDiffusion**: Generates protein backbone designs
- **ProteinMPNN**: Predicts sequences for backbones
- **AlphaFold2-Multimer**: Predicts complex structures

## Data Flow

### Job Processing Workflow

```
1. User submits protein sequence via Dashboard
         ↓
2. Dashboard sends POST request to MCP Server
         ↓
3. MCP Server creates job and returns job_id
         ↓
4. MCP Server starts background processing:
   
   Step 1: AlphaFold2
   - Input: Protein sequence
   - Output: Target protein structure (PDB)
         ↓
   Step 2: RFDiffusion
   - Input: Target structure
   - Output: N binder backbone designs (PDB)
         ↓
   Step 3: ProteinMPNN
   - Input: Binder backbones
   - Output: N binder sequences
         ↓
   Step 4: AlphaFold2-Multimer
   - Input: Target + Binder sequences
   - Output: N complex structures (PDB)
         ↓
5. Results stored and available via API
         ↓
6. Dashboard polls for updates and displays results
```

## API Architecture

### MCP Protocol Endpoints
```
GET  /mcp/v1/tools              # List available tools
GET  /mcp/v1/resources          # List available resources
GET  /mcp/v1/resources/{id}     # Get specific resource
```

### Job Management Endpoints
```
POST   /api/jobs                # Create new job
GET    /api/jobs                # List all jobs
GET    /api/jobs/{id}           # Get job status
DELETE /api/jobs/{id}           # Delete job
```

### Health & Monitoring
```
GET /health                     # Server health
GET /api/services/status        # NIM services status
```

## Deployment Options

### Option 1: Full Stack (Production)
```yaml
../deploy/docker-compose-full.yaml
  - All NIM services
  - MCP Server
  - MCP Dashboard
  - Jupyter Server
```

### Option 2: Development (Local)
```bash
# Terminal 1: NIM Services
cd deploy && docker compose up

# Terminal 2: MCP Server
cd mcp-server && python server.py

# Terminal 3: Dashboard
cd mcp-dashboard && npm run dev

# Terminal 4: Jupyter
cd src && jupyter notebook
```

### Option 3: NIMs Only (Original)
```bash
cd deploy && docker compose up
```

## Security Considerations

### Current Implementation (Development)
- No authentication on MCP Server
- No token required for Jupyter
- CORS enabled for all origins
- No HTTPS/TLS

### Production Recommendations
1. **Authentication**: Add JWT or OAuth2 to MCP Server
2. **Authorization**: Implement role-based access control
3. **Secrets**: Use secrets management (e.g., HashiCorp Vault)
4. **TLS**: Enable HTTPS with proper certificates
5. **CORS**: Restrict to specific origins
6. **Rate Limiting**: Implement API rate limiting
7. **Input Validation**: Strict validation of protein sequences
8. **Jupyter Security**: Enable token/password authentication

## Scalability Considerations

### Current Limitations
- In-memory job storage (lost on restart)
- Single MCP Server instance
- Synchronous job processing

### Production Recommendations
1. **Database**: Use PostgreSQL/MongoDB for job storage
2. **Message Queue**: Add Celery/RabbitMQ for async processing
3. **Caching**: Implement Redis for caching
4. **Load Balancing**: Multiple MCP Server instances
5. **Container Orchestration**: Use Kubernetes for auto-scaling
6. **Monitoring**: Add Prometheus/Grafana
7. **Logging**: Centralized logging (ELK stack)

## Network Architecture

```
Port Mapping:
3000  → MCP Dashboard (HTTP)
8000  → MCP Server (HTTP)
8081  → AlphaFold2 NIM
8082  → RFDiffusion NIM
8083  → ProteinMPNN NIM
8084  → AlphaFold2-Multimer NIM
8888  → Jupyter Notebook (HTTP)
```

## Technology Stack Summary

| Component | Technology | Language | Framework |
|-----------|-----------|----------|-----------|
| Dashboard | Frontend | TypeScript | Next.js 14, React 18 |
| MCP Server | Backend | Python 3.11 | FastAPI, Uvicorn |
| Jupyter | Interactive | Python 3.11 | Jupyter Notebook |
| NIMs | AI Services | N/A | NVIDIA NIMs |

## Development Workflow

```
1. Code Changes
         ↓
2. Local Testing
   - Lint: npm run lint (Dashboard)
   - Type Check: npx tsc --noEmit
   - Build: npm run build
   - Test: ./scripts/test-mcp-server.sh
         ↓
3. Docker Build
   - docker build -t mcp-server mcp-server/
   - docker build -t mcp-dashboard mcp-dashboard/
   - docker build -t jupyter-user user-container/
         ↓
4. Integration Testing
   - docker compose -f ../deploy/docker-compose-full.yaml up
         ↓
5. Deployment
```

## Performance Characteristics

### Expected Response Times (with NIMs running)
- Job Creation: < 100ms
- Job Status Query: < 50ms
- AlphaFold2 Prediction: 30-300 seconds
- RFDiffusion Generation: 10-60 seconds per design
- ProteinMPNN Sequence: 5-20 seconds per design
- AlphaFold2-Multimer: 60-600 seconds per complex

### Resource Requirements
- MCP Server: 2 CPU cores, 2GB RAM
- Dashboard: 2 CPU cores, 2GB RAM
- Jupyter: 2 CPU cores, 4GB RAM
- Each NIM: 1 GPU, 24+ CPU cores, 32+ GB RAM

## Error Handling

### MCP Server
- Graceful fallback to mock data when NIMs unavailable
- Detailed error messages in job progress
- Health check endpoints for monitoring

### Dashboard
- Automatic retry on network errors
- User-friendly error messages
- Service status indicators

### Job Processing
- Per-step error tracking
- Partial results on step failure
- Job state persistence
