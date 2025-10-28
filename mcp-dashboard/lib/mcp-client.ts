import axios, { AxiosInstance } from 'axios'
import { Job, ServiceStatus, ProteinSequenceInput } from './types'

const MCP_SERVER_URL = process.env.NEXT_PUBLIC_MCP_SERVER_URL || 'http://localhost:8001'

class MCPClient {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: MCP_SERVER_URL,
      headers: {
        'Content-Type': 'application/json',
      },
    })
  }

  // MCP Protocol methods
  async listTools() {
    const response = await this.client.get('/mcp/v1/tools')
    return response.data
  }

  async listResources() {
    const response = await this.client.get('/mcp/v1/resources')
    return response.data
  }

  async getResource(jobId: string) {
    const response = await this.client.get(`/mcp/v1/resources/${jobId}`)
    return response.data
  }

  // Job management methods
  async createJob(input: ProteinSequenceInput): Promise<Job> {
    const response = await this.client.post('/api/jobs', input)
    return response.data
  }

  async listJobs(): Promise<Job[]> {
    const response = await this.client.get('/api/jobs')
    return response.data
  }

  async getJob(jobId: string): Promise<Job> {
    const response = await this.client.get(`/api/jobs/${jobId}`)
    return response.data
  }

  async deleteJob(jobId: string): Promise<void> {
    await this.client.delete(`/api/jobs/${jobId}`)
  }

  async getServiceStatus(): Promise<ServiceStatus> {
    const response = await this.client.get('/api/services/status')
    return response.data
  }

  async healthCheck(): Promise<{ status: string }> {
    const response = await this.client.get('/health')
    return response.data
  }
}

export const mcpClient = new MCPClient()
