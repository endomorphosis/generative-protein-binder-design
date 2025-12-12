import { Client } from '@modelcontextprotocol/sdk/client/index.js'
import type { Transport } from '@modelcontextprotocol/sdk/shared/transport.js'
import type { JSONRPCMessage } from '@modelcontextprotocol/sdk/types.js'

function getMcpBaseUrl(): string {
  return (
    process.env.MCP_SERVER_URL ||
    process.env.NEXT_PUBLIC_MCP_SERVER_URL ||
    'http://localhost:8010'
  ).replace(/\/$/, '')
}

class HttpJsonRpcTransport implements Transport {
  onclose?: () => void
  onerror?: (error: Error) => void
  onmessage?: (message: JSONRPCMessage) => void

  private readonly endpoint: string
  private closed = false

  constructor(endpoint: string) {
    this.endpoint = endpoint
  }

  async start(): Promise<void> {
    // No-op for HTTP request/response transport.
  }

  async close(): Promise<void> {
    this.closed = true
    this.onclose?.()
  }

  async send(message: JSONRPCMessage): Promise<void> {
    if (this.closed) return

    try {
      const res = await fetch(this.endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(message),
        cache: 'no-store',
      })

      const text = await res.text()
      let payload: any
      try {
        payload = text ? JSON.parse(text) : null
      } catch {
        throw new Error(`Invalid JSON-RPC response: ${text.slice(0, 200)}`)
      }

      if (!res.ok) {
        const msg =
          payload?.error?.message ||
          payload?.detail ||
          `HTTP ${res.status} from MCP server`
        throw new Error(msg)
      }

      if (payload && this.onmessage) {
        this.onmessage(payload as JSONRPCMessage)
      }
    } catch (err: any) {
      const error = err instanceof Error ? err : new Error(String(err))
      this.onerror?.(error)
      throw error
    }
  }
}

async function withMcpClient<T>(fn: (client: Client) => Promise<T>): Promise<T> {
  const base = getMcpBaseUrl()
  const transport = new HttpJsonRpcTransport(`${base}/mcp`)
  const client = new Client(
    { name: 'mcp-dashboard', version: '1.0.0' },
    { capabilities: {} }
  )

  await client.connect(transport)
  try {
    return await fn(client)
  } finally {
    await transport.close()
  }
}

export async function mcpListTools() {
  return withMcpClient((client) => client.listTools())
}

export async function mcpListResources() {
  return withMcpClient((client) => client.listResources())
}

export async function mcpReadResource(uri: string) {
  return withMcpClient((client) => client.readResource({ uri }))
}

export async function mcpCallTool(name: string, arguments_: Record<string, any>) {
  return withMcpClient((client) => client.callTool({ name, arguments: arguments_ }))
}

export function extractFirstTextContent(result: any): string {
  const content = result?.content
  if (!Array.isArray(content) || content.length === 0) return ''
  const first = content[0]
  if (first?.type === 'text' && typeof first.text === 'string') return first.text
  return ''
}
