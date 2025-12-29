import { extractFirstTextContent, withMcpClient } from '@generative-protein/mcp-js-sdk'

function getMcpBaseUrl(): string {
  return (
    process.env.MCP_SERVER_URL ||
    process.env.NEXT_PUBLIC_MCP_SERVER_URL ||
    'http://localhost:8010'
  ).replace(/\/$/, '')
}

export async function mcpListTools() {
  return withMcpClient({ baseUrl: getMcpBaseUrl(), clientName: 'mcp-dashboard', clientVersion: '1.0.0' }, (client) =>
    client.listTools()
  )
}

export async function mcpListResources() {
  return withMcpClient({ baseUrl: getMcpBaseUrl(), clientName: 'mcp-dashboard', clientVersion: '1.0.0' }, (client) =>
    client.listResources()
  )
}

export async function mcpReadResource(uri: string) {
  return withMcpClient({ baseUrl: getMcpBaseUrl(), clientName: 'mcp-dashboard', clientVersion: '1.0.0' }, (client) =>
    client.readResource({ uri })
  )
}

export async function mcpCallTool(name: string, arguments_: Record<string, any>) {
  return withMcpClient({ baseUrl: getMcpBaseUrl(), clientName: 'mcp-dashboard', clientVersion: '1.0.0' }, (client) =>
    client.callTool({ name, arguments: arguments_ })
  )
}

// AlphaFold optimization settings functions
export async function getAlphaFoldSettings() {
  return mcpCallTool('get_alphafold_settings', {})
}

export async function updateAlphaFoldSettings(settings: Record<string, any>) {
  return mcpCallTool('update_alphafold_settings', settings)
}

export async function resetAlphaFoldSettings() {
  return mcpCallTool('reset_alphafold_settings', {})
}

export { extractFirstTextContent }

