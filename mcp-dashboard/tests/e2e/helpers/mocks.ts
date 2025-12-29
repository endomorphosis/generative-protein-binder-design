import type { Page, Route } from '@playwright/test'

export function installMockEventSource(page: Page) {
  return page.addInitScript(() => {
    class MockEventSource {
      url: string
      onmessage: ((event: { data: string }) => void) | null = null
      onerror: ((event: any) => void) | null = null
      constructor(url: string) {
        this.url = url
        // Send a couple of events to simulate a live connection.
        setTimeout(() => this.onmessage?.({ data: 'ready' }), 10)
        setTimeout(() => this.onmessage?.({ data: 'ping' }), 30)
      }
      close() {
        // no-op
      }
    }

    // @ts-expect-error override for test environment
    window.EventSource = MockEventSource
  })
}

export async function jsonRoute(route: Route, payload: any, status = 200) {
  await route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(payload),
  })
}

type ToolHandler =
  | any
  | ((args: Record<string, any>) => any)

/**
 * Mock the MCP server JSON-RPC endpoint (POST /mcp).
 *
 * This is used by the dashboard's MCP SDK wrapper when calling tools directly.
 * It implements a minimal `initialize` handshake plus `tools/call`.
 */
export async function installMockMcpJsonRpc(
  page: Page,
  tools: Record<string, ToolHandler>
) {
  await page.route(/\/mcp$/, async (route) => {
    const req = route.request()

    // Avoid catching dashboard routes like /api/mcp/*
    if (req.url().includes('/api/mcp/')) {
      await route.fallback()
      return
    }

    const method = req.method()
    if (method !== 'POST') {
      await route.fulfill({ status: 405, contentType: 'application/json', body: JSON.stringify({ error: 'method' }) })
      return
    }

    const body = req.postDataJSON() as any
    const id = body?.id
    const rpcMethod = body?.method

    // JSON-RPC notification: no id expected
    if (!id) {
      await route.fulfill({ status: 204, body: '' })
      return
    }

    if (rpcMethod === 'initialize') {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          jsonrpc: '2.0',
          id,
          result: {
            protocolVersion: '2024-11-05',
            capabilities: {},
            serverInfo: { name: 'mock-mcp', version: '0.0.0' },
          },
        }),
      })
      return
    }

    if (rpcMethod === 'tools/call') {
      const toolName = body?.params?.name
      const args = (body?.params?.arguments ?? {}) as Record<string, any>
      const handler = toolName ? tools[toolName] : undefined
      const payload =
        typeof handler === 'function'
          ? (handler as (a: Record<string, any>) => any)(args)
          : handler

      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          jsonrpc: '2.0',
          id,
          result: {
            content: [{ type: 'text', text: JSON.stringify(payload ?? {}) }],
            isError: false,
          },
        }),
      })
      return
    }

    // Default: respond with an empty success result.
    await route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ jsonrpc: '2.0', id, result: {} }),
    })
  })
}

export const examplePdb = `ATOM      1  N   ALA A   1      11.104  13.207   2.100  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.400   2.100  1.00 20.00           C
ATOM      3  C   ALA A   1      13.040  14.840   2.100  1.00 20.00           C
ATOM      4  O   ALA A   1      12.370  15.800   2.100  1.00 20.00           O
ATOM      5  N   GLY A   2      14.330  14.980   2.100  1.00 20.00           N
ATOM      6  CA  GLY A   2      14.930  16.320   2.100  1.00 20.00           C
TER
END
`
