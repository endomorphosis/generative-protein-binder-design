export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const revalidate = 0

function getMcpBaseUrl(): string {
  return (
    process.env.MCP_SERVER_URL ||
    process.env.NEXT_PUBLIC_MCP_SERVER_URL ||
    'http://localhost:8010'
  ).replace(/\/$/, '')
}

export async function GET(request: Request) {
  const upstream = await fetch(`${getMcpBaseUrl()}/sse`, {
    headers: { Accept: 'text/event-stream' },
    cache: 'no-store',
    signal: request.signal,
  })

  return new Response(upstream.body, {
    status: upstream.status,
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      Connection: 'keep-alive',
    },
  })
}
