import { NextResponse } from 'next/server'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const fetchCache = 'force-no-store'

function getMcpBaseUrl() {
  return (
    process.env.MCP_SERVER_URL ||
    process.env.NEXT_PUBLIC_MCP_SERVER_URL ||
    'http://localhost:8000'
  ).replace(/\/$/, '')
}

export async function POST() {
  const res = await fetch(`${getMcpBaseUrl()}/api/config/reset`, {
    method: 'POST',
    cache: 'no-store',
  })
  const body = await res.text()
  return new NextResponse(body, {
    status: res.status,
    headers: { 'Content-Type': 'application/json' },
  })
}
