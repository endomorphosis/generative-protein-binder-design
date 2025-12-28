import { NextResponse } from 'next/server'
import { mcpCallTool } from '@/lib/mcp-sdk-client'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const fetchCache = 'force-no-store'

export async function POST(req: Request) {
  const body = await req.json().catch(() => ({}))
  const name = body?.name
  const args = body?.arguments || {}

  if (!name || typeof name !== 'string') {
    return NextResponse.json({ error: 'Missing tool name' }, { status: 400 })
  }

  const result = await mcpCallTool(name, args)
  return NextResponse.json(result)
}
