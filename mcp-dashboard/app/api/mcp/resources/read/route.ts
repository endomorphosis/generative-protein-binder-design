import { NextResponse } from 'next/server'
import { mcpReadResource } from '@/lib/mcp-sdk-client'

export const runtime = 'nodejs'

export async function POST(req: Request) {
  const body = await req.json().catch(() => ({}))
  const uri = body?.uri

  if (!uri || typeof uri !== 'string') {
    return NextResponse.json({ error: 'Missing uri' }, { status: 400 })
  }

  const result = await mcpReadResource(uri)
  return NextResponse.json(result)
}
