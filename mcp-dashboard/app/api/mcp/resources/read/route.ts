import { NextResponse } from 'next/server'
import { mcpReadResource } from '@/lib/mcp-sdk-client'
import { isMockMode, mockReadResource } from '@/lib/mock'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const fetchCache = 'force-no-store'

export async function POST(req: Request) {
  const body = await req.json().catch(() => ({}))
  const uri = body?.uri

  if (!uri || typeof uri !== 'string') {
    return NextResponse.json({ error: 'Missing uri' }, { status: 400 })
  }

  if (isMockMode()) {
    return NextResponse.json(mockReadResource(uri))
  }

  const result = await mcpReadResource(uri)
  return NextResponse.json(result)
}
