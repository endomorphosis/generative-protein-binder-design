import { NextResponse } from 'next/server'
import { mcpListResources } from '@/lib/mcp-sdk-client'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const fetchCache = 'force-no-store'

export async function GET() {
  const result = await mcpListResources()
  return NextResponse.json(result)
}
