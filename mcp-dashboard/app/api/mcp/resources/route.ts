import { NextResponse } from 'next/server'
import { mcpListResources } from '@/lib/mcp-sdk-client'
import { isMockMode, mockListResources } from '@/lib/mock'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const fetchCache = 'force-no-store'

export async function GET() {
  if (isMockMode()) {
    return NextResponse.json(mockListResources())
  }

  const result = await mcpListResources()
  return NextResponse.json(result)
}
