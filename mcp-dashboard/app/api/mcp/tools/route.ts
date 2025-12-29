import { NextResponse } from 'next/server'
import { mcpListTools } from '@/lib/mcp-sdk-client'
import { isMockMode, mockTools } from '@/lib/mock'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const fetchCache = 'force-no-store'

export async function GET() {
  if (isMockMode()) {
    return NextResponse.json({ tools: mockTools })
  }

  const result = await mcpListTools()
  return NextResponse.json(result)
}
