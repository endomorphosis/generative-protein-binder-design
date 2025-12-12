import { NextResponse } from 'next/server'
import { mcpListResources } from '@/lib/mcp-sdk-client'

export const runtime = 'nodejs'

export async function GET() {
  const result = await mcpListResources()
  return NextResponse.json(result)
}
