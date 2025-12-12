import { NextResponse } from 'next/server'
import { mcpListTools } from '@/lib/mcp-sdk-client'

export const runtime = 'nodejs'

export async function GET() {
  const result = await mcpListTools()
  return NextResponse.json(result)
}
