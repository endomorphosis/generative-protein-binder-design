import { NextResponse } from 'next/server'
import { extractFirstTextContent, mcpCallTool } from '@/lib/mcp-sdk-client'

export const runtime = 'nodejs'
export const dynamic = 'force-dynamic'
export const fetchCache = 'force-no-store'

function tryParseJson(text: string): any {
  try {
    return JSON.parse(text)
  } catch {
    return null
  }
}

export async function POST(request: Request) {
  const payloadText = await request.text()
  const payload = payloadText ? tryParseJson(payloadText) : null
  const models = Array.isArray(payload?.models) ? payload.models : []
  const result = await mcpCallTool('embedded_bootstrap', { models })
  const text = extractFirstTextContent(result)
  const parsed = text ? tryParseJson(text) : null
  return NextResponse.json(parsed ?? {})
}
