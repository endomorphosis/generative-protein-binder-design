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

export async function GET() {
  const result = await mcpCallTool('get_runtime_config', {})
  const text = extractFirstTextContent(result)
  const parsed = text ? tryParseJson(text) : null
  return NextResponse.json(parsed ?? {})
}

export async function PUT(request: Request) {
  const payloadText = await request.text()
  const payload = payloadText ? tryParseJson(payloadText) : null
  const result = await mcpCallTool('update_runtime_config', { patch: payload ?? {} })
  const text = extractFirstTextContent(result)
  const parsed = text ? tryParseJson(text) : null
  return NextResponse.json(parsed ?? {})
}
