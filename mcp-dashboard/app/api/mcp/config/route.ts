import { NextResponse } from 'next/server'
import { extractFirstTextContent, mcpCallTool } from '@/lib/mcp-sdk-client'
import { handleMockToolCall, isMockMode } from '@/lib/mock'

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
  if (isMockMode()) {
    const result = handleMockToolCall('get_runtime_config', {})
    const text = (result as any)?.content?.find((c: any) => c?.type === 'text')?.text
    const parsed = text ? tryParseJson(String(text)) : null
    return NextResponse.json(parsed ?? {})
  }

  const result = await mcpCallTool('get_runtime_config', {})
  const text = extractFirstTextContent(result)
  const parsed = text ? tryParseJson(text) : null
  return NextResponse.json(parsed ?? {})
}

export async function PUT(request: Request) {
  const payloadText = await request.text()
  const payload = payloadText ? tryParseJson(payloadText) : null

  if (isMockMode()) {
    const result = handleMockToolCall('update_runtime_config', { patch: payload ?? {} })
    const text = (result as any)?.content?.find((c: any) => c?.type === 'text')?.text
    const parsed = text ? tryParseJson(String(text)) : null
    return NextResponse.json(parsed ?? {})
  }

  const result = await mcpCallTool('update_runtime_config', { patch: payload ?? {} })
  const text = extractFirstTextContent(result)
  const parsed = text ? tryParseJson(text) : null
  return NextResponse.json(parsed ?? {})
}
