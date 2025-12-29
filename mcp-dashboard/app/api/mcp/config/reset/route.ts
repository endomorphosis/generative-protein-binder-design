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

export async function POST() {
  if (isMockMode()) {
    const result = handleMockToolCall('reset_runtime_config', {})
    const text = (result as any)?.content?.find((c: any) => c?.type === 'text')?.text
    const parsed = text ? tryParseJson(String(text)) : null
    return NextResponse.json(parsed ?? {})
  }

  const result = await mcpCallTool('reset_runtime_config', {})
  const text = extractFirstTextContent(result)
  const parsed = text ? tryParseJson(text) : null
  return NextResponse.json(parsed ?? {})
}
