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
  try {
    const result = await mcpCallTool('check_services', {})
    const text = extractFirstTextContent(result)
    const parsed = text ? tryParseJson(text) : null
    if (parsed === null) {
      return NextResponse.json(
        { error: 'MCP returned a non-JSON services status payload', raw: text ?? '' },
        { status: 502 }
      )
    }
    return NextResponse.json(parsed)
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    return NextResponse.json(
      { error: 'Failed to fetch services status from MCP', detail: message },
      { status: 502 }
    )
  }
}
