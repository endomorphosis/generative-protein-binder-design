import { NextResponse } from 'next/server'
import { extractFirstTextContent, mcpCallTool } from '@/lib/mcp-sdk-client'

export const runtime = 'nodejs'

function tryParseJson(text: string): any {
  try {
    return JSON.parse(text)
  } catch {
    return null
  }
}

export async function GET() {
  const result = await mcpCallTool('list_jobs', {})
  const text = extractFirstTextContent(result)
  const parsed = text ? tryParseJson(text) : null
  return NextResponse.json(parsed ?? [])
}

export async function POST(req: Request) {
  const body = await req.json().catch(() => ({}))

  const result = await mcpCallTool('design_protein_binder', body)
  const text = extractFirstTextContent(result)
  const parsed = text ? tryParseJson(text) : null

  if (!parsed) {
    return NextResponse.json(
      { error: 'Unexpected tool response', raw: text || result },
      { status: 502 }
    )
  }

  return NextResponse.json(parsed)
}

export async function DELETE(req: Request) {
  const body = await req.json().catch(() => ({}))
  const jobId = body?.job_id

  if (!jobId || typeof jobId !== 'string') {
    return NextResponse.json({ error: 'Missing job_id' }, { status: 400 })
  }

  const result = await mcpCallTool('delete_job', { job_id: jobId })
  const text = extractFirstTextContent(result)
  const parsed = text ? tryParseJson(text) : null

  return NextResponse.json(parsed ?? { deleted: jobId })
}
