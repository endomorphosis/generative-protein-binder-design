import { NextResponse } from 'next/server'
import { extractFirstTextContent, mcpCallTool } from '@/lib/mcp-sdk-client'
import { getMockJob, isMockMode } from '@/lib/mock'

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

export async function POST(req: Request) {
  const body = await req.json().catch(() => ({}))
  const jobId = body?.job_id

  if (!jobId || typeof jobId !== 'string') {
    return NextResponse.json({ error: 'Missing job_id' }, { status: 400 })
  }

  if (isMockMode()) {
    const job = getMockJob(jobId)
    if (!job) {
      return NextResponse.json({ error: `Unknown job_id: ${jobId}` }, { status: 404 })
    }
    return NextResponse.json(job)
  }

  const result = await mcpCallTool('get_job_status', { job_id: jobId })
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
