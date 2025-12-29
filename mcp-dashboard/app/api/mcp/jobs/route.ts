import { NextResponse } from 'next/server'
import { extractFirstTextContent, mcpCallTool } from '@/lib/mcp-sdk-client'
import { createMockJob, deleteMockJob, isMockMode, listMockJobs } from '@/lib/mock'

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
    return NextResponse.json(listMockJobs())
  }

  const result = await mcpCallTool('list_jobs', {})
  const text = extractFirstTextContent(result)
  const parsed = text ? tryParseJson(text) : null
  return NextResponse.json(parsed ?? [])
}

export async function POST(req: Request) {
  const body = await req.json().catch(() => ({}))

  if (isMockMode()) {
    const sequence = body?.sequence
    if (!sequence || typeof sequence !== 'string') {
      return NextResponse.json({ error: 'Missing sequence' }, { status: 400 })
    }

    const job = createMockJob({
      sequence,
      job_name: typeof body?.job_name === 'string' ? body.job_name : undefined,
      num_designs: typeof body?.num_designs === 'number' ? body.num_designs : undefined,
    })
    return NextResponse.json(job)
  }

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

  if (isMockMode()) {
    deleteMockJob(jobId)
    return NextResponse.json({ deleted: jobId })
  }

  const result = await mcpCallTool('delete_job', { job_id: jobId })
  const text = extractFirstTextContent(result)
  const parsed = text ? tryParseJson(text) : null

  return NextResponse.json(parsed ?? { deleted: jobId })
}
