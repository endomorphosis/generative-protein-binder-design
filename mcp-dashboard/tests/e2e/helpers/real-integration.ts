import type { APIRequestContext, Page } from '@playwright/test'

export function isTruthyEnv(value: string | undefined): boolean {
  if (!value) return false
  const v = value.trim().toLowerCase()
  return v === '1' || v === 'true' || v === 'yes' || v === 'on'
}

export function shouldRunRealIntegration(): boolean {
  return isTruthyEnv(process.env.MCP_INTEGRATION_REAL)
}

export type ServicesStatus = Record<
  string,
  { status?: string; url?: string; backend?: string; error?: string } | string
>

async function tryGetJson(request: APIRequestContext, url: string): Promise<any> {
  const res = await request.get(url, { timeout: 30_000 })
  const text = await res.text()
  try {
    return JSON.parse(text)
  } catch {
    return { __raw: text, __status: res.status() }
  }
}

export async function waitForServicesReady(
  request: APIRequestContext,
  baseURL: string,
  opts?: {
    timeoutMs?: number
    pollMs?: number
    requireAlphaFoldReady?: boolean
  }
): Promise<ServicesStatus> {
  const timeoutMs = opts?.timeoutMs ?? 10 * 60_000
  const pollMs = opts?.pollMs ?? 10_000
  const requireAlphaFoldReady = opts?.requireAlphaFoldReady ?? true

  const started = Date.now()
  let last: any = null

  // Poll the dashboard proxy endpoint so this validates the SDK/proxy path.
  const url = `${baseURL}/api/mcp/services/status`

  // eslint-disable-next-line no-constant-condition
  while (true) {
    last = await tryGetJson(request, url)

    const alphafold = (last?.alphafold ?? last?.alphaFold ?? last?.AlphaFold) as any
    const status = typeof alphafold === 'string' ? alphafold : String(alphafold?.status ?? '')
    const backend = typeof alphafold === 'object' ? String(alphafold?.backend ?? '') : ''
    const serviceUrl = typeof alphafold === 'object' ? String(alphafold?.url ?? '') : ''

    const looksMock = backend.toLowerCase() === 'mock' || serviceUrl.startsWith('mock://')
    const ready = status.toLowerCase() === 'ready'

    if (!requireAlphaFoldReady) {
      return last as ServicesStatus
    }

    if (ready && !looksMock) {
      return last as ServicesStatus
    }

    if (Date.now() - started > timeoutMs) {
      const short = JSON.stringify(last, null, 2).slice(0, 3000)
      throw new Error(
        `Timed out waiting for real AlphaFold service to be ready at ${url}. ` +
          `Last status=${status || 'unknown'} backend=${backend || 'unknown'} url=${serviceUrl || 'unknown'}\n` +
          short
      )
    }

    await new Promise((r) => setTimeout(r, pollMs))
  }
}

export function extractPdbLikeText(text: string): string | null {
  if (!text || typeof text !== 'string') return null
  // Typical PDB starts with HEADER/ATOM; AlphaFold outputs include ATOM lines.
  if (/\bATOM\s+\d+\b/.test(text)) return text
  return null
}

export async function runPredictStructureViaToolsPanel(page: Page, sequence: string) {
  // Tool select
  const toolSelect = page.locator('#mcp-tool-select')
  await toolSelect.waitFor({ state: 'visible' })

  // Wait until the tool is present (listTools can be slow on real backends).
  // Note: <option> elements are not considered "visible" by Playwright.
  await page.waitForFunction(
    () => !!document.querySelector('#mcp-tool-select option[value="predict_structure"]'),
    null,
    { timeout: 120_000 }
  )

  // selectOption can occasionally race with React state updates; retry briefly.
  const start = Date.now()
  // eslint-disable-next-line no-constant-condition
  while (true) {
    try {
      await toolSelect.selectOption('predict_structure')
      break
    } catch {
      if (Date.now() - start > 30_000) throw new Error('Failed to select predict_structure tool')
      await new Promise((r) => setTimeout(r, 250))
    }
  }

  // Fill argument
  const seqInput = page.locator('#mcp-arg-sequence')
  await seqInput.waitFor({ state: 'visible' })
  await seqInput.fill(sequence)

  // Run tool and wait for the POST to complete (real inference can take a while)
  const runBtn = page.getByRole('button', { name: 'Run Tool' })
  await runBtn.waitFor({ state: 'visible' })

  const callPromise = page.waitForResponse(
    (r) => r.url().includes('/api/mcp/tools/call') && r.request().method() === 'POST',
    { timeout: 30 * 60_000 }
  )

  await runBtn.click()
  await callPromise

  const rawPre = page.locator('label', { hasText: 'Raw Result' }).locator('..').locator('pre')
  await rawPre.waitFor({ state: 'visible', timeout: 30 * 60_000 })

  const rawText = await rawPre.innerText()
  const pdbLike = extractPdbLikeText(rawText)
  if (!pdbLike) {
    throw new Error(`Expected a PDB-like result containing ATOM records, got:\n${rawText.slice(0, 2000)}`)
  }

  return { rawText }
}
