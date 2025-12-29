import { test, expect } from '@playwright/test'
import { runPredictStructureViaToolsPanel, shouldRunRealIntegration, waitForServicesReady } from './helpers/real-integration'

test.describe.configure({ mode: 'serial' })

test.describe('Real integration: AlphaFold inference via MCP dashboard', () => {
  test('predict_structure returns a PDB (real models)', async ({ page, request, baseURL }) => {
    test.skip(!shouldRunRealIntegration(), 'Set MCP_INTEGRATION_REAL=1 to run real-model integration tests')

    // Guard against accidentally running against dashboard-local mock mode.
    const mock = (process.env.MCP_DASHBOARD_MOCK || '').trim().toLowerCase()
    test.skip(mock === '1' || mock === 'true', 'Disable dashboard mock mode (set MCP_DASHBOARD_MOCK=0) to run real-model tests')

    test.setTimeout(35 * 60_000)

    const base = baseURL || 'http://127.0.0.1:3100'

    // Ensure the real AlphaFold backend is up (via the dashboard proxy path).
    await waitForServicesReady(request, base, { timeoutMs: 20 * 60_000, pollMs: 15_000, requireAlphaFoldReady: true })

    await page.goto('/')

    // A small-ish stable protein sequence (ubiquitin, 76 aa) so the test is deterministic.
    const sequence = 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG'

    const { rawText } = await runPredictStructureViaToolsPanel(page, sequence)

    // Sanity assertions beyond the helper's ATOM check.
    expect(rawText).toMatch(/\bATOM\s+\d+\b/)
    expect(rawText).toMatch(/\bALA\b|\bGLY\b|\bLYS\b|\bLEU\b|\bVAL\b/)
  })
})
