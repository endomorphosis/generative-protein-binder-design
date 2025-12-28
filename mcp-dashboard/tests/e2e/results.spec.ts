import { test, expect } from '@playwright/test'
import { examplePdb, installMockEventSource, jsonRoute } from './helpers/mocks'

function makeCompletedJob() {
  return {
    job_id: 'job_completed_0',
    status: 'completed',
    created_at: new Date(Date.now() - 60_000).toISOString(),
    updated_at: new Date().toISOString(),
    job_name: 'Completed Job',
    progress: {
      alphafold: 'completed',
      rfdiffusion: 'completed',
      proteinmpnn: 'completed',
      alphafold_multimer: 'completed',
    },
    results: {
      target_structure: { pdb: examplePdb },
      designs: [
        {
          design_id: 0,
          backbone: { pdb: examplePdb },
          sequence: { sequence: 'ACDEFGHIKLMNPQRSTVWY' },
          complex_structure: { pdb: examplePdb },
        },
      ],
    },
    error: null,
  }
}

test.describe('Results viewer', () => {
  test.beforeEach(async ({ page }) => {
    await installMockEventSource(page)
  })

  test('shows completed results + allows download', async ({ page }) => {
    const job = makeCompletedJob()

    await page.route('**/api/mcp/services/status', async (route) => {
      await jsonRoute(route, { alphafold: { status: 'ready', url: 'x' } })
    })

    await page.route('**/api/mcp/jobs', async (route) => {
      if (route.request().method() === 'GET') {
        await jsonRoute(route, [job])
        return
      }
      await route.fallback()
    })

    await page.goto('/')

    await page.getByText('Completed Job').click()

    await expect(page.getByText('✓ Completed')).toBeVisible()
    await expect(page.getByText('Target Structure')).toBeVisible()
    await expect(page.getByText(/Binder Designs/i)).toBeVisible()

    // Design 1 starts expanded by default
    await expect(page.getByText(/Binder Sequence/i)).toBeVisible()

    // Download All Results
    const downloadPromise = page.waitForEvent('download')
    await page.getByRole('button', { name: /Download All Results/i }).click()
    const download = await downloadPromise
    expect(download.suggestedFilename()).toMatch(/job_completed_0_results\.json$/)
  })
})
