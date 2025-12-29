import { test, expect } from '@playwright/test'
import { examplePdb, installMockEventSource, jsonRoute } from './helpers/mocks'

function makeCompletedJob() {
  return {
    job_id: 'job_completed_0',
    status: 'completed',
    created_at: new Date(Date.now() - 60_000).toISOString(),
    updated_at: new Date().toISOString(),
    job_name: 'Completed Job',
    input: { sequence: 'ACDEFGHIKLMNPQRSTVWY', num_designs: 5 },
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

  test('iterate from a completed job pre-fills the input form', async ({ page }) => {
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

    await page.getByRole('button', { name: 'Iterate From This Job' }).click()

    await expect(page.getByLabel(/Target Protein Sequence/i)).toHaveValue(job.input.sequence)
    await expect(page.getByLabel(/Number of Designs/i)).toHaveValue(String(job.input.num_designs))
  })

  test('3D viewer opens and closes', async ({ page }) => {
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

    await page.getByRole('button', { name: /View Target in 3D/i }).click()
    await expect(page.getByText('🔬 3D Protein Structure Viewer')).toBeVisible()

    await page.getByRole('button', { name: 'Close 3D Viewer' }).click()
    await expect(page.getByText('🔬 3D Protein Structure Viewer')).toBeHidden()
  })

  test('3D viewer can propose sequence variants (mock tool)', async ({ page }) => {
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

    // Open viewer from a design so the sequence is available.
    await page.getByRole('button', { name: /View 3D/i }).click()
    await expect(page.getByText('🔬 3D Protein Structure Viewer')).toBeVisible()

    await expect(page.getByLabel('Variant positions')).toBeVisible()
    await page.getByLabel('Variant positions').fill('1,2,3')
    await page.getByTestId('propose-variants').click()

    await expect(page.getByText(/Proposed Variants/i)).toBeVisible()

    // Use the top variant to prefill the main form.
    const variantSequenceEl = page.getByTestId('variant-sequence-0')
    await expect(variantSequenceEl).toBeVisible()
    const variantSequence = (await variantSequenceEl.innerText()).trim()
    await page.getByTestId('iterate-variant-0').click()

    await expect(page.getByText('🔬 3D Protein Structure Viewer')).toBeHidden()
    await expect(page.getByLabel(/Target Protein Sequence/i)).toHaveValue(variantSequence)

    // (modal already closed by iterate button)
  })
})
