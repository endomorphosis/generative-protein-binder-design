import { defineConfig, devices } from '@playwright/test'

const port = Number(process.env.E2E_PORT || 3100)

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 60_000,
  expect: {
    timeout: 10_000,
  },
  fullyParallel: true,
  retries: process.env.CI ? 2 : 0,
  reporter: process.env.CI ? [['list'], ['html', { open: 'never' }]] : [['list'], ['html']],
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    // Gated real-model integration tests.
    // Run with:
    //   MCP_INTEGRATION_REAL=1 MCP_DASHBOARD_MOCK=0 MCP_SERVER_URL=http://127.0.0.1:${MCP_SERVER_HOST_PORT:-8011} \
    //   npx playwright test tests/e2e/integration.real.spec.ts
    {
      name: 'chromium-real',
      testMatch: /.*\.real\.spec\.ts/,
      timeout: 35 * 60_000,
      expect: {
        timeout: 60_000,
      },
      use: { ...devices['Desktop Chrome'] },
    },
  ],
  webServer: {
    // Next dev server can be quite noisy (especially in this repo's environment);
    // redirect its output so Playwright logs stay readable and don't get truncated.
    command: `bash -lc 'npm run dev -- -p ${port} > .playwright-next-dev.log 2>&1'`,
    url: `http://127.0.0.1:${port}`,
    // Default to false so we don't accidentally point tests at a Docker container
    // (or any unrelated server) that happens to be listening on the same port.
    reuseExistingServer: process.env.E2E_REUSE_SERVER === '1',
    timeout: 120_000,
    env: {
      // Avoid Next.js telemetry in CI
      NEXT_TELEMETRY_DISABLED: '1',
      // Ensure the dashboard never makes outbound MCP calls during E2E.
      MCP_DASHBOARD_MOCK: process.env.MCP_DASHBOARD_MOCK || '1',
      // Ensure the dashboard routes resolve locally
      MCP_SERVER_URL: process.env.MCP_SERVER_URL || 'http://127.0.0.1:8010',
      // Exposed to tests for gated real runs.
      MCP_INTEGRATION_REAL: process.env.MCP_INTEGRATION_REAL || '',
    },
  },
})
