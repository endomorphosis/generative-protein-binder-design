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
  ],
  webServer: {
    command: `npm run dev -- -p ${port}`,
    url: `http://127.0.0.1:${port}`,
    // Default to false so we don't accidentally point tests at a Docker container
    // (or any unrelated server) that happens to be listening on the same port.
    reuseExistingServer: process.env.E2E_REUSE_SERVER === '1',
    timeout: 120_000,
    env: {
      // Avoid Next.js telemetry in CI
      NEXT_TELEMETRY_DISABLED: '1',
      // Ensure the dashboard routes resolve locally
      MCP_SERVER_URL: process.env.MCP_SERVER_URL || 'http://127.0.0.1:8010',
    },
  },
})
