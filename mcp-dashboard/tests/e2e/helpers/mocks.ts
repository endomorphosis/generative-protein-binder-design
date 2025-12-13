import type { Page, Route } from '@playwright/test'

export function installMockEventSource(page: Page) {
  return page.addInitScript(() => {
    class MockEventSource {
      url: string
      onmessage: ((event: { data: string }) => void) | null = null
      onerror: ((event: any) => void) | null = null
      constructor(url: string) {
        this.url = url
        // Send a couple of events to simulate a live connection.
        setTimeout(() => this.onmessage?.({ data: 'ready' }), 10)
        setTimeout(() => this.onmessage?.({ data: 'ping' }), 30)
      }
      close() {
        // no-op
      }
    }

    // @ts-expect-error override for test environment
    window.EventSource = MockEventSource
  })
}

export async function jsonRoute(route: Route, payload: any, status = 200) {
  await route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(payload),
  })
}

export const examplePdb = `ATOM      1  N   ALA A   1      11.104  13.207   2.100  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.400   2.100  1.00 20.00           C
ATOM      3  C   ALA A   1      13.040  14.840   2.100  1.00 20.00           C
ATOM      4  O   ALA A   1      12.370  15.800   2.100  1.00 20.00           O
ATOM      5  N   GLY A   2      14.330  14.980   2.100  1.00 20.00           N
ATOM      6  CA  GLY A   2      14.930  16.320   2.100  1.00 20.00           C
TER
END
`
