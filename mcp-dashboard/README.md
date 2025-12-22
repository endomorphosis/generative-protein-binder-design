# MCP Dashboard

Web-based dashboard for the Protein Binder Design MCP Server.

## Quick Start

### Using Docker
```bash
docker build -t mcp-dashboard .
docker run -p 3000:3000 -e NEXT_PUBLIC_MCP_SERVER_URL=http://localhost:8000 mcp-dashboard
```

### Local Development
```bash
npm install
npm run dev
```

The dashboard will be available at http://localhost:3000

## Features

- Submit new protein design jobs
- Monitor job progress in real-time
- View and download results
- Check service health status
- Launch Jupyter notebooks

## Environment Variables

- `NEXT_PUBLIC_MCP_SERVER_URL` - MCP Server endpoint
	- Local dev MCP server default: `http://localhost:8000`
	- Full stack (via `./scripts/run_dashboard_stack.sh`) default: `http://localhost:${MCP_SERVER_HOST_PORT:-8011}`

## Technology Stack

- Next.js 14
- React 18
- TypeScript
- Tailwind CSS
- Axios

## Building for Production

```bash
npm run build
npm start
```

## E2E Tests (Playwright)

```bash
npm run test:e2e
```

By default the test runner starts the dashboard dev server on port `3100` to avoid collisions with any Docker container already using `3000`.

- Override port: `E2E_PORT=3005 npm run test:e2e`
- Reuse an existing server instead of starting one: `E2E_REUSE_SERVER=1 npm run test:e2e`

To open the HTML report:

```bash
npx playwright show-report
```

See [DOCKER_MCP_README.md](../docs/DOCKER_MCP_README.md) for more information.
