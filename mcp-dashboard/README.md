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

- `NEXT_PUBLIC_MCP_SERVER_URL` - MCP Server endpoint (default: http://localhost:8000)

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

See [DOCKER_MCP_README.md](../docs/DOCKER_MCP_README.md) for more information.
