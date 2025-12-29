#!/usr/bin/env node

import { McpProteinDesignClient } from '@generative-protein/mcp-js-sdk'

function usage() {
  const msg = `
Usage: mcp-binder <command> [args]

Commands:
  env                          Run environment checks (tools + services + config)
  jobs                         List jobs
  job <job_id>                 Get job status
  delete <job_id>              Delete a job
  design --sequence <SEQ> [--job-name <NAME>] [--num-designs <N>]
                               Submit a binder design job

Environment:
  MCP_SERVER_URL               MCP server base URL (default: http://localhost:8010)
`
  console.error(msg.trim())
}

function getArg(flag) {
  const idx = process.argv.indexOf(flag)
  if (idx === -1) return undefined
  return process.argv[idx + 1]
}

function hasFlag(flag) {
  return process.argv.includes(flag)
}

async function main() {
  const [, , cmd, ...rest] = process.argv
  if (!cmd || hasFlag('-h') || hasFlag('--help')) {
    usage()
    process.exit(cmd ? 0 : 1)
  }

  const client = new McpProteinDesignClient({
    baseUrl: (process.env.MCP_SERVER_URL || 'http://localhost:8010').replace(/\/$/, ''),
    clientName: 'mcp-binder-cli',
    clientVersion: '0.1.0',
  })

  try {
    if (cmd === 'env') {
      const tools = await client.listTools()
      const services = await client.checkServices()
      const config = await client.getRuntimeConfig()
      console.log(
        JSON.stringify(
          {
            tools,
            services: services.json ?? services.text,
            runtime_config: config.json ?? config.text,
          },
          null,
          2
        )
      )
      return
    }

    if (cmd === 'jobs') {
      const jobs = await client.listJobs()
      console.log(JSON.stringify(jobs.json ?? jobs.text ?? jobs.raw, null, 2))
      return
    }

    if (cmd === 'job') {
      const jobId = rest[0]
      if (!jobId) {
        usage()
        process.exit(2)
      }
      const job = await client.getJobStatus(jobId)
      console.log(JSON.stringify(job.json ?? job.text ?? job.raw, null, 2))
      return
    }

    if (cmd === 'delete') {
      const jobId = rest[0]
      if (!jobId) {
        usage()
        process.exit(2)
      }
      const out = await client.deleteJob(jobId)
      console.log(JSON.stringify(out.json ?? out.text ?? out.raw, null, 2))
      return
    }

    if (cmd === 'design') {
      const sequence = getArg('--sequence')
      const jobName = getArg('--job-name')
      const numDesignsRaw = getArg('--num-designs')
      const numDesigns = numDesignsRaw ? Number(numDesignsRaw) : undefined

      if (!sequence) {
        usage()
        process.exit(2)
      }

      const out = await client.designProteinBinder({
        sequence,
        job_name: jobName || undefined,
        num_designs: Number.isFinite(numDesigns) ? numDesigns : undefined,
      })
      console.log(JSON.stringify(out.json ?? out.text ?? out.raw, null, 2))
      return
    }

    console.error(`Unknown command: ${cmd}`)
    usage()
    process.exit(2)
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err)
    console.error(message)
    process.exit(1)
  }
}

await main()
