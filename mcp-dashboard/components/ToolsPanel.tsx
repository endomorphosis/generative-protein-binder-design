'use client'

import { useEffect, useMemo, useState } from 'react'

type JsonSchema = {
  type?: string
  description?: string
  default?: any
  items?: any
}

type Tool = {
  name: string
  description?: string
  inputSchema?: {
    type?: string
    properties?: Record<string, JsonSchema>
    required?: string[]
  }
}

type CallToolResult = {
  content?: Array<{ type: string; text?: string }>
  isError?: boolean
}

function safeJsonParse(text: string): any {
  try {
    return JSON.parse(text)
  } catch {
    return null
  }
}

function schemaDefault(schema?: JsonSchema): any {
  if (schema && Object.prototype.hasOwnProperty.call(schema, 'default')) return schema.default
  if (schema?.type === 'integer' || schema?.type === 'number') return 0
  if (schema?.type === 'array') return []
  if (schema?.type === 'object') return {}
  return ''
}

function safeIdSuffix(input: string) {
  return input.toLowerCase().replace(/[^a-z0-9_-]+/g, '_')
}

export default function ToolsPanel() {
  const [tools, setTools] = useState<Tool[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedToolName, setSelectedToolName] = useState<string>('')
  const [args, setArgs] = useState<Record<string, any>>({})
  const [rawMode, setRawMode] = useState(false)
  const [rawArgsText, setRawArgsText] = useState<string>('{}')
  const [running, setRunning] = useState(false)
  const [resultText, setResultText] = useState<string>('')
  const [resultObj, setResultObj] = useState<any>(null)
  const [resultRaw, setResultRaw] = useState<string>('')
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      setError(null)
      try {
        const res = await fetch('/api/mcp/tools', { cache: 'no-store' })
        const data = await res.json()
        const list: Tool[] = data?.tools || []
        setTools(list)
        if (!selectedToolName && list.length > 0) {
          setSelectedToolName(list[0].name)
        }
      } catch (e: any) {
        setError(e?.message || 'Failed to load tools')
      } finally {
        setLoading(false)
      }
    }
    load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const selectedTool = useMemo(
    () => tools.find((t) => t.name === selectedToolName),
    [tools, selectedToolName]
  )

  useEffect(() => {
    // Reset args when tool selection changes
    setResultText('')
    setResultObj(null)
    setResultRaw('')
    setError(null)

    const props = selectedTool?.inputSchema?.properties || {}
    const nextArgs: Record<string, any> = {}
    for (const [key, schema] of Object.entries(props)) {
      nextArgs[key] = schemaDefault(schema)
    }
    setArgs(nextArgs)
    setRawArgsText(JSON.stringify(nextArgs, null, 2))
  }, [selectedToolName, selectedTool?.inputSchema?.properties])

  const handleArgChange = (key: string, value: any) => {
    setArgs((prev) => {
      const next = { ...prev, [key]: value }
      setRawArgsText(JSON.stringify(next, null, 2))
      return next
    })
  }

  const callTool = async (override?: { name?: string; args?: Record<string, any> }) => {
    setRunning(true)
    setError(null)
    setResultText('')

    try {
      const nameToCall = override?.name ?? selectedToolName
      const argsToCall = override?.args ?? (rawMode ? safeJsonParse(rawArgsText) : args)

      const bodyArgs = argsToCall
      if (rawMode && bodyArgs === null) {
        throw new Error('Arguments JSON is invalid')
      }

      const res = await fetch('/api/mcp/tools/call', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: nameToCall, arguments: bodyArgs || {} }),
      })

      const payload = await res.json()
      if (!res.ok) {
        throw new Error(payload?.error || `HTTP ${res.status}`)
      }

      const result: CallToolResult = payload
      const text = result?.content?.find((c) => c.type === 'text')?.text
      const parsed = text ? safeJsonParse(text) : null

      if (result?.isError) {
        setError(text || 'Tool returned an error')
        setResultText(text || '')
        setResultRaw(text || '')
        setResultObj(parsed)
      } else {
        setResultRaw(text || '')
        setResultObj(parsed)
        setResultText(parsed ? JSON.stringify(parsed, null, 2) : (text || JSON.stringify(payload, null, 2)))
      }
    } catch (e: any) {
      setError(e?.message || 'Tool call failed')
    } finally {
      setRunning(false)
    }
  }

  const runQuick = async (toolName: string, arguments_: Record<string, any>) => {
    setSelectedToolName(toolName)
    setRawMode(false)
    setArgs(arguments_)
    setRawArgsText(JSON.stringify(arguments_, null, 2))
    await callTool({ name: toolName, args: arguments_ })
  }

  const statusPill = (status?: string) => {
    const s = (status || '').toLowerCase()
    if (s === 'ready' || s === 'completed') {
      return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300'
    }
    if (s === 'running') {
      return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300'
    }
    if (s.includes('error') || s === 'failed' || s === 'not_ready') {
      return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300'
    }
    return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300'
  }

  const renderPrettyResult = () => {
    if (!resultObj) return null

    if (selectedToolName === 'check_services' && typeof resultObj === 'object') {
      const entries = Object.entries(resultObj as Record<string, any>)
      return (
        <div className="space-y-2">
          <div className="text-sm font-medium text-gray-900 dark:text-white">Service Status</div>
          <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 dark:bg-gray-900/40">
                <tr>
                  <th className="text-left px-3 py-2 text-gray-700 dark:text-gray-300">Service</th>
                  <th className="text-left px-3 py-2 text-gray-700 dark:text-gray-300">Status</th>
                  <th className="text-left px-3 py-2 text-gray-700 dark:text-gray-300">URL</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {entries.map(([name, info]) => (
                  <tr key={name} className="bg-white dark:bg-gray-800">
                    <td className="px-3 py-2 font-medium text-gray-900 dark:text-white">{name}</td>
                    <td className="px-3 py-2">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${statusPill(info?.status)}`}>
                        {info?.status || 'unknown'}
                      </span>
                      {info?.error && (
                        <div className="text-xs text-red-700 dark:text-red-300 mt-1 break-words">
                          {String(info.error)}
                        </div>
                      )}
                    </td>
                    <td className="px-3 py-2 text-xs text-gray-600 dark:text-gray-400 break-all">{info?.url}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )
    }

    if ((selectedToolName === 'list_jobs' || selectedToolName === 'get_job_status') && (Array.isArray(resultObj) || typeof resultObj === 'object')) {
      const jobs = Array.isArray(resultObj) ? resultObj : [resultObj]
      return (
        <div className="space-y-2">
          <div className="text-sm font-medium text-gray-900 dark:text-white">Jobs ({jobs.length})</div>
          <div className="space-y-2">
            {jobs.map((j: any) => (
              <div key={j?.job_id || Math.random()} className="border border-gray-200 dark:border-gray-700 rounded-lg p-3 bg-white dark:bg-gray-800">
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="font-medium text-gray-900 dark:text-white truncate">{j?.job_name || j?.job_id}</div>
                    <div className="text-xs text-gray-500 dark:text-gray-400 break-all">{j?.job_id}</div>
                  </div>
                  <span className={`px-2 py-1 rounded text-xs font-medium ${statusPill(j?.status)}`}>{j?.status || 'unknown'}</span>
                </div>
                {j?.progress && (
                  <div className="mt-2 grid grid-cols-2 gap-1 text-xs text-gray-600 dark:text-gray-400">
                    {Object.entries(j.progress).map(([k, v]) => (
                      <div key={k} className="flex justify-between gap-2">
                        <span className="capitalize">{String(k).replace('_', ' ')}:</span>
                        <span className="font-medium">{String(v)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )
    }

    if (selectedToolName === 'design_protein_binder' && typeof resultObj === 'object') {
      return (
        <div className="space-y-2">
          <div className="text-sm font-medium text-gray-900 dark:text-white">Job Created</div>
          <div className="border border-gray-200 dark:border-gray-700 rounded-lg p-3 bg-white dark:bg-gray-800">
            <div className="text-sm text-gray-900 dark:text-white">{resultObj?.job_name || resultObj?.job_id}</div>
            <div className="text-xs text-gray-500 dark:text-gray-400 break-all">{resultObj?.job_id}</div>
            {resultObj?.status && (
              <div className="mt-2">
                <span className={`px-2 py-1 rounded text-xs font-medium ${statusPill(resultObj.status)}`}>{resultObj.status}</span>
              </div>
            )}
          </div>
        </div>
      )
    }

    return null
  }

  if (loading) {
    return (
      <div className="flex justify-center items-center py-6">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap gap-2">
        <button
          type="button"
          onClick={() => runQuick('check_services', {})}
          disabled={running}
          className="text-xs bg-gray-100 hover:bg-gray-200 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-900 dark:text-gray-100 px-3 py-2 rounded"
        >
          Check Services
        </button>
        <button
          type="button"
          onClick={() => runQuick('list_jobs', {})}
          disabled={running}
          className="text-xs bg-gray-100 hover:bg-gray-200 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-900 dark:text-gray-100 px-3 py-2 rounded"
        >
          List Jobs
        </button>
        <button
          type="button"
          onClick={async () => {
            setLoading(true)
            setError(null)
            try {
              const res = await fetch('/api/mcp/tools', { cache: 'no-store' })
              const data = await res.json()
              setTools(data?.tools || [])
            } catch (e: any) {
              setError(e?.message || 'Failed to refresh tools')
            } finally {
              setLoading(false)
            }
          }}
          disabled={running}
          className="text-xs bg-gray-100 hover:bg-gray-200 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-900 dark:text-gray-100 px-3 py-2 rounded"
        >
          Refresh Tools
        </button>
      </div>

      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md p-3">
          <p className="text-sm text-red-800 dark:text-red-400">{error}</p>
        </div>
      )}

      <div>
        <label
          htmlFor="mcp-tool-select"
          className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
        >
          Tool
        </label>
        <select
          id="mcp-tool-select"
          value={selectedToolName}
          onChange={(e) => setSelectedToolName(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
        >
          {tools.map((t) => (
            <option key={t.name} value={t.name}>
              {t.name}
            </option>
          ))}
        </select>
        {selectedTool?.description && (
          <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">{selectedTool.description}</p>
        )}
      </div>

      <div className="flex items-center justify-between">
        <label htmlFor={rawMode ? 'mcp-raw-args' : undefined} className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Arguments
        </label>
        <label htmlFor="mcp-raw-mode" className="flex items-center gap-2 text-xs text-gray-600 dark:text-gray-400">
          <input
            id="mcp-raw-mode"
            type="checkbox"
            checked={rawMode}
            onChange={(e) => setRawMode(e.target.checked)}
          />
          Raw JSON
        </label>
      </div>

      {!rawMode && (
        <div className="space-y-3">
          {Object.entries(selectedTool?.inputSchema?.properties || {}).length === 0 && (
            <div className="text-xs text-gray-500 dark:text-gray-400">No arguments</div>
          )}

          {Object.entries(selectedTool?.inputSchema?.properties || {}).map(([key, schema]) => {
            const schemaType = schema?.type || 'string'
            const required = (selectedTool?.inputSchema?.required || []).includes(key)
            const fieldId = `mcp-arg-${safeIdSuffix(key)}`

            if (schemaType === 'integer' || schemaType === 'number') {
              return (
                <div key={key}>
                  <label htmlFor={fieldId} className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    {key}{required ? ' *' : ''}
                  </label>
                  <input
                    id={fieldId}
                    type="number"
                    value={args[key] ?? ''}
                    onChange={(e) => handleArgChange(key, e.target.value === '' ? '' : Number(e.target.value))}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                  />
                  {schema?.description && (
                    <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">{schema.description}</p>
                  )}
                </div>
              )
            }

            if (schemaType === 'array') {
              return (
                <div key={key}>
                  <label htmlFor={fieldId} className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                    {key}{required ? ' *' : ''} (JSON array)
                  </label>
                  <textarea
                    id={fieldId}
                    rows={3}
                    value={JSON.stringify(args[key] ?? [], null, 2)}
                    onChange={(e) => {
                      const parsed = safeJsonParse(e.target.value)
                      handleArgChange(key, parsed ?? e.target.value)
                    }}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white font-mono text-xs"
                  />
                  {schema?.description && (
                    <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">{schema.description}</p>
                  )}
                </div>
              )
            }

            return (
              <div key={key}>
                <label htmlFor={fieldId} className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                  {key}{required ? ' *' : ''}
                </label>
                <textarea
                  id={fieldId}
                  rows={schema?.description?.toLowerCase().includes('pdb') ? 6 : 2}
                  value={args[key] ?? ''}
                  onChange={(e) => handleArgChange(key, e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white font-mono text-xs"
                />
                {schema?.description && (
                  <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">{schema.description}</p>
                )}
              </div>
            )
          })}
        </div>
      )}

      {rawMode && (
        <textarea
          id="mcp-raw-args"
          rows={10}
          value={rawArgsText}
          onChange={(e) => setRawArgsText(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white font-mono text-xs"
        />
      )}

      <button
        onClick={() => void callTool()}
        disabled={running || !selectedToolName}
        className="w-full bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-400 text-white font-medium py-2 px-4 rounded-md transition-colors duration-200"
      >
        {running ? 'Running...' : 'Run Tool'}
      </button>

      {renderPrettyResult()}

      {(resultText || resultRaw) && (
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">Raw Result</label>
          <pre className="bg-gray-900 dark:bg-gray-950 rounded-md p-3 font-mono text-xs text-green-400 overflow-auto max-h-64">
            {resultText || resultRaw}
          </pre>
        </div>
      )}
    </div>
  )
}
