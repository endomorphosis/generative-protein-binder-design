'use client'

import { useState, useEffect } from 'react'
import { mcpClient } from '@/lib/mcp-client'
import { ServiceStatus as ServiceStatusType } from '@/lib/types'

export default function ServiceStatus() {
  const [status, setStatus] = useState<ServiceStatusType | null>(null)
  const [loading, setLoading] = useState(true)
  const [loadError, setLoadError] = useState<string | null>(null)

  useEffect(() => {
    loadStatus()
    const interval = setInterval(loadStatus, 10000) // Refresh every 10 seconds
    return () => clearInterval(interval)
  }, [])

  const loadStatus = async () => {
    try {
      const serviceStatus = await mcpClient.getServiceStatus()
      setStatus(serviceStatus)
      setLoadError(null)
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err)
      console.error('Failed to load service status:', message)
      setLoadError(message)
    } finally {
      setLoading(false)
    }
  }

  const buildWarnings = (serviceStatus: ServiceStatusType | null) => {
    if (!serviceStatus || typeof serviceStatus !== 'object') return []

    const warnings: Array<{ key: string; message: string; detail?: string }> = []
    for (const [service, info] of Object.entries(serviceStatus)) {
      if (!info || typeof info !== 'object') continue

      const serviceState = String((info as any).status ?? 'unknown')
      if (serviceState === 'ready' || serviceState === 'disabled') continue

      const url = typeof (info as any).url === 'string' ? (info as any).url : ''
      const httpStatus = typeof (info as any).http_status === 'number' ? (info as any).http_status : null
      const reason = typeof (info as any).reason === 'string' ? (info as any).reason : ''
      const error = typeof (info as any).error === 'string' ? (info as any).error : ''
      const backend = typeof (info as any).backend === 'string' ? (info as any).backend : ''
      const provider = typeof (info as any).selected_provider === 'string' ? (info as any).selected_provider : ''

      const parts: string[] = []
      parts.push(`${service} is ${serviceState}`)
      if (httpStatus !== null) parts.push(`HTTP ${httpStatus}`)
      if (provider) parts.push(`provider: ${provider}`)
      if (backend) parts.push(`backend: ${backend}`)
      if (url) parts.push(`url: ${url}`)

      const detail = (reason || error).trim()
      warnings.push({
        key: service,
        message: parts.join(' · '),
        detail: detail || undefined,
      })
    }

    // If the payload is empty but we didn't throw, it's still useful to warn.
    if (warnings.length === 0 && Object.keys(serviceStatus).length === 0) {
      warnings.push({
        key: 'status_payload',
        message: 'Service status is empty; MCP may be unreachable or returned a non-JSON status payload.',
      })
    }

    return warnings
  }

  const warnings = buildWarnings(status)

  const getStatusIcon = (serviceStatus: string) => {
    if (serviceStatus === 'ready') {
      return <span className="h-3 w-3 bg-green-500 rounded-full"></span>
    } else if (serviceStatus === 'not_ready') {
      return <span className="h-3 w-3 bg-yellow-500 rounded-full"></span>
    } else if (serviceStatus === 'disabled') {
      return <span className="h-3 w-3 bg-gray-400 rounded-full"></span>
    } else {
      return <span className="h-3 w-3 bg-red-500 rounded-full"></span>
    }
  }

  if (loading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">Service Status</h3>
        <div className="flex justify-center">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600"></div>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-4">
      <div className="flex justify-between items-center mb-3">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">Service Status</h3>
        <button
          onClick={loadStatus}
          className="text-sm text-blue-600 hover:text-blue-700 dark:text-blue-400 dark:hover:text-blue-300"
        >
          Refresh
        </button>
      </div>

      {loadError && (
        <div className="mb-3 rounded-md border border-yellow-200 bg-yellow-50 px-3 py-2 text-sm text-gray-700 dark:border-yellow-900/50 dark:bg-yellow-900/20 dark:text-gray-200">
          <p className="font-medium">Warning: failed to fetch service status</p>
          <p className="mt-0.5 break-words">{loadError}</p>
        </div>
      )}

      {status && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {Object.entries(status).map(([service, info]) => (
              <div key={service} className="flex items-center space-x-2">
                {getStatusIcon(info.status)}
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                    {service}
                  </p>
                  <p className="text-xs text-gray-500 dark:text-gray-400">
                    {info.status}
                    {typeof info.http_status === 'number' ? ` (HTTP ${info.http_status})` : ''}
                  </p>
                  {(info.reason || info.error) && (
                    <p
                      className="text-xs text-gray-500 dark:text-gray-400 mt-0.5 line-clamp-2"
                      title={info.reason || info.error}
                    >
                      {info.reason || info.error}
                    </p>
                  )}
                </div>
              </div>
            ))}
          </div>

          {warnings.length > 0 && (
            <div className="mt-4">
              <h4 className="text-sm font-semibold text-gray-900 dark:text-white">Warnings</h4>
              <ul className="mt-2 space-y-2">
                {warnings.map((w) => (
                  <li key={w.key} className="text-sm text-gray-700 dark:text-gray-200">
                    <p className="break-words">{w.message}</p>
                    {w.detail && (
                      <p className="mt-0.5 text-xs text-gray-500 dark:text-gray-400 break-words">{w.detail}</p>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </>
      )}

      {!status && (
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Unable to load service status
        </p>
      )}
    </div>
  )
}
