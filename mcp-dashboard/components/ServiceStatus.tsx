'use client'

import { useState, useEffect } from 'react'
import { mcpClient } from '@/lib/mcp-client'
import { ServiceStatus as ServiceStatusType } from '@/lib/types'

export default function ServiceStatus() {
  const [status, setStatus] = useState<ServiceStatusType | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadStatus()
    const interval = setInterval(loadStatus, 10000) // Refresh every 10 seconds
    return () => clearInterval(interval)
  }, [])

  const loadStatus = async () => {
    try {
      const serviceStatus = await mcpClient.getServiceStatus()
      setStatus(serviceStatus)
    } catch (err) {
      console.error('Failed to load service status:', err)
    } finally {
      setLoading(false)
    }
  }

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

      {status && (
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
                </p>
              </div>
            </div>
          ))}
        </div>
      )}

      {!status && (
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Unable to load service status
        </p>
      )}
    </div>
  )
}
