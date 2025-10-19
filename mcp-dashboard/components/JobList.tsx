'use client'

import { useState, useEffect } from 'react'
import { mcpClient } from '@/lib/mcp-client'
import { Job } from '@/lib/types'

interface Props {
  refreshTrigger: number
  onJobSelected: (job: Job) => void
}

export default function JobList({ refreshTrigger, onJobSelected }: Props) {
  const [jobs, setJobs] = useState<Job[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null)

  useEffect(() => {
    loadJobs()
  }, [refreshTrigger])

  useEffect(() => {
    // Auto-refresh every 5 seconds
    const interval = setInterval(loadJobs, 5000)
    return () => clearInterval(interval)
  }, [])

  const loadJobs = async () => {
    try {
      const jobList = await mcpClient.listJobs()
      setJobs(jobList.sort((a, b) => 
        new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      ))
    } catch (err) {
      console.error('Failed to load jobs:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleJobClick = (job: Job) => {
    setSelectedJobId(job.job_id)
    onJobSelected(job)
  }

  const handleDelete = async (jobId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    if (confirm('Are you sure you want to delete this job?')) {
      try {
        await mcpClient.deleteJob(jobId)
        loadJobs()
        if (selectedJobId === jobId) {
          setSelectedJobId(null)
        }
      } catch (err) {
        console.error('Failed to delete job:', err)
      }
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-300'
      case 'running': return 'bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-300'
      case 'failed': return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-300'
      default: return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300'
    }
  }

  if (loading) {
    return (
      <div className="flex justify-center items-center py-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    )
  }

  if (jobs.length === 0) {
    return (
      <div className="text-center text-gray-500 dark:text-gray-400 py-8">
        No jobs yet. Create one to get started!
      </div>
    )
  }

  return (
    <div className="space-y-2 max-h-[600px] overflow-y-auto">
      {jobs.map((job) => (
        <div
          key={job.job_id}
          onClick={() => handleJobClick(job)}
          className={`p-4 border rounded-lg cursor-pointer transition-all duration-200 ${
            selectedJobId === job.job_id
              ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20'
              : 'border-gray-200 dark:border-gray-700 hover:border-blue-300 dark:hover:border-blue-700'
          }`}
        >
          <div className="flex justify-between items-start mb-2">
            <div className="flex-1">
              <h3 className="font-medium text-gray-900 dark:text-white">
                {job.job_name || job.job_id}
              </h3>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                {new Date(job.created_at).toLocaleString()}
              </p>
            </div>
            <button
              onClick={(e) => handleDelete(job.job_id, e)}
              className="text-red-600 hover:text-red-800 dark:text-red-400 dark:hover:text-red-300 text-sm"
            >
              Delete
            </button>
          </div>
          
          <div className="flex items-center justify-between">
            <span className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(job.status)}`}>
              {job.status}
            </span>
            
            {job.status === 'running' && (
              <div className="flex items-center space-x-1">
                <div className="animate-pulse h-2 w-2 bg-blue-600 rounded-full"></div>
                <span className="text-xs text-gray-600 dark:text-gray-400">Processing...</span>
              </div>
            )}
          </div>

          {job.status === 'running' && (
            <div className="mt-3 space-y-1">
              {Object.entries(job.progress).map(([step, status]) => (
                <div key={step} className="flex justify-between text-xs">
                  <span className="text-gray-600 dark:text-gray-400">{step}:</span>
                  <span className={`font-medium ${
                    status === 'completed' ? 'text-green-600 dark:text-green-400' :
                    status === 'running' ? 'text-blue-600 dark:text-blue-400' :
                    status.startsWith('error') ? 'text-red-600 dark:text-red-400' :
                    'text-gray-500 dark:text-gray-500'
                  }`}>
                    {status}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
