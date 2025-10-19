'use client'

import { Job } from '@/lib/types'

interface Props {
  job: Job
}

export default function ResultsViewer({ job }: Props) {
  if (job.status === 'failed') {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md p-4">
        <h3 className="text-red-800 dark:text-red-400 font-medium mb-2">Job Failed</h3>
        <p className="text-sm text-red-700 dark:text-red-300">{job.error}</p>
      </div>
    )
  }

  if (job.status !== 'completed' || !job.results) {
    return (
      <div className="text-center text-gray-500 dark:text-gray-400 py-8">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
        <p>Job is {job.status}...</p>
        {job.status === 'running' && (
          <div className="mt-4 space-y-2">
            {Object.entries(job.progress).map(([step, status]) => (
              <div key={step} className="text-sm">
                <span className="font-medium">{step}:</span> {status}
              </div>
            ))}
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Job Info */}
      <div className="border-b border-gray-200 dark:border-gray-700 pb-4">
        <h3 className="font-semibold text-lg text-gray-900 dark:text-white mb-2">
          {job.job_name || job.job_id}
        </h3>
        <div className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
          <p>Status: <span className="text-green-600 dark:text-green-400 font-medium">Completed</span></p>
          <p>Created: {new Date(job.created_at).toLocaleString()}</p>
          <p>Completed: {new Date(job.updated_at).toLocaleString()}</p>
        </div>
      </div>

      {/* Target Structure */}
      <div>
        <h4 className="font-semibold text-gray-900 dark:text-white mb-2">Target Structure</h4>
        <div className="bg-gray-50 dark:bg-gray-700 rounded-md p-3">
          <pre className="text-xs overflow-auto max-h-32">
            {JSON.stringify(job.results.target_structure, null, 2)}
          </pre>
        </div>
      </div>

      {/* Binder Designs */}
      <div>
        <h4 className="font-semibold text-gray-900 dark:text-white mb-2">
          Binder Designs ({job.results.designs.length})
        </h4>
        <div className="space-y-3">
          {job.results.designs.map((design) => (
            <div 
              key={design.design_id}
              className="border border-gray-200 dark:border-gray-700 rounded-md p-4"
            >
              <h5 className="font-medium text-gray-900 dark:text-white mb-2">
                Design {design.design_id + 1}
              </h5>
              
              <div className="space-y-2 text-sm">
                <div>
                  <span className="font-medium text-gray-700 dark:text-gray-300">Backbone:</span>
                  <div className="bg-gray-50 dark:bg-gray-700 rounded mt-1 p-2">
                    <pre className="text-xs overflow-auto max-h-24">
                      {JSON.stringify(design.backbone, null, 2)}
                    </pre>
                  </div>
                </div>

                <div>
                  <span className="font-medium text-gray-700 dark:text-gray-300">Sequence:</span>
                  <div className="bg-gray-50 dark:bg-gray-700 rounded mt-1 p-2">
                    <pre className="text-xs overflow-auto max-h-24">
                      {JSON.stringify(design.sequence, null, 2)}
                    </pre>
                  </div>
                </div>

                <div>
                  <span className="font-medium text-gray-700 dark:text-gray-300">Complex Structure:</span>
                  <div className="bg-gray-50 dark:bg-gray-700 rounded mt-1 p-2">
                    <pre className="text-xs overflow-auto max-h-24">
                      {JSON.stringify(design.complex_structure, null, 2)}
                    </pre>
                  </div>
                </div>
              </div>

              <div className="mt-3 flex space-x-2">
                <button className="text-xs bg-blue-600 hover:bg-blue-700 text-white px-3 py-1 rounded">
                  Download PDB
                </button>
                <button className="text-xs bg-gray-600 hover:bg-gray-700 text-white px-3 py-1 rounded">
                  View in 3D
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Download All Results */}
      <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
        <button 
          onClick={() => {
            const blob = new Blob([JSON.stringify(job.results, null, 2)], { type: 'application/json' })
            const url = URL.createObjectURL(blob)
            const a = document.createElement('a')
            a.href = url
            a.download = `${job.job_id}_results.json`
            a.click()
            URL.revokeObjectURL(url)
          }}
          className="w-full bg-green-600 hover:bg-green-700 text-white font-medium py-2 px-4 rounded-md transition-colors duration-200"
        >
          Download All Results (JSON)
        </button>
      </div>
    </div>
  )
}
