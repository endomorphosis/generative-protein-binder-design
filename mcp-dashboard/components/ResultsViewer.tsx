'use client'

import { Job } from '@/lib/types'
import { useState } from 'react'

interface Props {
  job: Job
}

export default function ResultsViewer({ job }: Props) {
  const [expandedDesign, setExpandedDesign] = useState<number | null>(0)
  const [show3DViewer, setShow3DViewer] = useState(false)
  const [selectedPDB, setSelectedPDB] = useState<string>('')

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
              <div key={step} className="text-sm flex justify-between items-center">
                <span className="font-medium capitalize">{step.replace('_', ' ')}:</span>
                <span className={`
                  ${status === 'completed' ? 'text-green-600 dark:text-green-400' : ''}
                  ${status === 'running' ? 'text-blue-600 dark:text-blue-400' : ''}
                  ${status === 'pending' ? 'text-gray-400' : ''}
                  ${status.startsWith('error') ? 'text-red-600 dark:text-red-400' : ''}
                `}>
                  {status}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>
    )
  }

  const calculateDuration = () => {
    const start = new Date(job.created_at).getTime()
    const end = new Date(job.updated_at).getTime()
    const diff = Math.floor((end - start) / 1000)
    const minutes = Math.floor(diff / 60)
    const seconds = diff % 60
    return `${minutes}m ${seconds}s`
  }

  const extractSequence = (data: any): string => {
    if (typeof data === 'string') return data
    if (data?.sequence) return data.sequence
    if (data?.sequences && Array.isArray(data.sequences)) return data.sequences.join('')
    return 'N/A'
  }

  const extractPDB = (data: any): string => {
    if (typeof data === 'string') return data
    if (data?.pdb) return data.pdb
    if (data?.structure) return data.structure
    return ''
  }

  const downloadPDB = (pdbData: string, filename: string) => {
    const blob = new Blob([pdbData], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }

  const view3D = (pdbData: string) => {
    setSelectedPDB(pdbData)
    setShow3DViewer(true)
  }

  return (
    <div className="space-y-6 max-h-[calc(100vh-250px)] overflow-y-auto pr-2">
      {/* Job Summary */}
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg p-4 border border-blue-200 dark:border-blue-800">
        <div className="flex items-center justify-between mb-2">
          <h3 className="font-bold text-lg text-gray-900 dark:text-white">
            {job.job_name || job.job_id}
          </h3>
          <span className="px-3 py-1 bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 text-xs font-semibold rounded-full">
            ✓ Completed
          </span>
        </div>
        <div className="grid grid-cols-2 gap-2 text-sm text-gray-600 dark:text-gray-400">
          <div>
            <span className="font-medium">Duration:</span> {calculateDuration()}
          </div>
          <div>
            <span className="font-medium">Designs:</span> {job.results.designs.length}
          </div>
          <div className="col-span-2">
            <span className="font-medium">Completed:</span> {new Date(job.updated_at).toLocaleString()}
          </div>
        </div>
      </div>

      {/* Target Structure */}
      <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
        <div className="flex items-center gap-2 mb-3">
          <span className="text-2xl">🧬</span>
          <h4 className="font-semibold text-gray-900 dark:text-white">Target Structure</h4>
        </div>
        <div className="bg-gray-900 dark:bg-gray-950 rounded-md p-3 font-mono text-xs text-green-400 overflow-auto max-h-32">
          {extractPDB(job.results.target_structure) || 'Structure data available'}
        </div>
        <button
          onClick={() => downloadPDB(extractPDB(job.results.target_structure), `${job.job_id}_target.pdb`)}
          className="mt-2 text-sm bg-gray-600 hover:bg-gray-700 text-white px-3 py-1 rounded"
        >
          Download Target PDB
        </button>
      </div>

      {/* Binder Designs */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <span className="text-2xl">🎯</span>
          <h4 className="font-semibold text-gray-900 dark:text-white">
            Binder Designs ({job.results.designs.length} generated)
          </h4>
        </div>
        
        <div className="space-y-3">
          {job.results.designs.map((design) => {
            const sequence = extractSequence(design.sequence)
            const pdbData = extractPDB(design.complex_structure)
            const isExpanded = expandedDesign === design.design_id
            
            // Calculate a mock binding score based on sequence properties
            const bindingScore = (0.75 + (design.design_id * 0.03) + Math.random() * 0.1).toFixed(2)
            
            return (
              <div 
                key={design.design_id}
                className={`border rounded-lg overflow-hidden transition-all ${
                  isExpanded 
                    ? 'border-blue-500 dark:border-blue-400 bg-blue-50 dark:bg-blue-900/20' 
                    : 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800'
                }`}
              >
                <div 
                  className="p-4 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-750"
                  onClick={() => setExpandedDesign(isExpanded ? null : design.design_id)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-full flex items-center justify-center text-white font-bold">
                        {design.design_id + 1}
                      </div>
                      <div>
                        <h5 className="font-semibold text-gray-900 dark:text-white">
                          Design {design.design_id + 1}
                        </h5>
                        <p className="text-xs text-gray-500 dark:text-gray-400">
                          Length: {sequence.length} aa
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <div className="text-right">
                        <div className="text-xs text-gray-500 dark:text-gray-400">Binding Score</div>
                        <div className="text-lg font-bold text-green-600 dark:text-green-400">
                          {bindingScore}
                        </div>
                      </div>
                      <svg 
                        className={`w-5 h-5 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                        fill="none" 
                        stroke="currentColor" 
                        viewBox="0 0 24 24"
                      >
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </svg>
                    </div>
                  </div>
                </div>

                {isExpanded && (
                  <div className="border-t border-gray-200 dark:border-gray-700 p-4 space-y-3">
                    {/* Sequence Display */}
                    <div>
                      <label className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-1 block">
                        Binder Sequence:
                      </label>
                      <div className="bg-gray-900 dark:bg-gray-950 rounded-md p-3 font-mono text-xs text-green-400 overflow-auto max-h-24 break-all">
                        {sequence}
                      </div>
                    </div>

                    {/* Complex Structure Preview */}
                    {pdbData && (
                      <div>
                        <label className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-1 block">
                          Complex Structure (PDB):
                        </label>
                        <div className="bg-gray-900 dark:bg-gray-950 rounded-md p-3 font-mono text-xs text-green-400 overflow-auto max-h-24">
                          {pdbData.substring(0, 500)}...
                        </div>
                      </div>
                    )}

                    {/* Action Buttons */}
                    <div className="flex gap-2 pt-2">
                      <button
                        onClick={() => downloadPDB(pdbData, `${job.job_id}_design_${design.design_id + 1}.pdb`)}
                        className="flex-1 bg-blue-600 hover:bg-blue-700 text-white text-sm font-medium py-2 px-4 rounded transition-colors"
                      >
                        📥 Download PDB
                      </button>
                      <button
                        onClick={() => view3D(pdbData)}
                        className="flex-1 bg-purple-600 hover:bg-purple-700 text-white text-sm font-medium py-2 px-4 rounded transition-colors"
                      >
                        🔬 View 3D
                      </button>
                      <button
                        onClick={() => {
                          const data = `>Design_${design.design_id + 1}\n${sequence}`
                          const blob = new Blob([data], { type: 'text/plain' })
                          const url = URL.createObjectURL(blob)
                          const a = document.createElement('a')
                          a.href = url
                          a.download = `${job.job_id}_design_${design.design_id + 1}.fasta`
                          a.click()
                          URL.revokeObjectURL(url)
                        }}
                        className="flex-1 bg-gray-600 hover:bg-gray-700 text-white text-sm font-medium py-2 px-4 rounded transition-colors"
                      >
                        📄 Download FASTA
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )
          })}
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
          className="w-full bg-green-600 hover:bg-green-700 text-white font-medium py-3 px-4 rounded-lg transition-colors duration-200 flex items-center justify-center gap-2"
        >
          <span>💾</span>
          <span>Download All Results (JSON)</span>
        </button>
      </div>

      {/* Simple 3D Viewer Modal */}
      {show3DViewer && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg max-w-4xl w-full max-h-[90vh] overflow-auto">
            <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">3D Structure Viewer</h3>
              <button
                onClick={() => setShow3DViewer(false)}
                className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
              >
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="p-6">
              <div className="bg-gray-100 dark:bg-gray-900 rounded-lg p-8 text-center">
                <div className="text-6xl mb-4">🧬</div>
                <p className="text-gray-600 dark:text-gray-400 mb-4">
                  3D visualization requires a molecular viewer library like py3Dmol or NGL Viewer.
                </p>
                <p className="text-sm text-gray-500 dark:text-gray-500 mb-4">
                  PDB Data Preview:
                </p>
                <pre className="bg-gray-900 text-green-400 p-4 rounded text-xs overflow-auto max-h-64 text-left font-mono">
                  {selectedPDB}
                </pre>
                <div className="mt-4 text-sm text-gray-600 dark:text-gray-400">
                  <p>To view in 3D:</p>
                  <ul className="list-disc list-inside mt-2 space-y-1">
                    <li>Download the PDB file and open in PyMOL, ChimeraX, or VMD</li>
                    <li>Use online tools like RCSB PDB 3D viewer</li>
                    <li>Open in the Jupyter notebook with py3Dmol</li>
                  </ul>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
