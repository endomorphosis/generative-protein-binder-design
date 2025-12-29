'use client'

import { useEffect, useState } from 'react'
import { mcpClient } from '@/lib/mcp-client'
import { ProteinSequenceInput } from '@/lib/types'

interface Props {
  onJobCreated: () => void
  prefill?: Partial<ProteinSequenceInput>
}

export default function ProteinSequenceForm({ onJobCreated, prefill }: Props) {
  const [formData, setFormData] = useState<ProteinSequenceInput>({
    sequence: '',
    job_name: '',
    num_designs: 5,
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!prefill) return

    setFormData((prev) => ({
      ...prev,
      sequence: typeof prefill.sequence === 'string' ? prefill.sequence : prev.sequence,
      num_designs:
        typeof prefill.num_designs === 'number' && Number.isFinite(prefill.num_designs)
          ? prefill.num_designs
          : prev.num_designs,
    }))
  }, [prefill?.sequence, prefill?.num_designs])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    setError(null)

    try {
      await mcpClient.createJob({
        sequence: formData.sequence,
        job_name: formData.job_name || undefined,
        num_designs: formData.num_designs,
      })
      
      // Reset form
      setFormData({
        sequence: '',
        job_name: '',
        num_designs: 5,
      })
      
      onJobCreated()
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to create job')
    } finally {
      setLoading(false)
    }
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div>
        <label htmlFor="job_name" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          Job Name (Optional)
        </label>
        <input
          type="text"
          id="job_name"
          value={formData.job_name}
          onChange={(e) => setFormData({ ...formData, job_name: e.target.value })}
          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
          placeholder="My Protein Design"
        />
      </div>

      <div>
        <label htmlFor="sequence" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          Target Protein Sequence *
        </label>
        <textarea
          id="sequence"
          value={formData.sequence}
          onChange={(e) => setFormData({ ...formData, sequence: e.target.value })}
          required
          rows={6}
          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white font-mono text-sm"
          placeholder="MKFLKFSLLTAVLLSVVFAFSSCGDDDDTGYLPPSQAIQDLLKRMKV..."
        />
        <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
          Enter a valid amino acid sequence using standard one-letter codes
        </p>
      </div>

      <div>
        <label htmlFor="num_designs" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          Number of Designs
        </label>
        <input
          type="number"
          id="num_designs"
          value={formData.num_designs}
          onChange={(e) => setFormData({ ...formData, num_designs: parseInt(e.target.value) })}
          min="1"
          max="20"
          className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
        />
      </div>

      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md p-3">
          <p className="text-sm text-red-800 dark:text-red-400">{error}</p>
        </div>
      )}

      <button
        type="submit"
        disabled={loading || !formData.sequence}
        className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium py-2 px-4 rounded-md transition-colors duration-200"
      >
        {loading ? 'Creating Job...' : 'Start Design Job'}
      </button>
    </form>
  )
}
