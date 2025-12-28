'use client'

import { useState, useEffect } from 'react'
import { getAlphaFoldSettings, updateAlphaFoldSettings, resetAlphaFoldSettings, extractFirstTextContent } from '@/lib/mcp-sdk-client'

interface AlphaFoldSettings {
  speed_preset?: string
  disable_templates?: boolean
  num_recycles?: number
  num_ensemble?: number
  mmseqs2_max_seqs?: number
  msa_mode?: string
}

interface Props {
  onSettingsChanged?: () => void
}

export default function AlphaFoldSettings({ onSettingsChanged }: Props) {
  const [settings, setSettings] = useState<AlphaFoldSettings>({})
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)
  const [expanded, setExpanded] = useState(false)

  // Fetch current settings on mount
  useEffect(() => {
    fetchSettings()
  }, [])

  const fetchSettings = async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await getAlphaFoldSettings()
      const text = extractFirstTextContent(result)
      const parsed = JSON.parse(text)
      setSettings(parsed)
    } catch (err: any) {
      setError(`Failed to load settings: ${err.message}`)
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const handleSettingChange = (key: keyof AlphaFoldSettings, value: any) => {
    setSettings(prev => ({
      ...prev,
      [key]: value
    }))
  }

  const handleSave = async () => {
    setSaving(true)
    setError(null)
    setSuccess(null)
    try {
      const result = await updateAlphaFoldSettings(settings)
      const text = extractFirstTextContent(result)
      const parsed = JSON.parse(text)
      
      if (parsed.success) {
        setSettings(parsed.settings)
        setSuccess('AlphaFold settings updated successfully')
        onSettingsChanged?.()
        // Clear success message after 3 seconds
        setTimeout(() => setSuccess(null), 3000)
      } else {
        setError(parsed.message || 'Failed to update settings')
      }
    } catch (err: any) {
      setError(`Failed to save settings: ${err.message}`)
      console.error(err)
    } finally {
      setSaving(false)
    }
  }

  const handleReset = async () => {
    if (!confirm('Reset AlphaFold settings to defaults?')) return
    
    setSaving(true)
    setError(null)
    setSuccess(null)
    try {
      const result = await resetAlphaFoldSettings()
      const text = extractFirstTextContent(result)
      const parsed = JSON.parse(text)
      
      if (parsed.success) {
        setSettings(parsed.settings)
        setSuccess('AlphaFold settings reset to defaults')
        onSettingsChanged?.()
        // Clear success message after 3 seconds
        setTimeout(() => setSuccess(null), 3000)
      } else {
        setError(parsed.message || 'Failed to reset settings')
      }
    } catch (err: any) {
      setError(`Failed to reset settings: ${err.message}`)
      console.error(err)
    } finally {
      setSaving(false)
    }
  }

  if (loading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
        <div className="animate-pulse">
          <div className="h-6 bg-gray-300 dark:bg-gray-600 rounded w-1/3"></div>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full px-6 py-4 flex items-center justify-between hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
      >
        <div className="flex items-center space-x-3">
          <svg
            className="w-5 h-5 text-blue-600"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <div className="text-left">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              AlphaFold Optimization Settings
            </h3>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              Configure speed vs quality tradeoffs (⚡ 29% speedup available)
            </p>
          </div>
        </div>
        <svg
          className={`w-5 h-5 text-gray-500 transition-transform ${expanded ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
        </svg>
      </button>

      {expanded && (
        <div className="px-6 py-4 border-t border-gray-200 dark:border-gray-700 space-y-6">
          {/* Speed Preset */}
          <div>
            <label htmlFor="speed_preset" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Speed Preset
            </label>
            <div className="relative">
              <select
                id="speed_preset"
                value={settings.speed_preset || 'balanced'}
                onChange={(e) => handleSettingChange('speed_preset', e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white appearance-none cursor-pointer"
              >
                <option value="fast">⚡ Fast (29% faster - templates OFF, recycles=3)</option>
                <option value="balanced">⚖️ Balanced (20% faster - templates ON, recycles=3, default)</option>
                <option value="quality">🎯 Quality (slowest - templates ON, full recycles)</option>
              </select>
              <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700 dark:text-gray-300">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                </svg>
              </div>
            </div>
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
              {settings.speed_preset === 'fast' && 'Fastest option: removes templates, reduces iterations'}
              {settings.speed_preset === 'balanced' && 'Best balance of speed and quality (recommended)'}
              {settings.speed_preset === 'quality' && 'Slowest option: maximum accuracy for research/publication'}
            </p>
          </div>

          {/* Advanced Settings (Collapsible) */}
          <details className="group">
            <summary className="flex items-center cursor-pointer text-sm font-medium text-gray-700 dark:text-gray-300">
              <svg className="w-4 h-4 mr-2 group-open:rotate-90 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
              Advanced Settings
            </summary>

            <div className="mt-4 space-y-4 pl-6 border-l-2 border-gray-200 dark:border-gray-700">
              {/* Disable Templates */}
              <div>
                <label className="flex items-center space-x-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={settings.disable_templates || false}
                    onChange={(e) => handleSettingChange('disable_templates', e.target.checked)}
                    className="w-4 h-4 rounded border-gray-300 text-blue-600 focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:border-gray-600"
                  />
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    Disable Templates
                  </span>
                </label>
                <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                  Skip PDB template search (~10% speedup, less accurate for template-dependent proteins)
                </p>
              </div>

              {/* Number of Recycles */}
              <div>
                <label htmlFor="num_recycles" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Recycling Iterations: {settings.num_recycles || 'default'}
                </label>
                <input
                  type="number"
                  id="num_recycles"
                  min="-1"
                  max="50"
                  value={settings.num_recycles ?? ''}
                  onChange={(e) => handleSettingChange('num_recycles', e.target.value === '' ? null : parseInt(e.target.value))}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                  placeholder="3 (speed) to 20 (quality)"
                />
                <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                  3 for speed, -1 for model default (~20), higher = more iterations = slower
                </p>
              </div>

              {/* Number of Ensemble */}
              <div>
                <label htmlFor="num_ensemble" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Ensemble Evaluations: {settings.num_ensemble || 'default'}
                </label>
                <input
                  type="number"
                  id="num_ensemble"
                  min="1"
                  max="8"
                  value={settings.num_ensemble ?? ''}
                  onChange={(e) => handleSettingChange('num_ensemble', e.target.value === '' ? null : parseInt(e.target.value))}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                  placeholder="1 (speed) to 8 (quality)"
                />
                <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                  1 for speed, 8 for CASP14 quality
                </p>
              </div>

              {/* MMseqs2 Max Sequences */}
              <div>
                <label htmlFor="mmseqs2_max_seqs" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  MMseqs2 Max Sequences: {settings.mmseqs2_max_seqs || 'default'}
                </label>
                <input
                  type="number"
                  id="mmseqs2_max_seqs"
                  min="50"
                  step="50"
                  value={settings.mmseqs2_max_seqs ?? ''}
                  onChange={(e) => handleSettingChange('mmseqs2_max_seqs', e.target.value === '' ? null : parseInt(e.target.value))}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white"
                  placeholder="512 (speed) to 10000 (quality)"
                />
                <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                  512 for speed, 10000 for maximum coverage
                </p>
              </div>

              {/* MSA Mode */}
              <div>
                <label htmlFor="msa_mode" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  MSA Generation Mode
                </label>
                <div className="relative">
                  <select
                    id="msa_mode"
                    value={settings.msa_mode || 'mmseqs2'}
                    onChange={(e) => handleSettingChange('msa_mode', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 dark:bg-gray-700 dark:text-white appearance-none cursor-pointer"
                  >
                    <option value="mmseqs2">MMseqs2 (faster, requires database)</option>
                    <option value="jackhmmer">JackHMMER (slower, more compatible)</option>
                  </select>
                  <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-gray-700 dark:text-gray-300">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 14l-7 7m0 0l-7-7m7 7V3" />
                    </svg>
                  </div>
                </div>
                <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                  MMseqs2 is faster but requires a prepared database
                </p>
              </div>
            </div>
          </details>

          {/* Messages */}
          {error && (
            <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
              <p className="text-sm text-red-700 dark:text-red-400">{error}</p>
            </div>
          )}

          {success && (
            <div className="p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-md">
              <p className="text-sm text-green-700 dark:text-green-400">{success}</p>
            </div>
          )}

          {/* Buttons */}
          <div className="flex gap-3 pt-4 border-t border-gray-200 dark:border-gray-700">
            <button
              onClick={handleSave}
              disabled={saving}
              className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white font-medium rounded-md transition-colors"
            >
              {saving ? 'Saving...' : 'Save Settings'}
            </button>
            <button
              onClick={handleReset}
              disabled={saving}
              className="flex-1 px-4 py-2 bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 disabled:opacity-50 text-gray-900 dark:text-white font-medium rounded-md transition-colors"
            >
              Reset to Defaults
            </button>
          </div>

          {/* Current Settings Display */}
          <div className="bg-gray-50 dark:bg-gray-900 p-3 rounded-md">
            <p className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-2">Current Settings (JSON):</p>
            <pre className="text-xs text-gray-700 dark:text-gray-300 overflow-auto max-h-32">
              {JSON.stringify(settings, null, 2)}
            </pre>
          </div>
        </div>
      )}
    </div>
  )
}
