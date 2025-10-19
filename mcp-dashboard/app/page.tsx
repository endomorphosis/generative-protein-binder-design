'use client'

import { useState, useEffect } from 'react'
import ProteinSequenceForm from '@/components/ProteinSequenceForm'
import JobList from '@/components/JobList'
import ResultsViewer from '@/components/ResultsViewer'
import ServiceStatus from '@/components/ServiceStatus'
import JupyterLauncher from '@/components/JupyterLauncher'
import { Job } from '@/lib/types'

export default function Home() {
  const [selectedJob, setSelectedJob] = useState<Job | null>(null)
  const [refreshTrigger, setRefreshTrigger] = useState(0)

  const handleJobCreated = () => {
    setRefreshTrigger(prev => prev + 1)
  }

  const handleJobSelected = (job: Job) => {
    setSelectedJob(job)
  }

  return (
    <main className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
      <div className="container mx-auto px-4 py-8">
        <header className="mb-8">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-2">
                Protein Binder Design
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                NVIDIA BioNeMo Blueprint - MCP Dashboard
              </p>
            </div>
            {/* NVIDIA Logo would be displayed here if available */}
            <div className="h-12"></div>
          </div>
        </header>

        {/* Service Status */}
        <div className="mb-8">
          <ServiceStatus />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Input Form */}
          <div className="lg:col-span-1">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 mb-6">
              <h2 className="text-2xl font-semibold mb-4 text-gray-900 dark:text-white">
                New Design Job
              </h2>
              <ProteinSequenceForm onJobCreated={handleJobCreated} />
            </div>
            
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
              <h2 className="text-2xl font-semibold mb-4 text-gray-900 dark:text-white">
                Jupyter Notebooks
              </h2>
              <JupyterLauncher />
            </div>
          </div>

          {/* Middle Column - Job List */}
          <div className="lg:col-span-1">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
              <h2 className="text-2xl font-semibold mb-4 text-gray-900 dark:text-white">
                Jobs
              </h2>
              <JobList 
                refreshTrigger={refreshTrigger}
                onJobSelected={handleJobSelected}
              />
            </div>
          </div>

          {/* Right Column - Results */}
          <div className="lg:col-span-1">
            <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
              <h2 className="text-2xl font-semibold mb-4 text-gray-900 dark:text-white">
                Results
              </h2>
              {selectedJob ? (
                <ResultsViewer job={selectedJob} />
              ) : (
                <div className="text-center text-gray-500 dark:text-gray-400 py-12">
                  Select a job to view results
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </main>
  )
}
