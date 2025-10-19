'use client'

export default function JupyterLauncher() {
  const handleLaunchJupyter = () => {
    // Open Jupyter in a new tab
    window.open('http://localhost:8888', '_blank')
  }

  return (
    <div className="space-y-4">
      <p className="text-sm text-gray-600 dark:text-gray-400">
        Launch Jupyter Notebook to explore the protein design workflow interactively.
      </p>

      <button
        onClick={handleLaunchJupyter}
        className="w-full bg-orange-600 hover:bg-orange-700 text-white font-medium py-2 px-4 rounded-md transition-colors duration-200 flex items-center justify-center space-x-2"
      >
        <svg 
          className="w-5 h-5" 
          fill="currentColor" 
          viewBox="0 0 24 24"
        >
          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
        </svg>
        <span>Open Jupyter Notebook</span>
      </button>

      <div className="text-xs text-gray-500 dark:text-gray-400 space-y-1">
        <p>• Example notebook: protein-binder-design.ipynb</p>
        <p>• Default port: 8888</p>
        <p>• Make sure the Jupyter server is running</p>
      </div>

      <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
        <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-2">
          Quick Start
        </h4>
        <div className="bg-gray-50 dark:bg-gray-700 rounded-md p-3">
          <code className="text-xs text-gray-800 dark:text-gray-200">
            cd src<br />
            jupyter notebook
          </code>
        </div>
      </div>
    </div>
  )
}
