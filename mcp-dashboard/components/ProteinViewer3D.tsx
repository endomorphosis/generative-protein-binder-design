'use client'

import { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

interface ProteinViewer3DProps {
  pdbData: string
  onClose: () => void
  title?: string
}

export default function ProteinViewer3D({ pdbData, onClose, title }: ProteinViewer3DProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [error, setError] = useState<string | null>(null)
  const [renderMode, setRenderMode] = useState<'cartoon' | 'sphere' | 'stick'>('sphere')
  const sceneRef = useRef<THREE.Scene | null>(null)
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const controlsRef = useRef<OrbitControls | null>(null)

  useEffect(() => {
    if (!containerRef.current || !pdbData) return

    try {
      // Parse PDB data
      const atoms = parsePDB(pdbData)
      
      if (atoms.length === 0) {
        setError('No valid atoms found in PDB data')
        return
      }

      // Setup scene
      const scene = new THREE.Scene()
      scene.background = new THREE.Color(0x1a202c)
      sceneRef.current = scene

      // Setup camera
      const camera = new THREE.PerspectiveCamera(
        75,
        containerRef.current.clientWidth / containerRef.current.clientHeight,
        0.1,
        1000
      )
      camera.position.z = 50
      cameraRef.current = camera

      // Setup renderer
      const renderer = new THREE.WebGLRenderer({ antialias: true })
      renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight)
      containerRef.current.appendChild(renderer.domElement)
      rendererRef.current = renderer

      // Setup controls
      const controls = new OrbitControls(camera, renderer.domElement)
      controls.enableDamping = true
      controls.dampingFactor = 0.05
      controls.minDistance = 10
      controls.maxDistance = 200
      controlsRef.current = controls

      // Add lights
      const ambientLight = new THREE.AmbientLight(0xffffff, 0.6)
      scene.add(ambientLight)

      const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8)
      directionalLight1.position.set(1, 1, 1)
      scene.add(directionalLight1)

      const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4)
      directionalLight2.position.set(-1, -1, -1)
      scene.add(directionalLight2)

      // Create molecule
      createMolecule(scene, atoms, renderMode)

      // Center camera on molecule
      const box = new THREE.Box3().setFromObject(scene)
      const center = box.getCenter(new THREE.Vector3())
      const size = box.getSize(new THREE.Vector3())
      const maxDim = Math.max(size.x, size.y, size.z)
      const fov = camera.fov * (Math.PI / 180)
      let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2))
      cameraZ *= 1.5
      camera.position.set(center.x, center.y, center.z + cameraZ)
      camera.lookAt(center)
      controls.target.copy(center)

      // Animation loop
      const animate = () => {
        requestAnimationFrame(animate)
        controls.update()
        renderer.render(scene, camera)
      }
      animate()

      // Handle resize
      const handleResize = () => {
        if (!containerRef.current || !camera || !renderer) return
        camera.aspect = containerRef.current.clientWidth / containerRef.current.clientHeight
        camera.updateProjectionMatrix()
        renderer.setSize(containerRef.current.clientWidth, containerRef.current.clientHeight)
      }
      window.addEventListener('resize', handleResize)

      return () => {
        window.removeEventListener('resize', handleResize)
        if (containerRef.current && renderer.domElement) {
          containerRef.current.removeChild(renderer.domElement)
        }
        renderer.dispose()
        controls.dispose()
      }
    } catch (err) {
      setError(`Failed to render 3D structure: ${err}`)
    }
  }, [pdbData, renderMode])

  const parsePDB = (pdb: string) => {
    const atoms: Array<{
      element: string
      x: number
      y: number
      z: number
      residue: string
    }> = []

    const lines = pdb.split('\n')
    for (const line of lines) {
      if (line.startsWith('ATOM') || line.startsWith('HETATM')) {
        try {
          const element = line.substring(76, 78).trim() || line.substring(12, 16).trim().charAt(0)
          const x = parseFloat(line.substring(30, 38))
          const y = parseFloat(line.substring(38, 46))
          const z = parseFloat(line.substring(46, 54))
          const residue = line.substring(17, 20).trim()

          if (!isNaN(x) && !isNaN(y) && !isNaN(z)) {
            atoms.push({ element, x, y, z, residue })
          }
        } catch (err) {
          // Skip malformed lines
        }
      }
    }

    return atoms
  }

  const createMolecule = (
    scene: THREE.Scene,
    atoms: Array<{ element: string; x: number; y: number; z: number; residue: string }>,
    mode: 'cartoon' | 'sphere' | 'stick'
  ) => {
    // Clear existing molecule
    scene.children = scene.children.filter(
      child => !(child instanceof THREE.Mesh || child instanceof THREE.Line)
    )

    const colors: Record<string, number> = {
      C: 0x909090,
      N: 0x3050f8,
      O: 0xff0d0d,
      S: 0xffff30,
      H: 0xffffff,
      P: 0xff8000,
      default: 0xff00ff
    }

    if (mode === 'sphere') {
      // Ball-and-stick representation
      atoms.forEach(atom => {
        const geometry = new THREE.SphereGeometry(0.3, 16, 16)
        const color = colors[atom.element] || colors.default
        const material = new THREE.MeshPhongMaterial({ color })
        const sphere = new THREE.Mesh(geometry, material)
        sphere.position.set(atom.x, atom.y, atom.z)
        scene.add(sphere)
      })

      // Add bonds (simple distance-based)
      for (let i = 0; i < atoms.length; i++) {
        for (let j = i + 1; j < atoms.length; j++) {
          const dist = Math.sqrt(
            Math.pow(atoms[i].x - atoms[j].x, 2) +
            Math.pow(atoms[i].y - atoms[j].y, 2) +
            Math.pow(atoms[i].z - atoms[j].z, 2)
          )
          if (dist < 1.8) {
            const points = [
              new THREE.Vector3(atoms[i].x, atoms[i].y, atoms[i].z),
              new THREE.Vector3(atoms[j].x, atoms[j].y, atoms[j].z)
            ]
            const geometry = new THREE.BufferGeometry().setFromPoints(points)
            const material = new THREE.LineBasicMaterial({ color: 0x666666 })
            const line = new THREE.Line(geometry, material)
            scene.add(line)
          }
        }
      }
    } else if (mode === 'stick') {
      // Stick representation
      for (let i = 0; i < atoms.length; i++) {
        for (let j = i + 1; j < atoms.length; j++) {
          const dist = Math.sqrt(
            Math.pow(atoms[i].x - atoms[j].x, 2) +
            Math.pow(atoms[i].y - atoms[j].y, 2) +
            Math.pow(atoms[i].z - atoms[j].z, 2)
          )
          if (dist < 1.8) {
            const points = [
              new THREE.Vector3(atoms[i].x, atoms[i].y, atoms[i].z),
              new THREE.Vector3(atoms[j].x, atoms[j].y, atoms[j].z)
            ]
            const geometry = new THREE.BufferGeometry().setFromPoints(points)
            const color = colors[atoms[i].element] || colors.default
            const material = new THREE.LineBasicMaterial({ color, linewidth: 3 })
            const line = new THREE.Line(geometry, material)
            scene.add(line)
          }
        }
      }
    } else {
      // Cartoon representation (simplified as spheres for backbone atoms)
      const backboneAtoms = atoms.filter(a => a.element === 'C' && a.residue !== 'UNK')
      backboneAtoms.forEach((atom, i) => {
        const geometry = new THREE.SphereGeometry(0.5, 16, 16)
        const material = new THREE.MeshPhongMaterial({ color: 0x48bb78 })
        const sphere = new THREE.Mesh(geometry, material)
        sphere.position.set(atom.x, atom.y, atom.z)
        scene.add(sphere)

        // Connect backbone
        if (i < backboneAtoms.length - 1) {
          const next = backboneAtoms[i + 1]
          const points = [
            new THREE.Vector3(atom.x, atom.y, atom.z),
            new THREE.Vector3(next.x, next.y, next.z)
          ]
          const geometry = new THREE.BufferGeometry().setFromPoints(points)
          const material = new THREE.LineBasicMaterial({ color: 0x48bb78, linewidth: 2 })
          const line = new THREE.Line(geometry, material)
          scene.add(line)
        }
      })
    }
  }

  const changeRenderMode = (mode: 'cartoon' | 'sphere' | 'stick') => {
    setRenderMode(mode)
  }

  return (
    <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg w-full max-w-6xl h-[90vh] flex flex-col">
        {/* Header */}
        <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex justify-between items-center">
          <div>
            <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
              🔬 3D Protein Structure Viewer
            </h3>
            {title && <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{title}</p>}
          </div>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 p-2"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Controls */}
        <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex gap-2 items-center">
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Render Mode:</span>
          <button
            onClick={() => changeRenderMode('sphere')}
            className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
              renderMode === 'sphere'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300'
            }`}
          >
            Ball & Stick
          </button>
          <button
            onClick={() => changeRenderMode('stick')}
            className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
              renderMode === 'stick'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300'
            }`}
          >
            Stick
          </button>
          <button
            onClick={() => changeRenderMode('cartoon')}
            className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
              renderMode === 'cartoon'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300'
            }`}
          >
            Cartoon
          </button>
          <div className="ml-auto text-xs text-gray-500 dark:text-gray-400">
            Use mouse to rotate, scroll to zoom
          </div>
        </div>

        {/* 3D Viewer */}
        <div className="flex-1 relative">
          {error ? (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <div className="text-6xl mb-4">⚠️</div>
                <p className="text-red-600 dark:text-red-400">{error}</p>
              </div>
            </div>
          ) : (
            <div ref={containerRef} className="w-full h-full" />
          )}
        </div>

        {/* Footer with legend */}
        <div className="p-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
          <div className="flex items-center gap-6 text-sm">
            <span className="font-medium text-gray-700 dark:text-gray-300">Color Legend:</span>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-gray-500"></div>
              <span className="text-gray-600 dark:text-gray-400">Carbon</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-blue-500"></div>
              <span className="text-gray-600 dark:text-gray-400">Nitrogen</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-red-500"></div>
              <span className="text-gray-600 dark:text-gray-400">Oxygen</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-yellow-400"></div>
              <span className="text-gray-600 dark:text-gray-400">Sulfur</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 rounded-full bg-orange-500"></div>
              <span className="text-gray-600 dark:text-gray-400">Phosphorus</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
