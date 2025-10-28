'use client'

import { useEffect, useRef, useState } from 'react'
// Note: Three.js imports will be dynamically loaded to avoid SSR issues

interface ProteinViewer3DProps {
  pdbData: string
  onClose: () => void
  title?: string
}

export default function ProteinViewer3D({ pdbData, onClose, title }: ProteinViewer3DProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [error, setError] = useState<string | null>(null)
  const [renderMode, setRenderMode] = useState<'ribbon' | 'cartoon' | 'sphere' | 'stick'>('ribbon')
  const [showHeatmap, setShowHeatmap] = useState(false)
  const sceneRef = useRef<any>(null)
  const cameraRef = useRef<any>(null)
  const rendererRef = useRef<any>(null)
  const controlsRef = useRef<any>(null)
  const [THREE, setTHREE] = useState<any>(null)
  const [OrbitControls, setOrbitControls] = useState<any>(null)

  // Element colors (CPK coloring)
  const elementColors: { [key: string]: number } = {
    'C': 0x404040,  // Gray
    'N': 0x0080ff,  // Blue
    'O': 0xff4000,  // Red
    'S': 0xffff00,  // Yellow
    'P': 0xff8000,  // Orange
    'H': 0xffffff,  // White
  }

  // Secondary structure colors
  const secondaryStructureColors: { [key: string]: number } = {
    'helix': 0xff4081,    // Pink/Magenta
    'sheet': 0xffd600,    // Yellow
    'turn': 0x00bcd4,     // Cyan
    'coil': 0x808080,     // Gray
  }

  useEffect(() => {
    const loadThreeJS = async () => {
      try {
        const [threejs, { OrbitControls: OC }] = await Promise.all([
          import('three'),
          import('three/examples/jsm/controls/OrbitControls.js')
        ])
        setTHREE(threejs)
        setOrbitControls(() => OC)
      } catch (err) {
        setError('Failed to load 3D viewer libraries')
      }
    }
    loadThreeJS()
  }, [])

  useEffect(() => {
    if (!containerRef.current || !pdbData || !THREE || !OrbitControls) return

    try {
      // Parse PDB data
      const atoms = parsePDB(pdbData)
      
      if (atoms.length === 0) {
        setError('No valid atoms found in PDB data')
        return
      }

      // Clear container
      while (containerRef.current.firstChild) {
        containerRef.current.removeChild(containerRef.current.firstChild)
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
      createMolecule(scene, atoms, renderMode, showHeatmap)

      // Center camera on molecule
      const box = new THREE.Box3().setFromObject(scene)
      const center = box.getCenter(new THREE.Vector3())
      const size = box.getSize(new THREE.Vector3())
      
      camera.position.copy(center)
      camera.position.z += Math.max(size.x, size.y, size.z) * 1.5
      controls.target.copy(center)
      controls.update()

      // Animation loop
      const animate = () => {
        requestAnimationFrame(animate)
        controls.update()
        renderer.render(scene, camera)
      }
      animate()

      // Handle resize
      const handleResize = () => {
        if (containerRef.current && renderer && camera) {
          const width = containerRef.current.clientWidth
          const height = containerRef.current.clientHeight
          camera.aspect = width / height
          camera.updateProjectionMatrix()
          renderer.setSize(width, height)
        }
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
  }, [pdbData, renderMode, showHeatmap, THREE, OrbitControls])

  const parsePDB = (pdb: string) => {
    const atoms: Array<{
      element: string
      x: number
      y: number
      z: number
      residue: string
      residueNum: number
      chain: string
      atomName: string
      bFactor: number
    }> = []

    const lines = pdb.split('\n')
    for (const line of lines) {
      if (line.startsWith('ATOM') || line.startsWith('HETATM')) {
        try {
          const atomName = line.substring(12, 16).trim()
          const element = line.substring(76, 78).trim() || atomName.charAt(0)
          const x = parseFloat(line.substring(30, 38))
          const y = parseFloat(line.substring(38, 46))
          const z = parseFloat(line.substring(46, 54))
          const residue = line.substring(17, 20).trim()
          const residueNum = parseInt(line.substring(22, 26).trim())
          const chain = line.substring(21, 22).trim()
          const bFactor = parseFloat(line.substring(60, 66).trim()) || 0

          if (!isNaN(x) && !isNaN(y) && !isNaN(z)) {
            atoms.push({ element, x, y, z, residue, residueNum, chain, atomName, bFactor })
          }
        } catch (err) {
          // Skip malformed lines
        }
      }
    }
    return atoms
  }

  const detectSecondaryStructure = (atoms: any[]) => {
    // Simple secondary structure detection based on CA atom distances
    const caAtoms = atoms.filter(atom => atom.atomName === 'CA')
    const structures: Array<{ start: number; end: number; type: string }> = []
    
    // Detect alpha helices (i to i+3 distance ~5.4Å)
    for (let i = 0; i < caAtoms.length - 3; i++) {
      if (caAtoms[i + 3]) {
        const dist = Math.sqrt(
          Math.pow(caAtoms[i + 3].x - caAtoms[i].x, 2) +
          Math.pow(caAtoms[i + 3].y - caAtoms[i].y, 2) +
          Math.pow(caAtoms[i + 3].z - caAtoms[i].z, 2)
        )
        
        if (dist >= 4.5 && dist <= 6.5) {
          let helixEnd = i + 3
          // Extend helix
          for (let j = i + 4; j < caAtoms.length; j++) {
            if (!caAtoms[j]) break
            const nextDist = Math.sqrt(
              Math.pow(caAtoms[j].x - caAtoms[j - 3].x, 2) +
              Math.pow(caAtoms[j].y - caAtoms[j - 3].y, 2) +
              Math.pow(caAtoms[j].z - caAtoms[j - 3].z, 2)
            )
            if (nextDist < 4.5 || nextDist > 6.5) break
            helixEnd++
          }
          if (helixEnd - i >= 4) {
            structures.push({ start: i, end: helixEnd, type: 'helix' })
          }
        }
      }
    }
    
    return structures
  }

  const createMolecule = (
    scene: any,
    atoms: Array<{ element: string; x: number; y: number; z: number; residue: string; residueNum: number; chain: string; atomName: string; bFactor: number }>,
    mode: 'ribbon' | 'cartoon' | 'sphere' | 'stick',
    heatmap: boolean
  ) => {
    if (!THREE) return

    // Clear existing molecule
    scene.children = scene.children.filter(
      (child: any) => !(child instanceof THREE.Mesh || child instanceof THREE.Line)
    )

    if (mode === 'ribbon' || mode === 'cartoon') {
      // Get CA atoms for backbone
      const caAtoms = atoms.filter(atom => atom.atomName === 'CA')
      const structures = detectSecondaryStructure(atoms)

      if (caAtoms.length === 0) {
        // Fallback if no CA atoms found
        setError('No C-alpha atoms found for ribbon visualization')
        return
      }

      // Create ribbon segments
      for (let i = 0; i < caAtoms.length - 1; i++) {
        const curr = caAtoms[i]
        const next = caAtoms[i + 1]
        
        // Determine structure type for this segment
        let structureType = 'coil'
        for (const struct of structures) {
          if (i >= struct.start && i <= struct.end) {
            structureType = struct.type
            break
          }
        }

        const color = heatmap 
          ? new THREE.Color().setHSL((1 - curr.bFactor / 100) * 0.7, 1, 0.5)
          : new THREE.Color(secondaryStructureColors[structureType])

        // Create ribbon segment
        const direction = new THREE.Vector3(
          next.x - curr.x,
          next.y - curr.y,
          next.z - curr.z
        )
        const length = direction.length()
        direction.normalize()

        if (structureType === 'helix' && mode === 'ribbon') {
          // Helical ribbon - coil shape
          const curve = new THREE.CatmullRomCurve3([
            new THREE.Vector3(curr.x, curr.y, curr.z),
            new THREE.Vector3(next.x, next.y, next.z)
          ])
          const tubeGeometry = new THREE.TubeGeometry(curve, 8, 0.4, 8, false)
          const material = new THREE.MeshPhongMaterial({ 
            color, 
            side: THREE.DoubleSide,
            shininess: 30
          })
          const tube = new THREE.Mesh(tubeGeometry, material)
          scene.add(tube)
        } else if (structureType === 'sheet' && mode === 'ribbon') {
          // Beta sheet - flat arrow ribbon showing directionality
          const arrowWidth = 0.8
          const arrowHeight = 0.1
          
          // Create arrow shape pointing in direction of backbone
          const shape = new THREE.Shape()
          shape.moveTo(-arrowWidth/2, 0)
          shape.lineTo(arrowWidth/2, 0)
          shape.lineTo(arrowWidth/2, length * 0.7)
          shape.lineTo(arrowWidth * 0.7, length * 0.7)
          shape.lineTo(0, length)
          shape.lineTo(-arrowWidth * 0.7, length * 0.7)
          shape.lineTo(-arrowWidth/2, length * 0.7)
          shape.lineTo(-arrowWidth/2, 0)

          const extrudeSettings = {
            steps: 1,
            depth: arrowHeight,
            bevelEnabled: false
          }

          const geometry = new THREE.ExtrudeGeometry(shape, extrudeSettings)
          const material = new THREE.MeshPhongMaterial({ 
            color,
            side: THREE.DoubleSide,
            shininess: 50
          })
          const arrow = new THREE.Mesh(geometry, material)
          
          // Position and orient the arrow
          arrow.position.set(curr.x, curr.y, curr.z)
          arrow.lookAt(next.x, next.y, next.z)
          scene.add(arrow)
        } else {
          // Coil/turn - simple tube
          const curve = new THREE.CatmullRomCurve3([
            new THREE.Vector3(curr.x, curr.y, curr.z),
            new THREE.Vector3(next.x, next.y, next.z)
          ])
          const tubeGeometry = new THREE.TubeGeometry(curve, 4, 0.2, 6, false)
          const material = new THREE.MeshPhongMaterial({ color })
          const tube = new THREE.Mesh(tubeGeometry, material)
          scene.add(tube)
        }
      }
    } else {
      // Ball & stick or stick mode
      atoms.forEach((atom, i) => {
        const color = heatmap
          ? new THREE.Color().setHSL((1 - atom.bFactor / 100) * 0.7, 1, 0.5)
          : new THREE.Color(elementColors[atom.element] || 0x808080)

        if (mode === 'sphere') {
          // Create sphere for atom
          const radius = atom.element === 'H' ? 0.3 : 0.5
          const sphereGeometry = new THREE.SphereGeometry(radius, 16, 16)
          const material = new THREE.MeshPhongMaterial({ color })
          const sphere = new THREE.Mesh(sphereGeometry, material)
          sphere.position.set(atom.x, atom.y, atom.z)
          scene.add(sphere)
        }

        // Create bonds (for both sphere and stick modes)
        atoms.forEach((otherAtom, j) => {
          if (i >= j) return // Avoid duplicate bonds
          
          const distance = Math.sqrt(
            Math.pow(atom.x - otherAtom.x, 2) +
            Math.pow(atom.y - otherAtom.y, 2) +
            Math.pow(atom.z - otherAtom.z, 2)
          )

          // Bond if atoms are close enough (typical bond lengths)
          if (distance < 2.0 && Math.abs(atom.residueNum - otherAtom.residueNum) <= 1) {
            const bondGeometry = new THREE.CylinderGeometry(0.1, 0.1, distance)
            const bondMaterial = new THREE.MeshPhongMaterial({ color: 0x808080 })
            const bond = new THREE.Mesh(bondGeometry, bondMaterial)
            
            // Position and orient the bond
            const midpoint = new THREE.Vector3(
              (atom.x + otherAtom.x) / 2,
              (atom.y + otherAtom.y) / 2,
              (atom.z + otherAtom.z) / 2
            )
            bond.position.copy(midpoint)
            bond.lookAt(otherAtom.x, otherAtom.y, otherAtom.z)
            bond.rotateX(Math.PI / 2)
            scene.add(bond)
          }
        })
      })
    }
  }

  const changeRenderMode = (mode: 'ribbon' | 'cartoon' | 'sphere' | 'stick') => {
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
        <div className="p-4 border-b border-gray-200 dark:border-gray-700 flex gap-2 items-center flex-wrap">
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Render Mode:</span>
          <button
            onClick={() => changeRenderMode('ribbon')}
            className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
              renderMode === 'ribbon'
                ? 'bg-purple-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300'
            }`}
          >
            Ribbon
          </button>
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
                ? 'bg-green-600 text-white'
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
          
          <div className="border-l border-gray-300 dark:border-gray-600 h-8 mx-2"></div>
          
          <button
            onClick={() => setShowHeatmap(!showHeatmap)}
            className={`px-4 py-2 rounded text-sm font-medium transition-colors ${
              showHeatmap
                ? 'bg-orange-600 text-white'
                : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300'
            }`}
            title="B-factor heatmap (thermal motion/flexibility)"
          >
            🔥 B-Factor Heatmap
          </button>
          
          <div className="ml-auto text-xs text-gray-500 dark:text-gray-400">
            💡 Use mouse to rotate, scroll to zoom
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
          <div className="flex items-center gap-6 text-sm flex-wrap">
            {renderMode === 'ribbon' || renderMode === 'cartoon' ? (
              <>
                <span className="font-medium text-gray-700 dark:text-gray-300">Secondary Structure:</span>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full" style={{background: '#ff4081'}}></div>
                  <span className="text-gray-600 dark:text-gray-400">α-Helix</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full" style={{background: '#ffd600'}}></div>
                  <span className="text-gray-600 dark:text-gray-400">β-Sheet</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full" style={{background: '#00bcd4'}}></div>
                  <span className="text-gray-600 dark:text-gray-400">β-Turn</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded-full bg-gray-400"></div>
                  <span className="text-gray-600 dark:text-gray-400">Random Coil</span>
                </div>
              </>
            ) : (
              <>
                <span className="font-medium text-gray-700 dark:text-gray-300">Element Colors:</span>
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
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}